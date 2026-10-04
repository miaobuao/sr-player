// Real-ESRGAN x4plus restoration.
//
// The network takes `data` and produces `output`, both planar float RGB, and it
// is fixed at 4x: the model contains its own pixel-shuffle layers and there is
// no 2x or 3x path inside it. The `scale` argument exists so a future
// `sr_restorer_*` implementation can offer other factors without the caller
// changing, and so a mismatch is an error here rather than a wrong-sized frame
// three stages later.
//
// The input range is 0..1, unlike RIFE's 0..255. That is not an inconsistency in
// this file: it is what each model was exported for, and it is exactly the kind
// of detail that a "both networks take images" abstraction would have hidden.
//
// Tiling follows the reference implementation: tiles of `tile` pixels are
// extracted with `prepadding` pixels of context on every side, and the result is
// cropped back by `prepadding * scale`. Without the context the convolutions at
// a tile edge see a zero border and produce a visible seam — the artefact the QC
// stage is asked to catch, and cheaper to avoid than to detect.

#include "sr_internal.h"

#include <algorithm>
#include <string>
#include <vector>

#include "net.h"

namespace {

// The reference uses 10. It must be at least the network's receptive-field
// radius divided by its depth, or the seam comes back.
const int kPrepadding = 10;

// A tile edge of 0 means "one tile", which is right until the frame is large
// enough that the activation footprint matters. 1920 keeps a 1080p frame whole
// and splits 4K into four.
const int kAutoTile = 1920;

} // namespace

struct sr_restorer {
    ncnn::Net net;
    std::string model_dir;
    int model_scale = 4;
};

sr_restorer* sr_restorer_create(sr_context* ctx, const char* model_dir)
{
    if (!ctx)
        return nullptr;
    if (!model_dir || !*model_dir)
    {
        sr_set_error(ctx, "sr_restorer_create: no model directory given");
        return nullptr;
    }

    sr_ensure_instance();

    sr_restorer* restorer = new sr_restorer();
    restorer->model_dir = model_dir;

    restorer->net.opt.use_vulkan_compute = true;
    restorer->net.opt.num_threads = 1;
    restorer->net.set_vulkan_device(ctx->device_index);

    const std::string param = std::string(model_dir) + "/model.param";
    const std::string bin = std::string(model_dir) + "/model.bin";

    if (restorer->net.load_param(param.c_str()) != 0)
    {
        sr_set_error(ctx,
                     "sr_restorer_create: could not read %s (is the Real-ESRGAN x4plus model "
                     "installed, and is it named model.param/model.bin?)",
                     param.c_str());
        delete restorer;
        return nullptr;
    }
    if (restorer->net.load_model(bin.c_str()) != 0)
    {
        sr_set_error(ctx, "sr_restorer_create: could not read %s", bin.c_str());
        delete restorer;
        return nullptr;
    }

    return restorer;
}

void sr_restorer_destroy(sr_restorer* restorer)
{
    delete restorer;
}

int32_t sr_restorer_output_size(const sr_restorer* restorer,
                                int32_t in_width,
                                int32_t in_height,
                                int32_t scale,
                                int32_t* out_width,
                                int32_t* out_height)
{
    if (!restorer || !out_width || !out_height)
        return SR_ERR_INVALID_ARGUMENT;
    if (in_width <= 0 || in_height <= 0)
        return SR_ERR_INVALID_ARGUMENT;
    if (scale != restorer->model_scale)
        return SR_ERR_INVALID_ARGUMENT;

    *out_width = in_width * scale;
    *out_height = in_height * scale;
    return SR_OK;
}

int sr_restorer_process(sr_restorer* restorer,
                        const sr_image* in,
                        sr_image* out,
                        int32_t scale,
                        int32_t tile)
{
    if (!restorer)
        return SR_ERR_INVALID_ARGUMENT;
    if (!sr_image_is_valid(in) || !sr_image_is_valid(out))
        return SR_ERR_INVALID_ARGUMENT;
    if (scale != restorer->model_scale)
        return SR_ERR_INVALID_ARGUMENT;
    if (out->width != in->width * scale || out->height != in->height * scale)
        return SR_ERR_INVALID_ARGUMENT;
    if (tile != 0 && tile < 32)
        return SR_ERR_INVALID_ARGUMENT;

    const int w = in->width;
    const int h = in->height;
    const int tile_size = tile > 0 ? tile : kAutoTile;
    const int pre = kPrepadding;

    for (int ty = 0; ty < h; ty += tile_size)
    {
        const int th = std::min(tile_size, h - ty);
        for (int tx = 0; tx < w; tx += tile_size)
        {
            const int tw = std::min(tile_size, w - tx);

            // The region actually fed to the network: the tile plus context,
            // clipped to the frame.
            const int sx0 = std::max(tx - pre, 0);
            const int sy0 = std::max(ty - pre, 0);
            const int sx1 = std::min(tx + tw + pre, w);
            const int sy1 = std::min(ty + th + pre, h);
            const int sw = sx1 - sx0;
            const int sh = sy1 - sy0;

            ncnn::Mat input(sw, sh, 3);
            if (input.empty())
                return SR_ERR_OUT_OF_MEMORY;

            for (int y = 0; y < sh; y++)
            {
                const unsigned char* row =
                    in->data + (size_t)(sy0 + y) * (size_t)in->stride + (size_t)sx0 * 3;
                float* r = sr_channel(input, 0) + (size_t)y * (size_t)sw;
                float* g = sr_channel(input, 1) + (size_t)y * (size_t)sw;
                float* b = sr_channel(input, 2) + (size_t)y * (size_t)sw;
                for (int x = 0; x < sw; x++)
                {
                    r[x] = (float)row[x * 3 + 0] * (1.0f / 255.0f);
                    g[x] = (float)row[x * 3 + 1] * (1.0f / 255.0f);
                    b[x] = (float)row[x * 3 + 2] * (1.0f / 255.0f);
                }
            }

            ncnn::Extractor ex = restorer->net.create_extractor();
            ex.input("data", input);

            ncnn::Mat output;
            if (ex.extract("output", output) != 0 || output.empty())
                return SR_ERR_INTERNAL;
            if (output.w < sw * scale || output.h < sh * scale)
                return SR_ERR_INTERNAL;

            // Within the network's output, the tile's own pixels start after the
            // context that was added on the top/left, and the context on the
            // bottom/right is simply not copied.
            const int crop_x = (tx - sx0) * scale;
            const int crop_y = (ty - sy0) * scale;

            for (int y = 0; y < th * scale; y++)
            {
                unsigned char* dst = out->data + (size_t)(ty * scale + y) * (size_t)out->stride +
                                     (size_t)(tx * scale) * 3;
                const float* r =
                    sr_channel(output, 0) + (size_t)(crop_y + y) * (size_t)output.w + crop_x;
                const float* g =
                    sr_channel(output, 1) + (size_t)(crop_y + y) * (size_t)output.w + crop_x;
                const float* b =
                    sr_channel(output, 2) + (size_t)(crop_y + y) * (size_t)output.w + crop_x;
                for (int x = 0; x < tw * scale; x++)
                {
                    // 0..1 back to 0..255, clamped: a network asked for a
                    // slightly out-of-range value must not wrap around.
                    dst[x * 3 + 0] = sr_clamp_u8(r[x] * 255.0f);
                    dst[x * 3 + 1] = sr_clamp_u8(g[x] * 255.0f);
                    dst[x * 3 + 2] = sr_clamp_u8(b[x] * 255.0f);
                }
            }
        }
    }

    return SR_OK;
}
