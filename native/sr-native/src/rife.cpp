// RIFE 4.25 interpolation.
//
// The network is a single flownet with three inputs and one output:
//
//   in0  w x h x 3   the previous frame, planar float RGB in 0..255
//   in1  w x h x 3   the next frame, same encoding
//   in2  w x h x 1   the timestep, replicated across the whole plane
//   out0 w x h x 3   the synthesised frame, same encoding as the inputs
//
// `w` and `h` here are the *padded* dimensions. The reference implementation
// pads to a multiple of 32 and notes that newer models need 64; padding to 64 is
// harmless for the models that only need 32, so that is what this uses. The
// border is replicated rather than zero-filled, because a zero border is a hard
// edge that the flow estimator will happily invent motion around, and the
// coarse-to-fine pyramid carries that inward.
//
// The 0..255 range is not a choice. The reference multiplies its 0..1 inputs by
// 255 before the first layer and divides the output by 255 on the way out, so
// the weights were fitted in 0..255. Feeding 0..1 produces a plausible-looking
// but washed-out frame, which is the kind of error that survives a smoke test.

#include "sr_internal.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "net.h"
#include "rife_ops.h"

// The one custom layer this model needs. Everything else in the .param is a
// builtin ncnn layer; `rife.Warp` appears 18 times and is the bilinear warp
// defined in rife/warp.cpp.
DEFINE_LAYER_CREATOR(Warp)

namespace {

// Longest edge the network is asked to handle in one call. RIFE is fully
// convolutional and has no tile-size requirement, but a 4K frame at fp32 is
// ~100 MB of activations per pyramid level, so very large frames are processed
// in horizontal bands. Bands rather than a grid because the flow estimator's
// vertical receptive field is the one that would show seams.
const int kBandRows = 2160;

} // namespace

struct sr_rife {
    ncnn::Net net;
    std::string model_dir;
    int padding = 64;
};

sr_rife* sr_rife_create(sr_context* ctx, const char* model_dir)
{
    if (!ctx)
        return nullptr;
    if (!model_dir || !*model_dir)
    {
        sr_set_error(ctx, "sr_rife_create: no model directory given");
        return nullptr;
    }

    sr_ensure_instance();

    sr_rife* rife = new sr_rife();
    rife->model_dir = model_dir;

    rife->net.opt.use_vulkan_compute = true;

    // Exactly the option set the reference implementation uses (RIFE::load):
    //
    //     opt.use_fp16_packed   = vkdev ? true : false;
    //     opt.use_fp16_storage  = vkdev ? true : false;
    //     opt.use_fp16_arithmetic = false;
    //     opt.use_int8_storage  = false;
    //
    // Earlier attempts pinned packing and fp16 storage OFF, which is not what the
    // models were validated with, and they were only ever measured at a frame size
    // now known to suppress the network. This is the reference configuration.
    rife->net.opt.use_fp16_packed = true;
    rife->net.opt.use_fp16_storage = true;
    rife->net.opt.use_fp16_arithmetic = false;
    rife->net.opt.use_int8_storage = false;
    rife->net.set_vulkan_device(ctx->device_index);

    if (rife->net.register_custom_layer("rife.Warp", Warp_layer_creator) != 0)
    {
        sr_set_error(ctx, "sr_rife_create: could not register the rife.Warp layer");
        delete rife;
        return nullptr;
    }

    const std::string param = std::string(model_dir) + "/flownet.param";
    const std::string bin = std::string(model_dir) + "/flownet.bin";

    if (rife->net.load_param(param.c_str()) != 0)
    {
        sr_set_error(ctx, "sr_rife_create: could not read %s", param.c_str());
        delete rife;
        return nullptr;
    }
    if (rife->net.load_model(bin.c_str()) != 0)
    {
        sr_set_error(ctx, "sr_rife_create: could not read %s", bin.c_str());
        delete rife;
        return nullptr;
    }

    return rife;
}

void sr_rife_destroy(sr_rife* rife)
{
    delete rife;
}

namespace {

// One padded input plane: `wp x hp x 3`, edge-replicated, scaled to 0..255.
ncnn::Mat build_padded_frame(const sr_image* image, int wp, int hp)
{
    const int w = image->width;
    const int h = image->height;
    ncnn::Mat mat(wp, hp, 3);
    if (mat.empty())
        return mat;

    for (int y = 0; y < hp; y++)
    {
        const int sy = std::min(y, h - 1);
        const unsigned char* row = image->data + (size_t)sy * (size_t)image->stride;
        float* r = sr_channel(mat, 0) + (size_t)y * (size_t)wp;
        float* g = sr_channel(mat, 1) + (size_t)y * (size_t)wp;
        float* b = sr_channel(mat, 2) + (size_t)y * (size_t)wp;
        for (int x = 0; x < wp; x++)
        {
            const int sx = std::min(x, w - 1);
            r[x] = (float)row[sx * 3 + 0];
            g[x] = (float)row[sx * 3 + 1];
            b[x] = (float)row[sx * 3 + 2];
        }
    }
    return mat;
}

} // namespace

int sr_rife_process(sr_rife* rife,
                    const sr_image* prev,
                    const sr_image* next,
                    float timestep,
                    sr_image* out)
{
    if (!rife)
        return SR_ERR_INVALID_ARGUMENT;
    if (!sr_image_is_valid(prev) || !sr_image_is_valid(next) || !sr_image_is_valid(out))
        return SR_ERR_INVALID_ARGUMENT;
    if (prev->width != next->width || prev->height != next->height)
        return SR_ERR_INVALID_ARGUMENT;
    if (out->width != prev->width || out->height != prev->height)
        return SR_ERR_INVALID_ARGUMENT;
    // The endpoints are not synthesised: a caller that wants frame 0 already has
    // frame 0, and asking a network to reproduce an input it was given is a
    // silent route to a blurred frame.
    if (!(timestep > 0.0f && timestep < 1.0f))
        return SR_ERR_INVALID_ARGUMENT;

    const int w = prev->width;
    const int h = prev->height;

    for (int band_y = 0; band_y < h; band_y += kBandRows)
    {
        const int band_h = std::min(kBandRows, h - band_y);

        const int p = rife->padding;
        const int wp = (w + p - 1) / p * p;
        const int hp = (band_h + p - 1) / p * p;

        // The band's rows are contiguous within the source image, so a view
        // offset by the band's first row is all the padding builder needs.
        sr_image band_prev = *prev;
        sr_image band_next = *next;
        band_prev.data = prev->data + (size_t)band_y * (size_t)prev->stride;
        band_prev.height = band_h;
        band_next.data = next->data + (size_t)band_y * (size_t)next->stride;
        band_next.height = band_h;

        ncnn::Mat in0 = build_padded_frame(&band_prev, wp, hp);
        ncnn::Mat in1 = build_padded_frame(&band_next, wp, hp);
        ncnn::Mat in2(wp, hp, 1);
        if (in0.empty() || in1.empty() || in2.empty())
            return SR_ERR_OUT_OF_MEMORY;

        in2.fill(timestep);

        ncnn::Extractor ex = rife->net.create_extractor();
        ex.input("in0", in0);
        ex.input("in1", in1);
        ex.input("in2", in2);

        ncnn::Mat out_mat;
        if (ex.extract("out0", out_mat) != 0 || out_mat.empty())
            return SR_ERR_INTERNAL;
        if (out_mat.w < w || out_mat.h < band_h)
            return SR_ERR_INTERNAL;

        sr_planar_to_rgb8_region(out_mat,
                                 out->data,
                                 out->stride,
                                 0,
                                 band_y,
                                 0,
                                 0,
                                 w,
                                 band_h);
    }

    return SR_OK;
}
