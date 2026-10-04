// Gate 2: Real-ESRGAN x4plus runs end to end on a real frame.
//
// The assertions are chosen to fail for the failure modes that actually happen,
// not to restate the implementation:
//
//  * a black or washed-out frame is the documented ncnn failure on some machines,
//    so mean luminance is compared against the *input's*, not against a constant;
//  * a resample masquerading as restoration is caught by comparing the output
//    against a bilinear 4x upscale of the same input — if the two agree, no
//    network ran, whatever the log said;
//  * tiled output is compared against untiled output, because a tiling bug shows
//    up as seams and nowhere else.

#include "sr_native.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what)
{
    printf("  %s  %s\n", condition ? "ok  " : "FAIL", what);
    if (!condition)
        failures++;
}

const int kW = 96;
const int kH = 64;

// A pattern with real detail at several scales: fine stripes the network can
// sharpen, a coarse ramp it must not flatten, and a hard edge it must not ring.
unsigned char pattern(int x, int y)
{
    const double fine = 40.0 * std::sin(x * 3.14159265 / 2.0);
    const double medium = 35.0 * std::sin(y * 3.14159265 / 5.0);
    const double coarse = 40.0 * (double)x / (double)kW;
    double v = 128.0 + fine + medium + coarse;
    if (x > kW * 2 / 3)
        v -= 45.0;
    if (v < 0.0)
        v = 0.0;
    if (v > 255.0)
        v = 255.0;
    return (unsigned char)(v + 0.5);
}

void build_input(std::vector<unsigned char>& pixels)
{
    pixels.assign((size_t)kW * kH * 3, 0);
    for (int y = 0; y < kH; y++)
        for (int x = 0; x < kW; x++)
        {
            const unsigned char v = pattern(x, y);
            unsigned char* p = &pixels[((size_t)y * kW + x) * 3];
            p[0] = v;
            p[1] = (unsigned char)(255 - v);
            p[2] = (unsigned char)((v / 2) + 60);
        }
}

double mean_luma(const unsigned char* data, int stride, int w, int h)
{
    double total = 0.0;
    for (int y = 0; y < h; y++)
    {
        const unsigned char* row = data + (size_t)y * stride;
        for (int x = 0; x < w; x++)
            total += 0.299 * row[x * 3] + 0.587 * row[x * 3 + 1] + 0.114 * row[x * 3 + 2];
    }
    return total / (double)(w * h);
}

double mean_abs_diff(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b)
{
    if (a.size() != b.size() || a.empty())
        return -1.0;
    double total = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        total += std::fabs((double)a[i] - (double)b[i]);
    return total / (double)a.size();
}

// Bilinear 4x upscale: the thing restoration must not be.
std::vector<unsigned char> bilinear4x(const std::vector<unsigned char>& src)
{
    std::vector<unsigned char> dst((size_t)kW * 4 * kH * 4 * 3, 0);
    for (int y = 0; y < kH * 4; y++)
    {
        const double sy = (double)y / 4.0;
        const int y0 = (int)sy;
        const int y1 = y0 + 1 < kH ? y0 + 1 : y0;
        const double fy = sy - y0;
        for (int x = 0; x < kW * 4; x++)
        {
            const double sx = (double)x / 4.0;
            const int x0 = (int)sx;
            const int x1 = x0 + 1 < kW ? x0 + 1 : x0;
            const double fx = sx - x0;
            for (int c = 0; c < 3; c++)
            {
                const double v00 = src[((size_t)y0 * kW + x0) * 3 + c];
                const double v01 = src[((size_t)y0 * kW + x1) * 3 + c];
                const double v10 = src[((size_t)y1 * kW + x0) * 3 + c];
                const double v11 = src[((size_t)y1 * kW + x1) * 3 + c];
                const double top = v00 * (1 - fx) + v01 * fx;
                const double bot = v10 * (1 - fx) + v11 * fx;
                dst[((size_t)y * kW * 4 + x) * 3 + c] = (unsigned char)(top * (1 - fy) + bot * fy + 0.5);
            }
        }
    }
    return dst;
}

const char* model_dir(int argc, char** argv)
{
    if (argc > 1)
        return argv[1];
    return getenv("SR_RESTORE_MODEL_DIR");
}

} // namespace

// Which Vulkan device to run on. Defaults to 0; `SR_DEVICE=1` reaches the second
// one, which is how the same tests are run against another vendor's driver without
// a second machine.
static int sr_test_device()
{
    const char* value = getenv("SR_DEVICE");
    if (!value || !*value)
        return 0;
    return atoi(value);
}
int main(int argc, char** argv)
{
    printf("sr-native restore_image_test\n");

    const char* dir = model_dir(argc, argv);
    if (!dir || !*dir)
    {
        printf("  SKIPPED: set SR_RESTORE_MODEL_DIR to the Real-ESRGAN x4plus directory\n");
        return 77;
    }
    printf("  model: %s\n", dir);

    sr_context* ctx = sr_context_create(sr_test_device());
    if (!ctx)
    {
        printf("  FAIL  no Vulkan context\n");
        return 1;
    }

    sr_restorer* restorer = sr_restorer_create(ctx, dir);
    if (!restorer)
    {
        printf("  FAIL  sr_restorer_create: %s\n", sr_last_error(ctx));
        sr_context_destroy(ctx);
        return 1;
    }
    check(true, "the model loads");
    check(sr_restorer_create(ctx, "Z:/does/not/exist") == nullptr,
          "a missing model directory is refused, not silently accepted");
    check(sr_last_error(ctx)[0] != '\0', "and it says why");

    std::vector<unsigned char> input;
    build_input(input);

    sr_image in;
    in.data = input.data();
    in.width = kW;
    in.height = kH;
    in.stride = kW * 3;
    in.channels = 3;

    std::vector<unsigned char> output((size_t)kW * 4 * kH * 4 * 3, 0);
    sr_image out = in;
    out.data = output.data();
    out.width = kW * 4;
    out.height = kH * 4;
    out.stride = kW * 4 * 3;

    int32_t ow = 0, oh = 0;
    check(sr_restorer_output_size(restorer, kW, kH, 4, &ow, &oh) == SR_OK && ow == kW * 4 && oh == kH * 4,
          "output geometry is reported");

    check(sr_restorer_process(restorer, &in, &out, 4, 0) == SR_OK, "an untiled frame is restored");
    const std::vector<unsigned char> untiled = output;

    // --- the frame is a picture, not a black or constant rectangle -----------
    const double in_luma = mean_luma(in.data, in.stride, kW, kH);
    const double out_luma = mean_luma(out.data, out.stride, out.width, out.height);
    printf("  luminance: input %.1f -> output %.1f\n", in_luma, out_luma);
    check(out_luma > in_luma * 0.75 && out_luma < in_luma * 1.25,
          "mean luminance is preserved (a black frame is the documented ncnn failure)");

    {
        unsigned char lo = 255, hi = 0;
        for (unsigned char v : output)
        {
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        printf("  output range: %u..%u\n", lo, hi);
        check(hi - lo > 40, "the output is not a constant plane");
    }

    // --- a network ran, rather than a resample ------------------------------
    const std::vector<unsigned char> bilinear = bilinear4x(input);
    const double vs_bilinear = mean_abs_diff(output, bilinear);
    printf("  mean |restored - bilinear4x| = %.2f levels\n", vs_bilinear);
    check(vs_bilinear > 1.5,
          "the result differs from a bilinear upscale (restoration, not resampling)");

    // --- tiling does not introduce seams ------------------------------------
    std::vector<unsigned char> tiled(output.size(), 0);
    sr_image tiled_out = out;
    tiled_out.data = tiled.data();
    check(sr_restorer_process(restorer, &in, &tiled_out, 4, 32) == SR_OK, "a tiled frame is restored");
    const double seam = mean_abs_diff(tiled, untiled);
    printf("  mean |tiled - untiled| = %.2f levels\n", seam);
    check(seam < 8.0, "tiling agrees with the untiled result (no seam)");

    // A scale the model does not support must be refused rather than guessed at.
    check(sr_restorer_process(restorer, &in, &out, 2, 0) == SR_ERR_INVALID_ARGUMENT,
          "an unsupported scale is refused");

    sr_restorer_destroy(restorer);
    sr_context_destroy(ctx);

    if (failures)
    {
        printf("\nFAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("\nPASSED\n");
    return 0;
}
