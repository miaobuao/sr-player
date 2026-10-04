// Unit check on the one piece of custom code in the runtime.
//
// Everything else in the RIFE network is a builtin ncnn layer, so if the
// synthesised frame comes out as high-frequency noise with the right mean and
// range — which is what it does — the fault is either the weights or this layer.
// This test removes the layer from suspicion, or convicts it, without a GPU and
// without a model file.
//
// The operation is simple enough to have an exact expected answer: for every
// output pixel, sample the image bilinearly at (x + flow_x, y + flow_y), with
// coordinates clamped to the image. A constant flow of (+1, 0) is therefore a
// one-pixel shift, and a flow of (+0.5, 0) is a half-pixel blend that a nearest
// neighbour implementation cannot fake.

#include "rife_ops.h"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;

void check(bool condition, const char* what)
{
    printf("  %s  %s\n", condition ? "ok  " : "FAIL", what);
    if (!condition)
        failures++;
}

const int kW = 16;
const int kH = 8;

float source(int x, int y)
{
    // Non-linear in x so a wrong sample position cannot coincide with the right
    // one, and distinguishable per row so a transposed index is visible.
    return (float)(x * x + y * 3);
}

bool run_case(float flow_x, float flow_y, const char* label)
{
    ncnn::Mat image(kW, kH, 1);
    ncnn::Mat flow(kW, kH, 2);
    if (image.empty() || flow.empty())
    {
        printf("  FAIL  could not allocate\n");
        failures++;
        return false;
    }

    for (int y = 0; y < kH; y++)
    {
        float* row = image.channel(0);
        for (int x = 0; x < kW; x++)
            row[y * kW + x] = source(x, y);
    }
    for (int y = 0; y < kH; y++)
    {
        float* fx = flow.channel(0);
        float* fy = flow.channel(1);
        for (int x = 0; x < kW; x++)
        {
            fx[y * kW + x] = flow_x;
            fy[y * kW + x] = flow_y;
        }
    }

    Warp warp;
    ncnn::Option opt;
    opt.num_threads = 1;

    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = image;
    bottoms[1] = flow;
    std::vector<ncnn::Mat> tops(1);

    if (warp.forward(bottoms, tops, opt) != 0)
    {
        printf("  FAIL  %s: forward returned non-zero\n", label);
        failures++;
        return false;
    }

    const ncnn::Mat& out = tops[0];
    if (out.w != kW || out.h != kH || out.c != 1)
    {
        printf("  FAIL  %s: output is %dx%dx%d, expected %dx%dx1\n", label, out.w, out.h, out.c, kW, kH);
        failures++;
        return false;
    }

    // Bilinear sample of the source at (x + flow_x, y + flow_y), clamped.
    const float* out_row0 = (const float*)out.channel(0);
    double worst = 0.0;
    int worst_x = -1, worst_y = -1;
    for (int y = 0; y < kH; y++)
    {
        for (int x = 0; x < kW; x++)
        {
            const float sx = x + flow_x;
            const float sy = y + flow_y;
            int x0 = (int)std::floor(sx);
            int y0 = (int)std::floor(sy);
            const float ax = sx - (float)x0;
            const float ay = sy - (float)y0;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            x0 = x0 < 0 ? 0 : (x0 > kW - 1 ? kW - 1 : x0);
            y0 = y0 < 0 ? 0 : (y0 > kH - 1 ? kH - 1 : y0);
            x1 = x1 < 0 ? 0 : (x1 > kW - 1 ? kW - 1 : x1);
            y1 = y1 < 0 ? 0 : (y1 > kH - 1 ? kH - 1 : y1);

            const float v0 = source(x0, y0);
            const float v1 = source(x1, y0);
            const float v2 = source(x0, y1);
            const float v3 = source(x1, y1);
            const float expected = (v0 * (1 - ax) + v1 * ax) * (1 - ay) + (v2 * (1 - ax) + v3 * ax) * ay;

            const double diff = std::fabs((double)out_row0[y * kW + x] - (double)expected);
            if (diff > worst)
            {
                worst = diff;
                worst_x = x;
                worst_y = y;
            }
        }
    }

    printf("    %-22s worst |out - expected| = %.4f at (%d,%d)\n", label, worst, worst_x, worst_y);
    return worst < 0.01;
}

} // namespace

int main()
{
    printf("sr-native warp_unit_test (the vendored rife.Warp layer)\n");

    check(run_case(0.0f, 0.0f, "identity"), "a zero flow returns the image unchanged");
    check(run_case(1.0f, 0.0f, "integer shift +1x"), "an integer flow is an exact shift");
    check(run_case(0.0f, 1.0f, "integer shift +1y"), "the flow's second channel is the y axis");
    check(run_case(0.5f, 0.0f, "half-pixel +0.5x"), "a fractional flow interpolates bilinearly");
    check(run_case(-1.0f, 0.0f, "negative shift -1x"), "a negative flow shifts the other way");

    if (failures)
    {
        printf("\nFAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("\nPASSED\n");
    return 0;
}
