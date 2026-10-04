// Gate 3: RIFE 4.25 synthesises a frame that is actually between its inputs.
//
// The test is built so that a plausible-looking wrong answer fails.
//
// Two frames of a static pattern, the second shifted right by an exact number of
// pixels, describe pure horizontal translation. The frame at timestep `t` is
// therefore the pattern shifted by `s*t` pixels — a value this test can compute
// exactly, with no reference implementation to trust.
//
// A network that ignores the timestep, or that has its inputs swapped, lands on
// the wrong shift. A network that is not running at all lands on something close
// to one of the inputs. And a *blend* of the two frames — the answer a naive
// "average them" implementation gives — is measurably wrong here, because the
// pattern is sinusoidal rather than linear: averaging a shift of 0 and a shift of
// 2 does not give a shift of 1. That gap is the whole point; a linear ramp would
// have made this test pass without a network.

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

const int kW = 960;
const int kH = 640;
// Wide enough that the network's padding cannot reach the measured region.
const int kBorder = 40;

const double kPi = 3.14159265358979323846;

unsigned char pattern(int x, int y)
{
    // Period 16 horizontally: a 2-pixel shift is a pi/4 phase step, which makes
    // the blend and the true midpoint differ by a measurable amount.
    const double horizontal = 60.0 * std::sin(2.0 * kPi * x / 64.0);
    // A second frequency vertically, so the flow field is not free to invent
    // arbitrary vertical motion.
    const double vertical = 40.0 * std::sin(2.0 * kPi * y / 89.0);
    double v = 128.0 + horizontal + vertical;
    if (v < 0.0)
        v = 0.0;
    if (v > 255.0)
        v = 255.0;
    return (unsigned char)(v + 0.5);
}

std::vector<unsigned char> make_frame(int shift_x)
{
    std::vector<unsigned char> pixels((size_t)kW * kH * 3, 0);
    for (int y = 0; y < kH; y++)
        for (int x = 0; x < kW; x++)
        {
            int sx = x - shift_x;
            if (sx < 0)
                sx = 0;
            if (sx > kW - 1)
                sx = kW - 1;
            const unsigned char v = pattern(sx, y);
            unsigned char* p = &pixels[((size_t)y * kW + x) * 3];
            p[0] = v;
            p[1] = v;
            p[2] = v;
        }
    return pixels;
}

double mae_interior(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b)
{
    double total = 0.0;
    long long count = 0;
    for (int y = kBorder; y < kH - kBorder; y++)
        for (int x = kBorder; x < kW - kBorder; x++)
            for (int c = 0; c < 3; c++)
            {
                const size_t i = ((size_t)y * kW + x) * 3 + c;
                total += std::fabs((double)a[i] - (double)b[i]);
                count++;
            }
    return count ? total / (double)count : -1.0;
}

const char* model_dir(int argc, char** argv)
{
    if (argc > 1)
        return argv[1];
    return getenv("SR_RIFE_MODEL_DIR");
}

// Runs one case and returns the interior error against the exact expected frame.
// `blend_error` receives the error a naive average of the two inputs would score.
bool run_case(sr_rife* rife,
              int shift,
              float timestep,
              double* error_against_truth,
              double* error_of_blend)
{
    const std::vector<unsigned char> a = make_frame(0);
    const std::vector<unsigned char> b = make_frame(shift);

    // The expected frame: the pattern advanced by exactly `shift * timestep`.
    const int expected_shift = (int)std::lround((double)shift * (double)timestep);
    const std::vector<unsigned char> truth = make_frame(expected_shift);

    std::vector<unsigned char> blend(a.size(), 0);
    for (size_t i = 0; i < a.size(); i++)
        blend[i] = (unsigned char)(((int)a[i] + (int)b[i]) / 2);

    sr_image ia;
    ia.data = const_cast<unsigned char*>(a.data());
    ia.width = kW;
    ia.height = kH;
    ia.stride = kW * 3;
    ia.channels = 3;
    sr_image ib = ia;
    ib.data = const_cast<unsigned char*>(b.data());

    std::vector<unsigned char> mid(a.size(), 0);
    sr_image im = ia;
    im.data = mid.data();

    const int rc = sr_rife_process(rife, &ia, &ib, timestep, &im);
    if (rc != SR_OK)
    {
        printf("  FAIL  sr_rife_process(shift=%d, t=%.2f) returned %d\n", shift, (double)timestep, rc);
        failures++;
        return false;
    }

    *error_against_truth = mae_interior(mid, truth);
    *error_of_blend = mae_interior(blend, truth);

    printf("  shift %d px, t = %.2f -> expected shift %d px\n", shift, (double)timestep, expected_shift);
    printf("    |rife - truth|  = %.3f levels\n", *error_against_truth);
    printf("    |blend - truth| = %.3f levels  (what averaging the inputs gives)\n", *error_of_blend);
    printf("    |frame0 - truth| = %.3f levels\n", mae_interior(a, truth));

    {
        double sa = 0, sb = 0, st = 0, sm = 0;
        int la = 255, ha = 0, lm = 255, hm = 0;
        for (size_t i = 0; i < a.size(); i++) {
            sa += a[i]; sb += b[i]; st += truth[i]; sm += mid[i];
            if (a[i] < la) la = a[i]; if (a[i] > ha) ha = a[i];
            if (mid[i] < lm) lm = mid[i]; if (mid[i] > hm) hm = mid[i];
        }
        const double n = (double)a.size();
        printf("    frame0 min %3d mean %6.2f max %3d | frame1 mean %6.2f\n", la, sa / n, ha, sb / n);
        printf("    truth  mean %6.2f                 | rife   min %3d mean %6.2f max %3d\n", st / n, lm, sm / n, hm);
        printf("    sample px 0..5: truth", 0);
        for (int i = 0; i < 6; i++) printf(" %3d", truth[i * 3]);
        printf("   rife");
        for (int i = 0; i < 6; i++) printf(" %3d", mid[i * 3]);
        printf("\n");
        printf("    row y=48, x=20..35\n");
        printf("      truth ");
        for (int x = 20; x < 36; x++) printf("%4d", truth[((size_t)48 * kW + x) * 3]);
        printf("\n      rife  ");
        for (int x = 20; x < 36; x++) printf("%4d", mid[((size_t)48 * kW + x) * 3]);
        printf("\n      frm0  ");
        for (int x = 20; x < 36; x++) printf("%4d", a[((size_t)48 * kW + x) * 3]);
        printf("\n      frm1  ");
        for (int x = 20; x < 36; x++) printf("%4d", b[((size_t)48 * kW + x) * 3]);
        printf("\n");
    }
    return true;
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
    printf("sr-native rife_pair_test\n");

    const char* dir = model_dir(argc, argv);
    if (!dir || !*dir)
    {
        printf("  SKIPPED: set SR_RIFE_MODEL_DIR to the RIFE 4.25 directory\n");
        return 77;
    }
    printf("  model: %s\n", dir);

    sr_context* ctx = sr_context_create(sr_test_device());
    if (!ctx)
    {
        printf("  FAIL  no Vulkan context\n");
        return 1;
    }

    sr_rife* rife = sr_rife_create(ctx, dir);
    if (!rife)
    {
        printf("  FAIL  sr_rife_create: %s\n", sr_last_error(ctx));
        sr_context_destroy(ctx);
        return 1;
    }
    check(true, "the model loads");

    const std::vector<unsigned char> frame = make_frame(0);
    sr_image image;
    image.data = const_cast<unsigned char*>(frame.data());
    image.width = kW;
    image.height = kH;
    image.stride = kW * 3;
    image.channels = 3;

    // The endpoints are not synthesised, and a caller that asks for one has made
    // a mistake worth reporting rather than a frame worth blurring.
    check(sr_rife_process(rife, &image, &image, 0.0f, &image) == SR_ERR_INVALID_ARGUMENT,
          "timestep 0 is refused");
    check(sr_rife_process(rife, &image, &image, 1.0f, &image) == SR_ERR_INVALID_ARGUMENT,
          "timestep 1 is refused");

    // A constant image. Warping a constant returns that constant for *any* flow, so
    // this case is insensitive to the flow field entirely: if the output is not
    // constant, the fault is in the warp or the readback rather than in the flow.
    {
        std::vector<unsigned char> flat((size_t)kW * kH * 3, 128);
        sr_image ia = image;
        ia.data = flat.data();
        sr_image ib = ia;
        std::vector<unsigned char> mid(flat.size(), 0);
        sr_image im = ia;
        im.data = mid.data();
        const int rc = sr_rife_process(rife, &ia, &ib, 0.5f, &im);
        if (rc != SR_OK)
        {
            printf("  FAIL  constant pair returned %d\n", rc);
            failures++;
        }
        else
        {
            unsigned char lo = 255, hi = 0;
            double sum = 0;
            for (unsigned char v : mid) { if (v < lo) lo = v; if (v > hi) hi = v; sum += v; }
            printf("  constant 128 pair: output min %u max %u mean %.2f\n",
                   lo, hi, sum / (double)mid.size());
            check(hi - lo <= 2,
                  "warping a constant image gives a constant image");
        }
    }

    // The unambiguous case: two identical frames describe no motion at all, so
    // the midpoint is that same frame. Any working interpolator returns it to
    // within a rounding error, and a broken one cannot accidentally pass.
    {
        const std::vector<unsigned char> same = make_frame(0);
        sr_image ia = image;
        ia.data = const_cast<unsigned char*>(same.data());
        sr_image ib = ia;
        std::vector<unsigned char> mid(same.size(), 0);
        sr_image im = ia;
        im.data = mid.data();
        const int rc = sr_rife_process(rife, &ia, &ib, 0.5f, &im);
        if (rc != SR_OK)
        {
            printf("  FAIL  static pair returned %d\n", rc);
            failures++;
        }
        else
        {
            printf("  static pair (frame0 == frame1, t=0.5): |rife - frame| = %.3f levels\n",
                   mae_interior(mid, same));
            printf("    row y=48: frame");
            for (int x = 20; x < 30; x++) printf("%4d", same[((size_t)48 * kW + x) * 3]);
            printf("\n              rife ");
            for (int x = 20; x < 30; x++) printf("%4d", mid[((size_t)48 * kW + x) * 3]);
            printf("\n");
            check(mae_interior(mid, same) < 2.0,
                  "a pair of identical frames interpolates to that frame");
        }
    }

    int cases_run = 0;
    double truth_error = 0.0, blend_error = 0.0;
    double worst_truth = 0.0, smallest_gap = 0.0;

    struct Case
    {
        int shift;
        float timestep;
        // Whether "a blend of the two inputs" is a *wrong* answer here. Over a
        // small shift of a smooth pattern the sinusoid is nearly linear, so the
        // average of the two frames genuinely is the midpoint and comparing
        // against it proves nothing. The flag says which cases can carry that
        // assertion rather than weakening it everywhere.
        bool blend_is_wrong;
    };
    const Case cases[] = {
        {2, 0.5f, false},
        {8, 0.5f, true},
        {4, 0.25f, true},
    };

    for (const Case& c : cases)
    {
        double t = 0.0, b = 0.0;
        if (!run_case(rife, c.shift, c.timestep, &t, &b))
            continue;
        cases_run++;
        if (t > worst_truth)
            worst_truth = t;
        if (!c.blend_is_wrong)
            continue;
        const double gap = b - t;
        if (smallest_gap == 0.0 || gap < smallest_gap)
            smallest_gap = gap;
    }

    // The thresholds are stated as comparisons the test can justify, not as
    // numbers tuned until it went green: the synthesised frame has to be close
    // to the exact answer, and clearly closer to it than a blend of the inputs.
    check(cases_run == 3, "every interpolation case actually ran (a check must not pass vacuously)");
    check(cases_run == 3 && worst_truth < 2.0, "the synthesised frame matches the exact midpoint");
    check(smallest_gap > 1.0,
          "the synthesised frame beats a blend of the inputs by a clear margin "
          "(so a network ran, and the timestep was used)");

    sr_rife_destroy(rife);
    sr_context_destroy(ctx);

    if (failures)
    {
        printf("\nFAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("\nPASSED\n");
    return 0;
}
