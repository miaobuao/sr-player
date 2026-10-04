// Shared internals of sr-native. Not installed; nothing outside src/ includes it.

#ifndef SR_INTERNAL_H
#define SR_INTERNAL_H

#include "sr_native.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>

#include "mat.h"

// The opaque handle from the public header, defined here so every translation
// unit agrees on it.
struct sr_context {
    int device_index = -1;
    std::string last_error;
};

// Creates ncnn's process-wide Vulkan instance on first use.
//
// ncnn exposes `create_gpu_instance()` / `destroy_gpu_instance()` without any
// reference counting, so calling the first twice leaks and calling the second
// while a net is alive is a use-after-free. One `std::call_once` removes both
// possibilities; teardown happens at process exit, which is what ncnn's own
// documentation asks for.
void sr_ensure_instance();

void sr_set_error(sr_context* ctx, const char* format, ...);

// Records the message and returns `code`, so a caller can `return sr_fail(...)`.
int sr_fail(sr_context* ctx, int code, const char* format, ...);

inline unsigned char sr_clamp_u8(float value)
{
    // NaN must not become 0 through a cast, so this deliberately does not use
    // std::clamp alone: `value < 0.f` is false for NaN and `value > 255.f` is
    // false too, and the pair would fall through to a cast of NaN.
    if (!(value > 0.0f))
        return 0;
    if (value > 255.0f)
        return 255;
    return (unsigned char)(value + 0.5f);
}

// `ncnn::Mat::channel()` returns a Mat view, and ncnn's implicit pointer
// conversion does not survive being used in arithmetic, so the data pointer is
// taken explicitly. Every Mat this project touches has elempack 1, where a
// channel view's `data` is exactly that channel's first element.
inline float* sr_channel(ncnn::Mat& mat, int q)
{
    return (float*)mat.channel(q).data;
}

inline const float* sr_channel(const ncnn::Mat& mat, int q)
{
    return (const float*)mat.channel(q).data;
}

// Writes the `w x h` region of a planar float Mat whose origin is (src_x,
// src_y) into `dst` at (dst_x, dst_y), clamped to 8-bit and honouring stride.
//
// `scale` maps the model's own value range onto 0..255, and it is a parameter
// rather than a constant because the two networks here disagree: Real-ESRGAN
// emits 0..1, RIFE emits 0..1 as well but its preprocessing divides by 255 on the
// way in, so feeding it 0..255 is a 255x overdrive. Getting this wrong in either
// direction produces a plausible-looking failure -- a saturated frame or a black
// one -- rather than an error, which is why it is spelled out at every call site.
inline void sr_planar_to_rgb8_region(const ncnn::Mat& mat,
                                     unsigned char* dst,
                                     int dst_stride,
                                     int dst_x,
                                     int dst_y,
                                     int src_x,
                                     int src_y,
                                     int w,
                                     int h,
                                     float scale)
{
    const int mat_w = mat.w;
    const float* plane_r = sr_channel(mat, 0);
    const float* plane_g = sr_channel(mat, 1);
    const float* plane_b = sr_channel(mat, 2);

    for (int y = 0; y < h; y++)
    {
        unsigned char* out = dst + (size_t)(dst_y + y) * (size_t)dst_stride + (size_t)dst_x * 3;
        const float* r = plane_r + (size_t)(src_y + y) * (size_t)mat_w + src_x;
        const float* g = plane_g + (size_t)(src_y + y) * (size_t)mat_w + src_x;
        const float* b = plane_b + (size_t)(src_y + y) * (size_t)mat_w + src_x;
        for (int x = 0; x < w; x++)
        {
            out[x * 3 + 0] = sr_clamp_u8(r[x] * scale);
            out[x * 3 + 1] = sr_clamp_u8(g[x] * scale);
            out[x * 3 + 2] = sr_clamp_u8(b[x] * scale);
        }
    }
}

// True when an image view is usable: non-null, positive, RGB, and with a stride
// that can actually hold a row. Checked at every entry point, because a caller
// that gets this wrong would otherwise corrupt memory rather than get an error.
inline bool sr_image_is_valid(const sr_image* image)
{
    if (!image || !image->data)
        return false;
    if (image->width <= 0 || image->height <= 0)
        return false;
    if (image->channels != 3)
        return false;
    if (image->stride < image->width * 3)
        return false;
    return true;
}

#endif // SR_INTERNAL_H
