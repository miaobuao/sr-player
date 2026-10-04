/*
 * sr_infer.h — vendor-neutral inference plugin ABI, version 1.
 *
 * A plugin is a shared library that exports the three functions below. The
 * engine never links against a vendor SDK: it loads this interface and asks what
 * the plugin can do. The same ABI can be backed by Vulkan compute, DirectML,
 * OpenVINO, MIGraphX, CUDA/TensorRT or a plain CPU kernel.
 *
 * Build a plugin:
 *   Windows:  cl /LD my_backend.c /Fe:sr_infer.dll
 *   Linux:    cc -shared -fPIC -o libsr_infer.so my_backend.c
 *   Rust:     see crates/sr-infer-plugin-example (a working reference)
 *
 * Install it next to the executable, in ./plugins/, or point SR_INFER_PLUGIN at
 * it. If no plugin is present the engine uses its deterministic FFmpeg path, so
 * a plugin is an accelerator and never a requirement.
 */

#ifndef SR_INFER_H
#define SR_INFER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SR_INFER_ABI_VERSION 1u

#define SR_OK                   0
#define SR_ERR_UNSUPPORTED     -1
#define SR_ERR_INVALID_ARGUMENT -2
#define SR_ERR_RUNTIME         -3
#define SR_ERR_OUT_OF_MEMORY   -4

/* Operations */
#define SR_OP_SCALE             1u  /* out = resample(a) by scale_num/scale_den */
#define SR_OP_BLEND_INTERPOLATE 2u  /* out = interpolate(a, b)                  */

/* Pixel formats */
#define SR_PIXEL_RGB8  1u
#define SR_PIXEL_RGBA8 2u
#define SR_PIXEL_GRAY8 3u

typedef struct sr_infer_capabilities {
    uint32_t abi_version;   /* must equal SR_INFER_ABI_VERSION */
    uint32_t scale;         /* non-zero when SR_OP_SCALE is supported */
    uint32_t interpolate;   /* non-zero when SR_OP_BLEND_INTERPOLATE is supported */
    uint32_t restore;       /* non-zero when the plugin does generative restoration */
    uint64_t max_pixels;    /* 0 = no stated limit */
    const char *backend;    /* "vulkan", "dml", "openvino", "cpu", ... */
    const char *precision;  /* "fp32", "fp16", "fp8", ... */
    const char *vendor;     /* NULL for vendor-neutral implementations */
} sr_infer_capabilities;

typedef struct sr_infer_frame {
    uint8_t *data;
    uint32_t width;
    uint32_t height;
    uint32_t stride;        /* bytes per row; 0 = tightly packed */
    uint32_t pixel_format;
} sr_infer_frame;

typedef struct sr_infer_request {
    uint32_t op;
    uint32_t input_count;   /* 1 for scale, 2 for interpolation */
    sr_infer_frame a;
    sr_infer_frame b;
    sr_infer_frame out;
    uint32_t scale_num;
    uint32_t scale_den;
} sr_infer_request;

typedef struct sr_infer_response {
    uint32_t frames_written;
    uint64_t vram_used_bytes;
    const char *message;    /* static or plugin-owned; may be NULL */
} sr_infer_response;

uint32_t sr_infer_abi_version(void);

/* Fills `out`. Returns SR_OK, or a negative error code. */
int32_t sr_infer_capabilities(sr_infer_capabilities *out);

/* Runs one operation. Buffers belong to the caller and must not be retained. */
int32_t sr_infer_run(const sr_infer_request *request, sr_infer_response *response);

#ifdef __cplusplus
}
#endif

#endif /* SR_INFER_H */
