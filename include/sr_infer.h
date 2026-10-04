/*
 * sr_infer.h — vendor-neutral inference plugin ABI, version 2.
 *
 * A plugin is a shared library that exports the symbols declared at the bottom
 * of this file. The engine never links against a vendor SDK: it enumerates
 * devices, opens a session, and hands the session frames. The same ABI can be
 * backed by Vulkan compute, DirectML, OpenVINO, MIGraphX, CUDA/TensorRT, ncnn
 * or a plain CPU kernel, and the engine cannot tell the difference.
 *
 * Relationship to version 1
 * -------------------------
 * Version 1 was a prototype: one call, two 8-bit frames in, one frame out, no
 * device, no session, no temporal window, no dtype, no tiling, no memory
 * budget. It could not express what a real restoration model needs, so it has
 * been replaced rather than extended. This header is the contract; version 1
 * plugins are rejected with SR_ERR_UNSUPPORTED at load time.
 *
 * Design rules
 * ------------
 * 1. Every struct starts with `struct_size`. A plugin fills in only the fields
 *    it understands and must not write past `struct_size`; the engine only
 *    reads fields it knows about and passes its own size. That is what makes it
 *    possible to add fields in 2.1 without breaking 2.0 binaries.
 * 2. Every struct starts at the same version marker so a mismatched struct can
 *    be detected rather than misread.
 * 3. Memory is caller-owned unless the plugin advertises SR_CAP_DEVICE_MEMORY.
 *    The engine allocates host frames, hands them over, and keeps them alive
 *    for the duration of the call. Nothing is retained by the plugin.
 * 4. Failures are codes plus a message. SR_ERR_OUT_OF_MEMORY is a *signal*, not
 *    an error: the engine answers it by shrinking tiles, batch or precision and
 *    running the same work again. A plugin that returns it must leave no state
 *    behind that would make a retry unsafe.
 * 5. Time is rational. Frame timestamps cross the boundary as num/den so a
 *    23.976 -> 47.952 conversion is exact instead of accumulated float drift.
 *
 * Build a plugin:
 *   Windows:  cl /LD my_backend.c /Fe:sr_infer.dll
 *   Linux:    cc -shared -fPIC -o libsr_infer.so my_backend.c
 *   Rust:     see crates/sr-infer-plugin-example (a working reference)
 *
 * Install it next to the executable, in ./plugins/, or point SR_INFER_PLUGIN at
 * it. With no plugin present the engine uses its deterministic FFmpeg path, so
 * a plugin is an accelerator and never a requirement.
 */

#ifndef SR_INFER_H
#define SR_INFER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SR_INFER_ABI_VERSION 2u

/* ---- error codes -------------------------------------------------------- */

#define SR_OK                    0
#define SR_ERR_UNSUPPORTED      -1  /* op/dtype/layout this plugin cannot do    */
#define SR_ERR_INVALID_ARGUMENT -2  /* malformed struct, bad geometry, no buffer */
#define SR_ERR_RUNTIME          -3  /* device lost, driver error, model failure  */
#define SR_ERR_OUT_OF_MEMORY    -4  /* retryable: shrink and call again          */
#define SR_ERR_CANCELLED        -5  /* the engine asked to stop                  */
#define SR_ERR_DEVICE_LOST      -6  /* unrecoverable: the session must be reopened */
#define SR_ERR_NO_DEVICE        -7  /* nothing to run on                         */

/* ---- operations (bitmask in sr_infer_caps.ops) --------------------------- */

/* N frames in, N frames out: denoise/deblur/upscale. Geometry may change by
 * caps.upscale_num / caps.upscale_den. */
#define SR_OP_RESTORE      (1u << 0)
/* N frames in, M frames out: temporal interpolation, M = (N-1)*multiplier + 1. */
#define SR_OP_INTERPOLATE  (1u << 1)
/* 1 frame in, 1 frame out: pure resample, no detail invented. */
#define SR_OP_SCALE        (1u << 2)

/* ---- dtypes (bitmask in sr_infer_caps.dtypes) ---------------------------- */

#define SR_DTYPE_U8    (1u << 0)
#define SR_DTYPE_U16   (1u << 1)
#define SR_DTYPE_F16   (1u << 2)
#define SR_DTYPE_BF16  (1u << 3)
#define SR_DTYPE_F32   (1u << 4)

/* ---- layout ------------------------------------------------------------- */

#define SR_LAYOUT_INTERLEAVED  1u   /* RGBRGB… YUVYUV…                        */
#define SR_LAYOUT_PLANAR       2u   /* RRR…GGG…BBB…                           */
#define SR_LAYOUT_SEMI_PLANAR  3u   /* NV12-style: Y plane + interleaved chroma */

/* ---- colour ------------------------------------------------------------- */

#define SR_COLOR_RGB   1u
#define SR_COLOR_BGR   2u
#define SR_COLOR_GRAY  3u
#define SR_COLOR_YUV   4u

#define SR_RANGE_FULL     0u  /* 0..255 / 0..1023 / 0..1.0                    */
#define SR_RANGE_LIMITED  1u  /* 16..235 / 64..940                            */

/* ---- buffers ------------------------------------------------------------ */

#define SR_MEM_HOST     (1u << 0)  /* `data` points at accessible memory      */
#define SR_MEM_DEVICE   (1u << 1)  /* `device_handle` is a device pointer     */
#define SR_MEM_ALIAS    (1u << 2)  /* outputs may alias inputs               */

/* ---- capability flags --------------------------------------------------- */

#define SR_CAP_TILING          (1u << 0)  /* honours tile_width/tile_height   */
#define SR_CAP_DEVICE_MEMORY   (1u << 1)  /* accepts/returns device handles   */
#define SR_CAP_ASYNC           (1u << 2)  /* execute may return before finish */
#define SR_CAP_TEMPORAL        (1u << 3)  /* wants more than 2 input frames   */
#define SR_CAP_MULTI_DEVICE    (1u << 4)  /* more than one device is usable   */
#define SR_CAP_PARTIAL_OUTPUT  (1u << 5)  /* may write fewer than requested   */

/* ---- device ------------------------------------------------------------- */

#define SR_DEVICE_CPU         1u
#define SR_DEVICE_INTEGRATED  2u
#define SR_DEVICE_DISCRETE    3u
#define SR_DEVICE_VIRTUAL     4u

typedef struct sr_infer_device {
    uint32_t struct_size;
    uint32_t index;             /* pass back in sr_infer_session_desc        */
    uint32_t vendor_id;         /* PCI vendor id: 0x10DE, 0x1002, 0x8086 …   */
    uint32_t device_id;
    uint32_t device_type;       /* SR_DEVICE_*                               */
    uint32_t reserved;
    const char *name;           /* static, NUL-terminated, may be NULL       */
    const char *driver;         /* version string, may be NULL               */
    uint64_t vram_bytes;        /* 0 = unknown or unified                    */
    /* From VK_EXT_memory_budget / NVML / DXGI when the backend can see it. */
    uint64_t vram_budget_bytes; /* 0 = unknown                               */
    uint64_t vram_used_bytes;   /* 0 = unknown                               */
} sr_infer_device;

/* ---- capabilities ------------------------------------------------------- */

typedef struct sr_infer_caps {
    uint32_t struct_size;
    uint32_t abi_version;       /* must equal SR_INFER_ABI_VERSION           */
    uint32_t ops;               /* bitmask of SR_OP_*                        */
    uint32_t dtypes;            /* bitmask of SR_DTYPE_*                     */
    uint32_t flags;             /* bitmask of SR_CAP_*                       */
    uint32_t device_count;
    uint32_t max_batch;         /* frames per RESTORE call, >= 1             */
    uint32_t min_temporal_window; /* smallest useful INTERPOLATE/restore window */
    uint32_t max_temporal_window;
    uint32_t tile_min;          /* smallest tile edge the model accepts      */
    uint32_t tile_max;          /* largest tile edge before OOM on 16 GB     */
    uint32_t upscale_num;       /* geometry change of RESTORE, e.g. 2/1      */
    uint32_t upscale_den;
    uint64_t max_pixels;
    uint64_t vram_bytes;        /* what the model needs resident             */
    const char *backend;        /* "vulkan", "dml", "openvino", "cpu", …     */
    const char *precision;      /* "fp32", "fp16", "bf16", "fp8"             */
    const char *vendor;         /* NULL for vendor-neutral implementations   */
    const char *model;          /* model family, e.g. "rife", "seedvr2"      */
    const char *model_version;
} sr_infer_caps;

/* ---- images ------------------------------------------------------------- */

typedef struct sr_infer_image {
    uint32_t struct_size;
    uint32_t memory;            /* SR_MEM_*                                  */
    uint32_t dtype;             /* SR_DTYPE_*                                */
    uint32_t layout;            /* SR_LAYOUT_*                               */
    uint32_t color;             /* SR_COLOR_*                                */
    uint32_t range;             /* SR_RANGE_*                                */
    uint32_t bit_depth;         /* 8, 10, 12, 16, 32                         */
    uint32_t planes;            /* 1, 3 or 4                                 */
    uint32_t width;
    uint32_t height;
    uint32_t stride;            /* bytes per row, plane 0; 0 = packed        */
    uint32_t plane_stride[4];   /* bytes per row per plane; 0 = packed       */
    void    *data;              /* SR_MEM_HOST                               */
    uint64_t device_handle;     /* SR_MEM_DEVICE: backend-native pointer     */
    /* Rational presentation timestamp of this frame in the source timebase. */
    int64_t pts_num;
    int64_t pts_den;
} sr_infer_image;

/* ---- jobs --------------------------------------------------------------- */

#define SR_JOB_ASYNC    (1u << 0)  /* do not wait; return a fence           */
#define SR_JOB_INPLACE  (1u << 1)  /* outputs may reuse input buffers       */

typedef struct sr_infer_job {
    uint32_t struct_size;
    uint32_t op;                /* one SR_OP_*                               */
    uint32_t flags;             /* SR_JOB_*                                  */
    uint32_t input_count;       /* 1 for SCALE, >= min_temporal_window else  */
    uint32_t output_count;      /* RESTORE: == input_count; INTERPOLATE:     */
                                /* (input_count-1)*multiplier + 1            */
    const sr_infer_image *inputs;
    sr_infer_image *outputs;
    /* INTERPOLATE: multiplier >= 2. */
    uint32_t multiplier;
    /* RESTORE: model strength, 0..1. 1.0 = full model, lower = blended back. */
    float strength;
    /* Tiling. 0 = let the plugin decide; the engine sets these when the
     * degrade ladder needs a smaller working set. */
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t tile_pad;
    uint32_t reserved;
    /* Deterministic noise/grain seed; the engine keeps it stable across a
     * resume so a re-run reproduces the same pixels. */
    uint64_t seed;
    /* Correlates the call with the engine's chunk table and the plugin's log. */
    uint64_t chunk_id;
    /* Opaque fence returned by the plugin for SR_JOB_ASYNC. */
    uint64_t fence;
} sr_infer_job;

typedef struct sr_infer_result {
    uint32_t struct_size;
    uint32_t outputs_written;
    uint32_t tiles;             /* how many tiles the job was split into     */
    uint32_t reserved;
    uint64_t vram_used_bytes;   /* peak during this job, 0 = unknown         */
    uint64_t vram_budget_bytes;
    uint64_t fence;             /* valid when SR_JOB_ASYNC was requested     */
    /* NUL-terminated, owned by the plugin, valid until the next call on this
     * session. May be NULL. */
    const char *message;
} sr_infer_result;

/* ---- sessions ----------------------------------------------------------- */

typedef struct sr_infer_session sr_infer_session; /* opaque */

typedef struct sr_infer_session_desc {
    uint32_t struct_size;
    uint32_t device_index;      /* from sr_infer_device.index; 0 is CPU-first  */
    uint32_t flags;
    uint32_t reserved;
    const char *model_path;     /* weights file; NULL = the plugin's built-in  */
    const char *model_name;     /* e.g. "rife-v4.6"; NULL = plugin default     */
    /* Free-form JSON for model-specific knobs. The plugin ignores keys it does
     * not know, so the engine can pass its whole working-set description. */
    const char *config_json;
    /* Hard ceiling the plugin must respect; 0 = no stated limit. */
    uint64_t vram_budget_bytes;
    uint64_t host_memory_budget_bytes;
} sr_infer_session_desc;

/* ---- the ABI ------------------------------------------------------------ */

/* Returns SR_INFER_ABI_VERSION. Checked before anything else is called. */
uint32_t sr_infer_abi_version(void);

/* Static description of what this plugin can do, before any session exists.
 * `out->struct_size` must be set by the caller. */
int32_t sr_infer_query(sr_infer_caps *out);

/* Enumerates devices. Call with `out == NULL` and capacity 0 to learn `count`.
 * Returns SR_ERR_NO_DEVICE when the plugin has nothing to run on. */
int32_t sr_infer_devices(sr_infer_device *out, uint32_t capacity, uint32_t *count);

/* Opens a session: device selection, model load, memory reservation. */
int32_t sr_infer_open(const sr_infer_session_desc *desc, sr_infer_session **out);

/* Releases a session. Safe to call with NULL. Blocking. */
void sr_infer_close(sr_infer_session *session);

/* Runs one job. On SR_ERR_OUT_OF_MEMORY the engine shrinks the working set and
 * calls again with the same inputs. On SR_ERR_DEVICE_LOST it reopens. */
int32_t sr_infer_execute(sr_infer_session *session,
                         const sr_infer_job *job,
                         sr_infer_result *result);

/* Waits for an async job. `timeout_ms` of 0 polls. */
int32_t sr_infer_poll(sr_infer_session *session, uint64_t fence, uint32_t timeout_ms);

/* Copies the last error message into `buffer` (always NUL-terminated when
 * capacity > 0). Returns the number of bytes written, excluding the NUL. */
int32_t sr_infer_last_error(sr_infer_session *session, char *buffer, uint32_t capacity);

#ifdef __cplusplus
}
#endif

#endif /* SR_INFER_H */
