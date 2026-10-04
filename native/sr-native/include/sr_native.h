/*
 * sr_native.h — the one AI runtime's C ABI.
 *
 * This header is deliberately shaped like the product rather than like a
 * framework. This program runs exactly two networks:
 *
 *   Real-ESRGAN x4plus   restoration, single frame in, single frame out
 *   RIFE 4.25            interpolation, a pair plus a timestep in, one frame out
 *
 * so the ABI has a restorer, an interpolator and a device — and nothing else.
 * There is no tensor type, no graph, no operator registry, no capability
 * negotiation and no plugin discovery, because none of those describe anything
 * this program does. A previous revision of this project had all of them, and
 * the generality is what let "a model ran" and "something that looks similar
 * ran" become indistinguishable from the outside.
 *
 * Everything below is a *pixel* API. Frames are handed over through host memory
 * as interleaved 8-bit RGB; they are never written to a file and no child
 * process is ever spawned. Inside, ncnn (Vulkan) owns the device and the
 * weights.
 *
 * Ownership, in one paragraph: `sr_*_create` returns a handle you own and must
 * destroy; the destroy functions are safe on NULL; every `process` call is
 * synchronous and the output buffer is written only on success; and no function
 * retains a pointer to your memory after it returns.
 *
 * Threading: a context, a restorer and a rife handle are each owned by one
 * thread at a time. They are not internally synchronised, because the executor
 * drives exactly one model at a time by design — two networks plus their
 * activation buffers do not fit on the 16 GB card this is built for.
 *
 * License: this header is part of sr-player. The implementations link ncnn
 * (BSD-3-Clause), RIFE (MIT) and Real-ESRGAN (MIT), pinned by commit and
 * recorded in the release manifest.
 */

#ifndef SR_NATIVE_H
#define SR_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ errors */

enum {
    SR_OK = 0,
    /* A pointer, size, or argument combination cannot be honoured. */
    SR_ERR_INVALID_ARGUMENT = 1,
    /* No Vulkan device, or the requested index does not exist. */
    SR_ERR_NO_DEVICE = 2,
    /* The model directory is missing files, or a file failed its hash check. */
    SR_ERR_MODEL_MISSING = 3,
    /* The device refused an allocation. The caller should walk its tile ladder. */
    SR_ERR_OUT_OF_MEMORY = 4,
    /* The Vulkan device was lost. Not retryable on this context. */
    SR_ERR_DEVICE_LOST = 5,
    SR_ERR_INTERNAL = 6
};

/* ------------------------------------------------------------------- images */

/*
 * A borrowed view of one interleaved RGB8 frame.
 *
 * `stride` is in BYTES and may exceed `width * 3`, so a caller can hand over a
 * row-padded buffer without copying it. `channels` must be 3.
 *
 * The library never writes through a const `sr_image`, and never reads through
 * the `data` of an output image — an output's buffer is only ever written, and
 * it must already be large enough for the result (see `sr_restorer_output_size`
 * and `sr_rife_process`).
 */
typedef struct sr_image {
    uint8_t* data;
    int32_t  width;
    int32_t  height;
    /* Bytes between the start of one row and the start of the next. */
    int32_t  stride;
    /* Must be 3 (interleaved RGB, 8 bits per component). */
    int32_t  channels;
} sr_image;

/* ------------------------------------------------------------------ devices */

enum {
    SR_DEVICE_OTHER = 0,
    SR_DEVICE_INTEGRATED = 1,
    SR_DEVICE_DISCRETE = 2,
    SR_DEVICE_VIRTUAL = 3,
    SR_DEVICE_CPU = 4
};

typedef struct sr_device_info_t {
    /* Set this to sizeof(sr_device_info_t) before the call. */
    uint32_t struct_size;
    int32_t  index;

    /* NUL-terminated, UTF-8, truncated rather than refused if it is long. */
    char     name[256];
    char     driver_version[64];
    char     api_version[32];

    uint32_t vendor_id;
    uint32_t device_id;
    int32_t  device_type;
    /* Size of the device-local heap, 0 when the driver will not say. */
    uint64_t total_mib;
    /*
     * What the driver will let THIS process allocate (VK_EXT_memory_budget),
     * minus what it is already holding. 0 when unsupported, which is a real
     * answer and not an error: the caller then falls back to its own probe.
     */
    uint64_t budget_mib;
    /* Non-zero when device-local memory is system RAM shared with the OS. */
    int32_t  unified_memory;
} sr_device_info_t;

/* How many Vulkan devices the runtime can see. 0 means it cannot run. */
int sr_device_count(void);

/*
 * Fills `out` for device `index`. Returns SR_ERR_NO_DEVICE if there is no such
 * device, SR_ERR_INVALID_ARGUMENT if `struct_size` is wrong.
 *
 * The struct carries the `_t` suffix and the function does not, which is not
 * decoration: C++ has one identifier namespace for both, so a struct named
 * `sr_device_info` alongside a function of that name cannot be compiled as C++ at
 * all — `sr_device_info info;` parses as a call.
 */
int sr_device_info(int32_t index, sr_device_info_t* out);

/* ------------------------------------------------------------------ context */

typedef struct sr_context sr_context;

/*
 * Opens device `index` and creates the Vulkan instance and queue the models
 * will use. Returns NULL on failure; there is no context to report the error on
 * yet, so the reason goes to stderr.
 *
 * One context owns the device. A restorer and a rife handle may share it, but
 * they are used one at a time.
 */
sr_context* sr_context_create(int32_t device_index);

/* Safe on NULL. Destroys any handles still attached before releasing the device. */
void sr_context_destroy(sr_context* ctx);

/*
 * The most recent failure on this context, or "" if none. Valid until the next
 * call on the same context.
 *
 * This is the one addition to the API the design called for, and it earns its
 * place: an integer code cannot say *which* file was missing from a model
 * directory, and the alternative is a diagnostic on stderr that a GUI cannot
 * show. It is not a general logging facility.
 */
const char* sr_last_error(sr_context* ctx);

/* ------------------------------------------------- restoration (Real-ESRGAN) */

typedef struct sr_restorer sr_restorer;

/*
 * Loads `model_dir`, which must contain the pinned `model.param`, `model.bin`
 * and a `manifest.json` whose hash matches. Returns NULL and sets the context's
 * last error otherwise; a mismatch is refused rather than warned about, because
 * silently running different weights is the failure this project exists to stop.
 */
sr_restorer* sr_restorer_create(sr_context* ctx, const char* model_dir);

void sr_restorer_destroy(sr_restorer* restorer);

/*
 * The size `out` must be for a given input and scale. Callers use this to size
 * their buffer instead of assuming the model's advertised scale is the whole
 * story; it returns 0 on a NULL or nonsensical argument.
 */
int32_t sr_restorer_output_size(const sr_restorer* restorer,
                                int32_t in_width,
                                int32_t in_height,
                                int32_t scale,
                                int32_t* out_width,
                                int32_t* out_height);

/*
 * Restores one frame.
 *
 * `scale` must be a factor the loaded model supports (4 for x4plus).
 * `tile` is the working-set knob: 0 lets the runtime choose, otherwise it is the
 * tile edge in pixels, and it must be >= 32. A smaller tile has a smaller peak
 * activation footprint and a higher chance of visible seams, which is the
 * trade the caller's ladder is making.
 *
 * Returns SR_ERR_OUT_OF_MEMORY when the device refuses an allocation, and the
 * caller retries the same frame with a smaller tile. Any other failure is not
 * capacity-related and retrying with a different tile will not help.
 */
int sr_restorer_process(sr_restorer* restorer,
                        const sr_image* in,
                        sr_image* out,
                        int32_t scale,
                        int32_t tile);

/* ------------------------------------------------- interpolation (RIFE 4.25) */

typedef struct sr_rife sr_rife;

/*
 * Loads the pinned RIFE 4.25 weights from `model_dir`, always with
 * `ensemble = false`: the ensemble doubles the work for a difference nobody has
 * demonstrated on real film, and a single frozen configuration is what makes
 * "zero per-tune tuning" a testable claim.
 */
sr_rife* sr_rife_create(sr_context* ctx, const char* model_dir);

void sr_rife_destroy(sr_rife* rife);

/*
 * Synthesises the frame `timestep` of the way from `prev` to `next`, writing it
 * to `out`.
 *
 * `timestep` is in (0, 1) exclusive — 0.5 is the midpoint, which is what a 2x
 * "film mode" run always asks for. The endpoints are not synthesised: a caller
 * that wants them already has them.
 *
 * `prev` and `next` must have identical dimensions, and `out` must be sized to
 * match. Nothing here knows about shots or cuts: the executor guarantees that
 * `prev` and `next` come from the same shot by never constructing the pair
 * otherwise, which is why this function takes two frames rather than a window.
 *
 * Note the reciprocal of that guarantee — this call cannot check it. Two frames
 * from different shots will produce a confident, wrong, blended frame, so the
 * cut-safety contract lives entirely on the caller's side of this boundary, and
 * `pipeline::segments` is where it is kept.
 */
int sr_rife_process(sr_rife* rife,
                    const sr_image* prev,
                    const sr_image* next,
                    float timestep,
                    sr_image* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SR_NATIVE_H */
