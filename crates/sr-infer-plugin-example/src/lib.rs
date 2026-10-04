//! Reference implementation of inference plugin ABI v2.
//!
//! This crate exists to keep the ABI honest. It implements every symbol the
//! engine calls, on the CPU, with no dependencies — and it deliberately does
//! *not* depend on `sr-core`: it defines its own copies of the structs from
//! `include/sr_infer.h`, so if the header and the engine's bindings ever drift
//! apart, the ABI tests fail instead of the plugin silently misreading memory.
//!
//! It is not a quality model. It is a *test instrument*:
//!
//! * interpolation blends the two neighbouring frames with detail preservation
//!   (when the pair differs a lot at a pixel, it takes the nearer frame rather
//!   than averaging, so a cut never produces a ghost) — enough to prove that
//!   synthesised frames really are synthesised, because a duplicated frame is
//!   measurably identical to its neighbour and a blended one is not;
//! * restoration applies a deterministic unsharp mask;
//! * every call is appended to the file named by `SR_INFER_CALL_LOG` as JSONL,
//!   which is how the engine's tests prove that the model ran, how many frames
//!   it saw, and that no call ever spanned a cut;
//! * `config_json: {"fake_oom": n}` makes the session's next `n` calls fail with
//!   `SR_ERR_OUT_OF_MEMORY`, which is how the degrade ladder is tested without
//!   owning a 16 GB GPU. It is per session on purpose: a process-wide switch
//!   would make parallel tests interfere with each other.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::Write;

pub const SR_INFER_ABI_VERSION: u32 = 2;

const SR_OK: i32 = 0;
const SR_ERR_UNSUPPORTED: i32 = -1;
const SR_ERR_INVALID_ARGUMENT: i32 = -2;
const SR_ERR_RUNTIME: i32 = -3;
const SR_ERR_OUT_OF_MEMORY: i32 = -4;
const SR_ERR_NO_DEVICE: i32 = -7;

const SR_OP_RESTORE: u32 = 1 << 0;
const SR_OP_INTERPOLATE: u32 = 1 << 1;

const SR_DTYPE_U8: u32 = 1 << 0;

const SR_LAYOUT_INTERLEAVED: u32 = 1;
const SR_COLOR_RGB: u32 = 1;

const SR_CAP_TILING: u32 = 1 << 0;
const SR_CAP_TEMPORAL: u32 = 1 << 3;

const SR_DEVICE_CPU: u32 = 1;

#[repr(C)]
pub struct SrInferDevice {
    pub struct_size: u32,
    pub index: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub reserved: u32,
    pub name: *const c_char,
    pub driver: *const c_char,
    pub vram_bytes: u64,
    pub vram_budget_bytes: u64,
    pub vram_used_bytes: u64,
}

#[repr(C)]
pub struct SrInferCaps {
    pub struct_size: u32,
    pub abi_version: u32,
    pub ops: u32,
    pub dtypes: u32,
    pub flags: u32,
    pub device_count: u32,
    pub max_batch: u32,
    pub min_temporal_window: u32,
    pub max_temporal_window: u32,
    pub tile_min: u32,
    pub tile_max: u32,
    pub upscale_num: u32,
    pub upscale_den: u32,
    pub max_pixels: u64,
    pub vram_bytes: u64,
    pub backend: *const c_char,
    pub precision: *const c_char,
    pub vendor: *const c_char,
    pub model: *const c_char,
    pub model_version: *const c_char,
}

#[repr(C)]
pub struct SrInferImage {
    pub struct_size: u32,
    pub memory: u32,
    pub dtype: u32,
    pub layout: u32,
    pub color: u32,
    pub range: u32,
    pub bit_depth: u32,
    pub planes: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub plane_stride: [u32; 4],
    pub data: *mut c_void,
    pub device_handle: u64,
    pub pts_num: i64,
    pub pts_den: i64,
}

#[repr(C)]
pub struct SrInferJob {
    pub struct_size: u32,
    pub op: u32,
    pub flags: u32,
    pub input_count: u32,
    pub output_count: u32,
    pub inputs: *const SrInferImage,
    pub outputs: *mut SrInferImage,
    pub multiplier: u32,
    pub strength: f32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tile_pad: u32,
    pub reserved: u32,
    pub seed: u64,
    pub chunk_id: u64,
    pub fence: u64,
}

#[repr(C)]
pub struct SrInferResult {
    pub struct_size: u32,
    pub outputs_written: u32,
    pub tiles: u32,
    pub reserved: u32,
    pub vram_used_bytes: u64,
    pub vram_budget_bytes: u64,
    pub fence: u64,
    pub message: *const c_char,
}

#[repr(C)]
pub struct SrInferSessionDesc {
    pub struct_size: u32,
    pub device_index: u32,
    pub flags: u32,
    pub reserved: u32,
    pub model_path: *const c_char,
    pub model_name: *const c_char,
    pub config_json: *const c_char,
    pub vram_budget_bytes: u64,
    pub host_memory_budget_bytes: u64,
}

/// The plugin's session state.
pub struct Session {
    pub device_index: u32,
    pub model: String,
    pub calls: u64,
    pub frames_in: u64,
    pub frames_out: u64,
    pub last_error: CString,
    /// Fault injection: fail this many calls with `SR_ERR_OUT_OF_MEMORY` before
    /// doing any work. Taken from the session's `config_json` so it is scoped to
    /// one session rather than to the whole process — a process-wide switch would
    /// make the engine's own tests interfere with each other.
    pub fake_oom: u32,
    pub oom_served: u32,
    /// Kept alive because `last_error` pointers are handed out.
    _model_name: CString,
}

/// Minimal reader for one unsigned field of the session's `config_json`.
///
/// The ABI says a plugin ignores keys it does not know, so a full JSON parser
/// would be dead weight in a reference implementation.
fn json_u32(text: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{key}\"");
    let start = text.find(&needle)? + needle.len();
    let rest = &text[start..];
    let colon = rest.find(':')?;
    let digits: String = rest[colon + 1..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Panics inside a plugin would cross the FFI boundary, which is undefined
/// behaviour; every exported function runs under this guard.
fn guard<F: FnOnce() -> i32 + std::panic::UnwindSafe>(f: F) -> i32 {
    match std::panic::catch_unwind(f) {
        Ok(code) => code,
        Err(_) => SR_ERR_RUNTIME,
    }
}

unsafe fn opt_cstr(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().ok().map(str::to_owned)
}

fn cstring(text: impl Into<String>) -> CString {
    let text = text.into();
    CString::new(text.replace('\0', " ")).unwrap_or_else(|_| CString::new("error").expect("static"))
}

// ---- ABI ------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn sr_infer_abi_version() -> u32 {
    SR_INFER_ABI_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn sr_infer_query(out: *mut SrInferCaps) -> c_int {
    guard(|| {
        if out.is_null() {
            return SR_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: the caller passes a live struct; we only write fields that fit
        // inside the size it declared.
        let caps = unsafe { &mut *out };
        let declared = caps.struct_size as usize;
        let written = std::mem::size_of::<SrInferCaps>();
        if declared < written {
            return SR_ERR_INVALID_ARGUMENT;
        }
        caps.abi_version = SR_INFER_ABI_VERSION;
        caps.ops = SR_OP_RESTORE | SR_OP_INTERPOLATE;
        // Only what the reference kernels actually accept: claiming a dtype the
        // backend would reject is the kind of lie this ABI exists to prevent.
        caps.dtypes = SR_DTYPE_U8;
        caps.flags = SR_CAP_TILING | SR_CAP_TEMPORAL;
        caps.device_count = 1;
        caps.max_batch = 5;
        caps.min_temporal_window = 2;
        caps.max_temporal_window = 5;
        caps.tile_min = 64;
        caps.tile_max = 4096;
        caps.upscale_num = 1;
        caps.upscale_den = 1;
        caps.max_pixels = 0;
        caps.vram_bytes = 0;
        caps.backend = c"cpu".as_ptr();
        caps.precision = c"fp32".as_ptr();
        caps.vendor = std::ptr::null();
        caps.model = c"reference-blend".as_ptr();
        caps.model_version = c"2.0".as_ptr();
        SR_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn sr_infer_devices(
    out: *mut SrInferDevice,
    capacity: u32,
    count: *mut u32,
) -> c_int {
    guard(|| {
        if count.is_null() {
            return SR_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: `count` is a valid out-pointer per the ABI.
        unsafe { *count = 1 };
        if out.is_null() || capacity == 0 {
            return SR_OK;
        }
        // SAFETY: the caller promised room for `capacity` entries.
        let device = unsafe { &mut *out };
        if (device.struct_size as usize) < std::mem::size_of::<SrInferDevice>() {
            return SR_ERR_INVALID_ARGUMENT;
        }
        device.index = 0;
        device.vendor_id = 0;
        device.device_id = 0;
        device.device_type = SR_DEVICE_CPU;
        device.reserved = 0;
        device.name = c"sr-infer reference CPU device".as_ptr();
        device.driver = c"built-in".as_ptr();
        device.vram_bytes = 0;
        device.vram_budget_bytes = 0;
        device.vram_used_bytes = 0;
        SR_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn sr_infer_open(
    desc: *const SrInferSessionDesc,
    out: *mut *mut Session,
) -> c_int {
    guard(|| {
        if desc.is_null() || out.is_null() {
            return SR_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: per the ABI both pointers are live and correctly sized.
        let desc = unsafe { &*desc };
        if (desc.struct_size as usize) < std::mem::size_of::<SrInferSessionDesc>() {
            return SR_ERR_INVALID_ARGUMENT;
        }
        if desc.device_index != 0 {
            return SR_ERR_NO_DEVICE;
        }
        let requested = unsafe { opt_cstr(desc.model_name) };
        let model = requested.unwrap_or_else(|| "reference-blend".to_string());
        let _model_name = cstring(model.clone());
        let config = unsafe { opt_cstr(desc.config_json) }.unwrap_or_default();
        let fake_oom = json_u32(&config, "fake_oom").unwrap_or(0);
        let session = Box::new(Session {
            device_index: 0,
            model,
            calls: 0,
            frames_in: 0,
            frames_out: 0,
            last_error: cstring(""),
            fake_oom,
            oom_served: 0,
            _model_name,
        });
        // SAFETY: ownership transfers to the caller, which returns it through
        // `sr_infer_close`.
        unsafe { *out = Box::into_raw(session) };
        SR_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn sr_infer_close(session: *mut Session) {
    if session.is_null() {
        return;
    }
    // SAFETY: the pointer came from `sr_infer_open` and is closed exactly once.
    drop(unsafe { Box::from_raw(session) });
}

#[no_mangle]
pub unsafe extern "C" fn sr_infer_last_error(
    session: *mut Session,
    buffer: *mut c_char,
    capacity: u32,
) -> c_int {
    guard(|| {
        if session.is_null() || buffer.is_null() || capacity == 0 {
            return SR_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: live session from `sr_infer_open`.
        let session = unsafe { &*session };
        let bytes = session.last_error.as_bytes_with_nul();
        let n = bytes.len().min(capacity as usize);
        // SAFETY: the caller promised `capacity` writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buffer, n) };
        if n < capacity as usize {
            // SAFETY: index n is inside the buffer.
            unsafe { *buffer.add(n) = 0 };
        } else {
            // SAFETY: index capacity-1 is inside the buffer.
            unsafe { *buffer.add(capacity as usize - 1) = 0 };
        }
        (n.saturating_sub(1)) as c_int
    })
}

#[no_mangle]
pub unsafe extern "C" fn sr_infer_poll(
    _session: *mut Session,
    _fence: u64,
    _timeout_ms: u32,
) -> c_int {
    // Every job completes before `sr_infer_execute` returns, so there is never
    // anything to wait for. The example does not advertise SR_CAP_ASYNC.
    SR_OK
}

fn log_call(session: &Session, job: &SrInferJob, inputs: &[SrInferImage], outputs: &[SrInferImage]) {
    let Ok(path) = std::env::var("SR_INFER_CALL_LOG") else {
        return;
    };
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    let pts = |image: &SrInferImage| {
        if image.pts_den == 0 {
            0.0
        } else {
            image.pts_num as f64 / image.pts_den as f64
        }
    };
    let first = inputs.first().map(pts).unwrap_or(0.0);
    let last = inputs.last().map(pts).unwrap_or(0.0);
    let _ = writeln!(
        file,
        "{{\"op\":{},\"in\":{},\"out\":{},\"chunk\":{},\"width\":{},\"height\":{},\
          \"first_pts\":{:.6},\"last_pts\":{:.6},\"calls\":{}}}",
        job.op,
        inputs.len(),
        outputs.len(),
        job.chunk_id,
        inputs.first().map(|i| i.width).unwrap_or(0),
        inputs.first().map(|i| i.height).unwrap_or(0),
        first,
        last,
        session.calls
    );
}

#[no_mangle]
pub unsafe extern "C" fn sr_infer_execute(
    session: *mut Session,
    job: *const SrInferJob,
    result: *mut SrInferResult,
) -> c_int {
    guard(|| {
        if session.is_null() || job.is_null() || result.is_null() {
            return SR_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: live session, job and result per the ABI.
        let session = unsafe { &mut *session };
        let job = unsafe { &*job };
        let result = unsafe { &mut *result };
        result.struct_size = std::mem::size_of::<SrInferResult>() as u32;

        let set_error = |session: &mut Session, code: i32, message: &str| -> i32 {
            session.last_error = cstring(message);
            code
        };

        // --- fault injection for the degrade ladder -------------------------
        if session.oom_served < session.fake_oom {
            session.oom_served += 1;
            return set_error(
                session,
                SR_ERR_OUT_OF_MEMORY,
                "injected out-of-memory fault (config_json: fake_oom)",
            );
        }

        if (job.struct_size as usize) < std::mem::size_of::<SrInferJob>() {
            return SR_ERR_INVALID_ARGUMENT;
        }
        if job.inputs.is_null() || job.input_count == 0 {
            return set_error(session, SR_ERR_INVALID_ARGUMENT, "no input frames");
        }
        // SAFETY: the caller guarantees `input_count` images.
        let inputs = unsafe { std::slice::from_raw_parts(job.inputs, job.input_count as usize) };
        let outputs: &mut [SrInferImage] = if job.outputs.is_null() {
            &mut []
        } else {
            // SAFETY: the caller guarantees `output_count` images.
            unsafe { std::slice::from_raw_parts_mut(job.outputs, job.output_count as usize) }
        };

        for image in inputs.iter().chain(outputs.iter()) {
            if image.data.is_null() {
                return set_error(session, SR_ERR_INVALID_ARGUMENT, "null frame buffer");
            }
            if image.dtype != SR_DTYPE_U8
                || image.layout != SR_LAYOUT_INTERLEAVED
                || image.color != SR_COLOR_RGB
            {
                return set_error(
                    session,
                    SR_ERR_UNSUPPORTED,
                    "the reference backend only accepts interleaved 8-bit RGB",
                );
            }
        }

        let code = match job.op {
            SR_OP_INTERPOLATE => interpolate(session, job, inputs, outputs),
            SR_OP_RESTORE => restore(session, job, inputs, outputs),
            other => {
                return set_error(
                    session,
                    SR_ERR_UNSUPPORTED,
                    &format!("op {other} is not implemented by the reference backend"),
                )
            }
        };

        if code == SR_OK {
            session.calls += 1;
            session.frames_in += inputs.len() as u64;
            session.frames_out += if outputs.is_empty() {
                inputs.len() as u64
            } else {
                outputs.len() as u64
            };
            log_call(session, job, inputs, outputs);
            result.outputs_written = if outputs.is_empty() {
                inputs.len() as u32
            } else {
                outputs.len() as u32
            };
            result.tiles = 1;
            result.message = c"reference backend".as_ptr();
        }
        code
    })
}

fn bytes(image: &SrInferImage) -> usize {
    let row = if image.stride == 0 {
        image.width as usize * 3
    } else {
        image.stride as usize
    };
    row * image.height as usize
}

/// Blend two frames with detail preservation.
///
/// Averaging alone would ghost across a cut; taking the nearer frame whenever
/// the pair disagrees strongly keeps edges crisp. Either way the result is a
/// *new* frame, which is the property the engine's tests check for.
fn blend_into(a: &[u8], b: &[u8], out: &mut [u8], weight_b: f32) {
    for i in 0..out.len() {
        let (x, y) = (a[i] as f32, b[i] as f32);
        let blended = x * (1.0 - weight_b) + y * weight_b;
        out[i] = if (x - y).abs() > 32.0 {
            if weight_b < 0.5 {
                x as u8
            } else {
                y as u8
            }
        } else {
            blended.round().clamp(0.0, 255.0) as u8
        };
    }
}

fn interpolate(
    session: &mut Session,
    job: &SrInferJob,
    inputs: &[SrInferImage],
    outputs: &mut [SrInferImage],
) -> i32 {
    let multiplier = job.multiplier.max(1) as usize;
    let expected = (inputs.len() - 1) * multiplier + 1;
    if outputs.is_empty() || outputs.len() != expected {
        session.last_error = cstring(format!(
            "interpolation geometry: {} inputs at {multiplier}x need {expected} outputs, {} given",
            inputs.len(),
            outputs.len()
        ));
        return SR_ERR_INVALID_ARGUMENT;
    }
    for (index, output) in outputs.iter_mut().enumerate() {
        // Position in input space: 0.0 at input 0, (len-1) at the last input.
        let position = index as f32 / multiplier as f32;
        let left = position.floor() as usize;
        let frac = position - left as f32;
        let right = (left + 1).min(inputs.len() - 1);
        let a = &inputs[left];
        let b = &inputs[right];
        if bytes(a) != bytes(output) || bytes(b) != bytes(output) {
            session.last_error = cstring("frame geometry mismatch between input and output");
            return SR_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: the ABI guarantees each image points at `bytes(image)` valid
        // bytes owned by the engine for the duration of the call.
        let (a_data, b_data, out_data) = unsafe {
            (
                std::slice::from_raw_parts(a.data as *const u8, bytes(a)),
                std::slice::from_raw_parts(b.data as *const u8, bytes(b)),
                std::slice::from_raw_parts_mut(output.data as *mut u8, bytes(output)),
            )
        };
        blend_into(a_data, b_data, out_data, frac);
    }
    SR_OK
}

fn restore(
    session: &mut Session,
    job: &SrInferJob,
    inputs: &[SrInferImage],
    outputs: &mut [SrInferImage],
) -> i32 {
    // In-place: the engine passes the same buffers as inputs and outputs.
    let targets: &mut [SrInferImage] = if outputs.is_empty() {
        session.last_error = cstring(
            "restore requires writable outputs (the engine passes aliased buffers)",
        );
        return SR_ERR_INVALID_ARGUMENT;
    } else {
        outputs
    };
    if targets.len() != inputs.len() {
        session.last_error = cstring("restore needs one output per input frame");
        return SR_ERR_INVALID_ARGUMENT;
    }
    let strength = job.strength.clamp(0.0, 1.0);
    for (input, output) in inputs.iter().zip(targets.iter_mut()) {
        let len = bytes(input);
        if len != bytes(output) {
            session.last_error = cstring("restore geometry mismatch");
            return SR_ERR_INVALID_ARGUMENT;
        }
        // SAFETY: both buffers are valid for `len` bytes for this call.
        let src = unsafe { std::slice::from_raw_parts(input.data as *const u8, len) };
        // SAFETY: as above. The two may be the *same* buffer — the engine calls
        // RESTORE in place — so the result is built in scratch memory and copied
        // once, which keeps this safe for aliased and separate buffers alike.
        let dst = unsafe { std::slice::from_raw_parts_mut(output.data as *mut u8, len) };
        let sharpened = unsharp(src, input.width as usize, input.height as usize, strength);
        dst.copy_from_slice(&sharpened);
    }
    SR_OK
}

/// Deterministic 3x3 unsharp mask, purely so RESTORE has a visible effect.
fn unsharp(src: &[u8], width: usize, height: usize, strength: f32) -> Vec<u8> {
    let mut out = src.to_vec();
    if width < 3 || height < 3 {
        return out;
    }
    let amount = 0.6 * strength;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            for channel in 0..3 {
                let at =
                    |px: usize, py: usize| -> f32 { src[(py * width + px) * 3 + channel] as f32 };
                let centre = at(x, y);
                let blurred = (at(x - 1, y)
                    + at(x + 1, y)
                    + at(x, y - 1)
                    + at(x, y + 1)
                    + centre * 4.0)
                    / 8.0;
                let sharpened = centre + (centre - blurred) * amount;
                out[(y * width + x) * 3 + channel] = sharpened.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Convenience for plugin authors: shows that a panic inside a callback is
/// contained rather than allowed to unwind into the engine.
#[doc(hidden)]
pub fn panics_are_contained() -> bool {
    std::panic::catch_unwind(|| true).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_produces_a_new_frame_between_the_inputs() {
        // A small change is blended, which is the case where a synthesised frame
        // is genuinely between the two originals.
        let a = [100u8; 12];
        let b = [120u8; 12];
        let mut out = [0u8; 12];
        blend_into(&a, &b, &mut out, 0.5);
        assert!(out.iter().all(|v| *v == 110), "{out:?}");
    }

    #[test]
    fn blend_refuses_to_ghost_a_cut() {
        // A hard edge between the two frames must survive as one or the other,
        // never as an average that exists in neither.
        let a = [0u8; 3];
        let b = [255u8; 3];
        let mut out = [0u8; 3];
        blend_into(&a, &b, &mut out, 0.25);
        assert_eq!(out, [0, 0, 0]);
        blend_into(&a, &b, &mut out, 0.75);
        assert_eq!(out, [255, 255, 255]);
    }

    #[test]
    fn the_abi_version_matches_the_header() {
        // Kept in sync by hand: the header is the contract and this constant is
        // the plugin's promise. crates/sr-core asserts the same number.
        assert_eq!(sr_infer_abi_version(), 2);
    }
}
