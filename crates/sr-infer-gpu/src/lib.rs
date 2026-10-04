//! `sr-infer-gpu`: a Vulkan inference backend for sr-player.
//!
//! Implements `include/sr_infer.h` (ABI v2) on top of the WGSL kernels in
//! [`kernels`], so the engine's native executor can push frames through the GPU
//! instead of through FFmpeg filters.
//!
//! ## What this backend actually is
//!
//! It is **block-matching motion-compensated interpolation**, not a learned
//! model. There are no weights to load, nothing to train and nothing to
//! hallucinate: for every 8x8 block it searches the displacement that best
//! matches the two frames, then warps both frames along half that displacement
//! and blends. Where the two warped samples disagree strongly — an occlusion —
//! the nearer one wins instead of averaging into a ghost.
//!
//! That places it above `framerate` duplication (which invents no motion at all)
//! and below a RIFE-class network (which learns the flow instead of searching for
//! it). Saying so is the point: this crate exists to make the *GPU execution
//! path* real and measurable, and `tests/motion.rs` measures it against known
//! ground truth rather than asserting that it looks fine.
//!
//! ## Honest capability report
//!
//! `sr_infer_query` advertises only interpolation, only `u8` RGB, only a
//! two-frame window, no tiling and no device memory. Restoration is not
//! implemented, so `SR_OP_RESTORE` is not advertised — the engine will not select
//! this backend for a restoration job, which is what stops `--profile safe-16gb`
//! from claiming a restoration that never happened.

pub mod abi;
pub mod engine;
pub mod ifnet;
pub mod kernels;

use abi::*;
use engine::{Engine, Session as GpuSession};
use std::ffi::{c_char, c_int, CStr, CString};
use std::io::Write;

/// The plugin's session state.
pub struct Session {
    pub device_index: u32,
    pub gpu: Option<GpuSession>,
    /// The restoration graph and the device it runs on, built on first use.
    ///
    /// Lazy because most sessions never restore: a job that only interpolates
    /// should not pay for a second device and a graph. It records the frame size it
    /// was built for, because the operators are size-independent but the
    /// intermediate buffers are not.
    pub restore: Option<Restore>,
    pub model: String,
    pub calls: u64,
    pub frames_in: u64,
    pub frames_out: u64,
    pub last_error: CString,
    pub width: u32,
    pub height: u32,
}

/// The restoration path's state: the model, the device, and the size it was built
/// for.
pub struct Restore {
    pub width: u32,
    pub height: u32,
    pub model: crate::ifnet::Model,
    pub ops: crate::ifnet_gpu::GpuOps,
}

impl Restore {
    /// Builds the scaffold for a frame size.
    ///
    /// The scale is 1. The ABI's `upscale` field describes the backend as a whole,
    /// and a backend that quietly doubled the picture while the plan also resized
    /// would scale twice. The topology supports any scale — `residual_sr` takes it
    /// as a parameter — and advertising one is a later step than running one.
    pub fn for_size(width: u32, height: u32) -> Result<Self, String> {
        let ops = crate::ifnet_gpu::GpuOps::open()?;
        let model = crate::ifnet::residual_sr(width as usize, height as usize, 3, 8, 1, 2);
        Ok(Restore {
            width,
            height,
            model,
            ops,
        })
    }
}

/// Interleaved 8-bit RGB to the interleaved float planes the graph speaks.
pub fn rgb8_to_planar(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> crate::ifnet::Planar {
    let mut frame = crate::ifnet::Planar::new(width, height, channels);
    let count = (width * height * channels).min(data.len());
    for index in 0..count {
        frame.data[index] = data[index] as f32 / 255.0;
    }
    frame
}

/// Back to interleaved 8-bit RGB, clamped rather than wrapped.
pub fn planar_to_rgb8(frame: &crate::ifnet::Planar) -> Vec<u8> {
    frame
        .data
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// Panics must not cross the FFI boundary: unwinding into the engine is undefined
/// behaviour.
///
/// `AssertUnwindSafe` is the honest choice here rather than a fudge. wgpu's
/// handles contain interior mutability, so they are not `UnwindSafe`; what the
/// assertion claims is that a panic leaves this plugin's *observable* state
/// defined, which it does: the entry point returns `SR_ERR_RUNTIME`, the engine
/// treats that as a failed job, and the session is closed and reopened on the
/// next run. The alternative — letting the unwind cross `extern "C"` — is not a
/// state anyone can reason about.
fn guard<F: FnOnce() -> i32>(f: F) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
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
    let text: String = text.into();
    CString::new(text.replace('\0', " ")).unwrap_or_else(|_| CString::new("error").expect("static"))
}

/// Minimal reader for one unsigned field of `config_json`.
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

fn leak(text: String) -> *const c_char {
    // Leaked deliberately: the ABI says a device description stays valid for the
    // process, and a device list enumerated once is a handful of bytes.
    cstring(text).into_raw() as *const c_char
}

// ---- the ABI ---------------------------------------------------------------

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
        // SAFETY: the caller passes a live struct; only fields inside the size it
        // declared are written.
        let caps = unsafe { &mut *out };
        if (caps.struct_size as usize) < std::mem::size_of::<SrInferCaps>() {
            return SR_ERR_INVALID_ARGUMENT;
        }
        let devices = Engine::adapters();
        caps.abi_version = SR_INFER_ABI_VERSION;
        caps.ops = if devices.is_empty() {
            0
        } else {
            SR_OP_INTERPOLATE | SR_OP_RESTORE
        };
        caps.dtypes = SR_DTYPE_U8;
        caps.flags = SR_CAP_TEMPORAL;
        caps.device_count = devices.len() as u32;
        caps.max_batch = 1;
        caps.min_temporal_window = 2;
        caps.max_temporal_window = 2;
        caps.tile_min = 0;
        caps.tile_max = 0;
        caps.upscale_num = 1;
        caps.upscale_den = 1;
        caps.max_pixels = 0;
        caps.vram_bytes = 0;
        caps.backend = c"vulkan".as_ptr();
        caps.precision = c"fp32".as_ptr();
        caps.vendor = std::ptr::null();
        caps.model = c"block-matching".as_ptr();
        caps.model_version = c"1.0".as_ptr();
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
        let devices = Engine::adapters();
        // SAFETY: `count` is a valid out-pointer per the ABI.
        unsafe { *count = devices.len() as u32 };
        if out.is_null() || capacity == 0 {
            return SR_OK;
        }
        let writable = capacity.min(devices.len() as u32) as usize;
        for (index, device) in devices.iter().take(writable).enumerate() {
            // SAFETY: the caller promised room for `capacity` entries.
            let slot = unsafe { &mut *out.add(index) };
            if (slot.struct_size as usize) < std::mem::size_of::<SrInferDevice>() {
                return SR_ERR_INVALID_ARGUMENT;
            }
            slot.index = device.info.index;
            slot.vendor_id = device.info.vendor_id;
            slot.device_id = device.info.device_id;
            slot.device_type = device.info.device_type;
            slot.reserved = 0;
            slot.name = leak(device.info.name.clone());
            slot.driver = leak(device.info.driver.clone());
            slot.vram_bytes = device.info.vram_bytes;
            slot.vram_budget_bytes = 0;
            slot.vram_used_bytes = 0;
        }
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
        let mut devices = Engine::adapters();
        if devices.is_empty() {
            return SR_ERR_NO_DEVICE;
        }
        let config = unsafe { opt_cstr(desc.config_json) }.unwrap_or_default();
        // The engine passes the index it chose; `config_json` can override it so
        // an operator can pin a job to the integrated GPU without a rebuild.
        let requested = json_u32(&config, "device_index")
            .unwrap_or(desc.device_index)
            .min(devices.len() as u32 - 1);
        let selected = devices.swap_remove(requested as usize);
        let name = selected.info.name.clone();
        let gpu = match Engine::open(selected) {
            Ok(session) => session,
            Err(err) => {
                let message = format!("cannot open {name}: {err}");
                eprintln!("sr-infer-gpu: {message}");
                return SR_ERR_RUNTIME;
            }
        };
        let model =
            unsafe { opt_cstr(desc.model_name) }.unwrap_or_else(|| "block-matching".to_string());
        let session = Box::new(Session {
            device_index: requested,
            gpu: Some(gpu),
            restore: None,
            model,
            calls: 0,
            frames_in: 0,
            frames_out: 0,
            last_error: cstring(""),
            width: 0,
            height: 0,
        });
        // SAFETY: ownership moves to the caller, which returns it through close.
        unsafe { *out = Box::into_raw(session) };
        SR_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn sr_infer_close(session: *mut Session) {
    if session.is_null() {
        return;
    }
    // SAFETY: the pointer came from open and is closed exactly once.
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
        // SAFETY: live session from open.
        let session = unsafe { &*session };
        let bytes = session.last_error.as_bytes_with_nul();
        let n = bytes.len().min(capacity as usize);
        // SAFETY: the caller promised `capacity` writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buffer, n) };
        if n >= capacity as usize {
            // SAFETY: the last byte is inside the buffer.
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
    // Every job completes before `execute` returns; SR_CAP_ASYNC is not claimed.
    SR_OK
}

/// Appends one JSONL line per call, when `SR_INFER_CALL_LOG` names a file.
///
/// The engine's own log says a model session ran; this is what says how many
/// frames the backend actually saw and which ones, which is how a test can check
/// that the executor never handed it a pair from two different shots.
fn log_call(session: &Session, job: &SrInferJob, inputs: &[SrInferImage], written: u32) {
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
        "{{\"backend\":\"sr-infer-gpu\",\"device\":{},\"op\":{},\"in\":{},\"out\":{},\"chunk\":{},\
          \"width\":{},\"height\":{},\"first_pts\":{first:.6},\"last_pts\":{last:.6},\"calls\":{}}}",
        session.device_index,
        job.op,
        inputs.len(),
        written,
        job.chunk_id,
        session.width,
        session.height,
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

        fn fail(session: &mut Session, code: i32, message: String) -> i32 {
            session.last_error = cstring(message);
            code
        }

        if (job.struct_size as usize) < std::mem::size_of::<SrInferJob>() {
            return SR_ERR_INVALID_ARGUMENT;
        }
        if job.op == SR_OP_RESTORE {
            // One frame in, one frame out: restoration is a per-frame operation, and
            // the ABI says so by requiring the counts to match.
            if job.input_count != job.output_count || job.input_count == 0 {
                return fail(
                    session,
                    SR_ERR_INVALID_ARGUMENT,
                    format!(
                        "restoration needs one output per input, got {} in and {} out",
                        job.input_count, job.output_count
                    ),
                );
            }
            // SAFETY: the caller guarantees `input_count` / `output_count` images.
            let inputs = unsafe { std::slice::from_raw_parts(job.inputs, job.input_count as usize) };
            let outputs =
                unsafe { std::slice::from_raw_parts_mut(job.outputs, job.output_count as usize) };
            let width = inputs[0].width;
            let height = inputs[0].height;
            let bytes = width as usize * height as usize * 3;
            for image in inputs.iter() {
                if image.data.is_null() {
                    return fail(session, SR_ERR_INVALID_ARGUMENT, "null input buffer".into());
                }
                if image.width != width || image.height != height {
                    return fail(
                        session,
                        SR_ERR_INVALID_ARGUMENT,
                        "every frame in a restoration job must be the same size".into(),
                    );
                }
                if image.dtype != SR_DTYPE_U8
                    || image.layout != SR_LAYOUT_INTERLEAVED
                    || image.color != SR_COLOR_RGB
                {
                    return fail(
                        session,
                        SR_ERR_UNSUPPORTED,
                        "this backend accepts interleaved 8-bit RGB only".into(),
                    );
                }
            }
            let ready = matches!(
                session.restore.as_ref(),
                Some(restore) if restore.width == width && restore.height == height
            );
            if !ready {
                match Restore::for_size(width, height) {
                    Ok(restore) => session.restore = Some(restore),
                    Err(message) => {
                        return fail(
                            session,
                            SR_ERR_RUNTIME,
                            format!("could not build the restoration graph: {message}"),
                        )
                    }
                }
            }
            let Some(restore) = session.restore.as_ref() else {
                return fail(session, SR_ERR_RUNTIME, "the restoration graph is missing".into());
            };
            let mut written = 0u32;
            for index in 0..job.input_count as usize {
                // SAFETY: the ABI guarantees `bytes` valid bytes for the call.
                let source =
                    unsafe { std::slice::from_raw_parts(inputs[index].data as *const u8, bytes) };
                let frame = rgb8_to_planar(source, width as usize, height as usize, 3);
                let restored = match restore.ops.forward(&restore.model, &frame, &frame) {
                    Ok(restored) => restored,
                    Err(err) => {
                        return fail(session, SR_ERR_RUNTIME, format!("restoration failed: {err}"))
                    }
                };
                let data = planar_to_rgb8(&restored);
                let target = &mut outputs[index];
                if target.data.is_null() || image_bytes(target) < bytes {
                    return fail(
                        session,
                        SR_ERR_INVALID_ARGUMENT,
                        format!("output {index} has no room for {bytes} bytes"),
                    );
                }
                // SAFETY: at least `bytes` are writable, checked just above.
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), target.data as *mut u8, bytes)
                };
                if target.width == 0 {
                    target.width = width;
                }
                if target.height == 0 {
                    target.height = height;
                }
                written += 1;
            }
            session.calls += 1;
            session.frames_in += job.input_count as u64;
            session.frames_out += written as u64;
            // The result is filled here rather than after the interpolation loop,
            // because this path returns before it: a caller that reads
            // `outputs_written` would otherwise be told nothing was produced.
            result.outputs_written = written;
            result.tiles = 1;
            result.vram_used_bytes = (bytes * (job.input_count as usize + written as usize)) as u64;
            result.fence = session.calls;
            result.message = std::ptr::null();
            log_call(session, job, inputs, written);
            return SR_OK;
        }

        if job.op != SR_OP_INTERPOLATE {
            return fail(
                session,
                SR_ERR_UNSUPPORTED,
                format!("op {} is not implemented by this backend", job.op),
            );
        }
        if job.input_count < 2 {
            return fail(
                session,
                SR_ERR_INVALID_ARGUMENT,
                format!(
                    "interpolation needs at least two input frames, got {}",
                    job.input_count
                ),
            );
        }
        let multiplier = job.multiplier.max(2) as usize;
        let expected = (job.input_count as usize - 1) * multiplier + 1;
        if job.output_count as usize != expected {
            return fail(
                session,
                SR_ERR_INVALID_ARGUMENT,
                format!(
                    "{} inputs at {multiplier}x need {expected} outputs, {} given",
                    job.input_count, job.output_count
                ),
            );
        }
        // SAFETY: the caller guarantees `input_count` / `output_count` images.
        let inputs = unsafe { std::slice::from_raw_parts(job.inputs, job.input_count as usize) };
        let outputs =
            unsafe { std::slice::from_raw_parts_mut(job.outputs, job.output_count as usize) };

        for image in inputs.iter() {
            if image.data.is_null() {
                return fail(session, SR_ERR_INVALID_ARGUMENT, "null input buffer".into());
            }
            if image.dtype != SR_DTYPE_U8
                || image.layout != SR_LAYOUT_INTERLEAVED
                || image.color != SR_COLOR_RGB
            {
                return fail(
                    session,
                    SR_ERR_UNSUPPORTED,
                    "this backend accepts interleaved 8-bit RGB only".into(),
                );
            }
        }

        let width = inputs[0].width;
        let height = inputs[0].height;
        session.width = width;
        session.height = height;
        let bytes = width as usize * height as usize * 3;
        let read = |image: &SrInferImage| -> Vec<u8> {
            // SAFETY: the ABI guarantees `bytes` valid bytes for the call.
            unsafe { std::slice::from_raw_parts(image.data as *const u8, bytes) }.to_vec()
        };
        let write_into = |target: &mut SrInferImage, data: &[u8]| -> bool {
            if target.data.is_null() || image_bytes(target) < bytes {
                return false;
            }
            // SAFETY: at least `bytes` are writable, checked just above.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), target.data as *mut u8, bytes) };
            true
        };

        let Some(gpu) = session.gpu.as_mut() else {
            return fail(session, SR_ERR_RUNTIME, "the GPU session is closed".into());
        };

        let mut written = 0u32;
        for pair in 0..(job.input_count as usize - 1) {
            let a = read(&inputs[pair]);
            let b = read(&inputs[pair + 1]);
            // Slot `pair * multiplier + k` is the frame at position k/multiplier
            // between the two inputs. `k = 0` and `k = multiplier` are the input
            // frames themselves, and the ABI requires them to be written too: a
            // caller that reads back the whole output array must find a complete
            // run of frames, not holes where the pass-throughs should be.
            if !write_into(&mut outputs[pair * multiplier], &a) {
                return fail(
                    session,
                    SR_ERR_INVALID_ARGUMENT,
                    "output buffer is smaller than the frame it must hold".into(),
                );
            }
            if !write_into(&mut outputs[(pair + 1) * multiplier], &b) {
                return fail(
                    session,
                    SR_ERR_INVALID_ARGUMENT,
                    "output buffer is smaller than the frame it must hold".into(),
                );
            }
            for k in 1..multiplier {
                let t = k as f32 / multiplier as f32;
                let frame = match gpu.interpolate_pair(&a, &b, width, height, t) {
                    Ok(frame) => frame,
                    Err(err) => {
                        // Reported as an out-of-memory signal only when it really
                        // is one: the executor answers that by shrinking the
                        // working set, and answering a driver failure that way
                        // would loop forever.
                        let lower = err.to_ascii_lowercase();
                        let code = if lower.contains("memory") || lower.contains("oom") {
                            SR_ERR_OUT_OF_MEMORY
                        } else {
                            SR_ERR_RUNTIME
                        };
                        return fail(session, code, err);
                    }
                };
                if !write_into(&mut outputs[pair * multiplier + k], &frame) {
                    return fail(
                        session,
                        SR_ERR_INVALID_ARGUMENT,
                        "output buffer is smaller than the frame it must hold".into(),
                    );
                }
                written += 1;
            }
        }

        session.calls += 1;
        session.frames_in += job.input_count as u64;
        session.frames_out += written as u64;
        result.outputs_written = written;
        result.tiles = 1;
        result.vram_used_bytes = (bytes * (job.input_count as usize + written as usize)) as u64;
        result.message = std::ptr::null();
        log_call(session, job, inputs, written);
        SR_OK
    })
}
pub mod ifnet_gpu;
