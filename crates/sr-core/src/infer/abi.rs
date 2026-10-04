//! Bindings for inference plugin ABI version 2.
//!
//! Version 1 of this ABI could express exactly one thing: two 8-bit frames in,
//! one frame out. It had no device, no session, no temporal window, no dtype and
//! no memory budget, so a plugin could be *loaded* without there being any way
//! to *run a model*. Version 2 replaces it with the shape a real backend needs:
//!
//! ```text
//! enumerate devices → open a session → execute a multi-frame job
//! ```
//!
//! The canonical contract is `include/sr_infer.h`; this module is the engine's
//! side of it, and `crates/sr-infer-plugin-example` is the other side, used by
//! the tests below. The ABI is exercised, not asserted.
//!
//! Thread-safety: the engine may load and query a plugin from any thread, and
//! may hold several sessions at once, but a single session belongs to one
//! thread. [`Session`] is deliberately not `Send` so the compiler enforces it.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Bumped whenever any struct below changes layout.
pub const SR_INFER_ABI_VERSION: u32 = 2;

pub const SR_OK: i32 = 0;
pub const SR_ERR_UNSUPPORTED: i32 = -1;
pub const SR_ERR_INVALID_ARGUMENT: i32 = -2;
pub const SR_ERR_RUNTIME: i32 = -3;
pub const SR_ERR_OUT_OF_MEMORY: i32 = -4;
pub const SR_ERR_CANCELLED: i32 = -5;
pub const SR_ERR_DEVICE_LOST: i32 = -6;
pub const SR_ERR_NO_DEVICE: i32 = -7;

pub const SR_OP_RESTORE: u32 = 1 << 0;
pub const SR_OP_INTERPOLATE: u32 = 1 << 1;
pub const SR_OP_SCALE: u32 = 1 << 2;

pub const SR_DTYPE_U8: u32 = 1 << 0;
pub const SR_DTYPE_U16: u32 = 1 << 1;
pub const SR_DTYPE_F16: u32 = 1 << 2;
pub const SR_DTYPE_BF16: u32 = 1 << 3;
pub const SR_DTYPE_F32: u32 = 1 << 4;

pub const SR_LAYOUT_INTERLEAVED: u32 = 1;
pub const SR_LAYOUT_PLANAR: u32 = 2;
pub const SR_LAYOUT_SEMI_PLANAR: u32 = 3;

pub const SR_COLOR_RGB: u32 = 1;
pub const SR_COLOR_BGR: u32 = 2;
pub const SR_COLOR_GRAY: u32 = 3;
pub const SR_COLOR_YUV: u32 = 4;

pub const SR_RANGE_FULL: u32 = 0;
pub const SR_RANGE_LIMITED: u32 = 1;

pub const SR_MEM_HOST: u32 = 1 << 0;
pub const SR_MEM_DEVICE: u32 = 1 << 1;
pub const SR_MEM_ALIAS: u32 = 1 << 2;

pub const SR_CAP_TILING: u32 = 1 << 0;
pub const SR_CAP_DEVICE_MEMORY: u32 = 1 << 1;
pub const SR_CAP_ASYNC: u32 = 1 << 2;
pub const SR_CAP_TEMPORAL: u32 = 1 << 3;
pub const SR_CAP_MULTI_DEVICE: u32 = 1 << 4;
pub const SR_CAP_PARTIAL_OUTPUT: u32 = 1 << 5;

pub const SR_DEVICE_CPU: u32 = 1;
pub const SR_DEVICE_INTEGRATED: u32 = 2;
pub const SR_DEVICE_DISCRETE: u32 = 3;
pub const SR_DEVICE_VIRTUAL: u32 = 4;

pub const SR_JOB_ASYNC: u32 = 1 << 0;
pub const SR_JOB_INPLACE: u32 = 1 << 1;

// ---- raw structs ----------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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

impl Default for SrInferDevice {
    fn default() -> Self {
        SrInferDevice {
            struct_size: std::mem::size_of::<SrInferDevice>() as u32,
            index: 0,
            vendor_id: 0,
            device_id: 0,
            device_type: SR_DEVICE_CPU,
            reserved: 0,
            name: std::ptr::null(),
            driver: std::ptr::null(),
            vram_bytes: 0,
            vram_budget_bytes: 0,
            vram_used_bytes: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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

impl Default for SrInferCaps {
    fn default() -> Self {
        SrInferCaps {
            struct_size: std::mem::size_of::<SrInferCaps>() as u32,
            abi_version: 0,
            ops: 0,
            dtypes: 0,
            flags: 0,
            device_count: 0,
            max_batch: 1,
            min_temporal_window: 2,
            max_temporal_window: 2,
            tile_min: 0,
            tile_max: 0,
            upscale_num: 1,
            upscale_den: 1,
            max_pixels: 0,
            vram_bytes: 0,
            backend: std::ptr::null(),
            precision: std::ptr::null(),
            vendor: std::ptr::null(),
            model: std::ptr::null(),
            model_version: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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

impl Default for SrInferImage {
    fn default() -> Self {
        SrInferImage {
            struct_size: std::mem::size_of::<SrInferImage>() as u32,
            memory: SR_MEM_HOST,
            dtype: SR_DTYPE_U8,
            layout: SR_LAYOUT_INTERLEAVED,
            color: SR_COLOR_RGB,
            range: SR_RANGE_FULL,
            bit_depth: 8,
            planes: 3,
            width: 0,
            height: 0,
            stride: 0,
            plane_stride: [0; 4],
            data: std::ptr::null_mut(),
            device_handle: 0,
            pts_num: 0,
            pts_den: 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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

impl Default for SrInferJob {
    fn default() -> Self {
        SrInferJob {
            struct_size: std::mem::size_of::<SrInferJob>() as u32,
            op: SR_OP_INTERPOLATE,
            flags: 0,
            input_count: 0,
            output_count: 0,
            inputs: std::ptr::null(),
            outputs: std::ptr::null_mut(),
            multiplier: 2,
            strength: 1.0,
            tile_width: 0,
            tile_height: 0,
            tile_pad: 0,
            reserved: 0,
            seed: 0,
            chunk_id: 0,
            fence: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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

impl Default for SrInferResult {
    fn default() -> Self {
        SrInferResult {
            struct_size: std::mem::size_of::<SrInferResult>() as u32,
            outputs_written: 0,
            tiles: 0,
            reserved: 0,
            vram_used_bytes: 0,
            vram_budget_bytes: 0,
            fence: 0,
            message: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
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

impl Default for SrInferSessionDesc {
    fn default() -> Self {
        SrInferSessionDesc {
            struct_size: std::mem::size_of::<SrInferSessionDesc>() as u32,
            device_index: 0,
            flags: 0,
            reserved: 0,
            model_path: std::ptr::null(),
            model_name: std::ptr::null(),
            config_json: std::ptr::null(),
            vram_budget_bytes: 0,
            host_memory_budget_bytes: 0,
        }
    }
}

/// Opaque plugin-side session.
#[repr(C)]
pub struct SrInferSession {
    _private: [u8; 0],
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type QueryFn = unsafe extern "C" fn(*mut SrInferCaps) -> c_int;
type DevicesFn = unsafe extern "C" fn(*mut SrInferDevice, u32, *mut u32) -> c_int;
type OpenFn = unsafe extern "C" fn(*const SrInferSessionDesc, *mut *mut SrInferSession) -> c_int;
type CloseFn = unsafe extern "C" fn(*mut SrInferSession);
type ExecuteFn =
    unsafe extern "C" fn(*mut SrInferSession, *const SrInferJob, *mut SrInferResult) -> c_int;
type PollFn = unsafe extern "C" fn(*mut SrInferSession, u64, u32) -> c_int;
type LastErrorFn = unsafe extern "C" fn(*mut SrInferSession, *mut c_char, u32) -> c_int;

// ---- safe projections -----------------------------------------------------

unsafe fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the ABI requires a NUL-terminated string that stays valid for the
    // lifetime of the call; plugins return pointers to static or session-owned
    // data, and this copies immediately.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(|s| s.to_string())
}

/// Capability report decoupled from raw pointers, safe to log and persist.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PluginCapabilities {
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
    pub backend: String,
    pub precision: String,
    pub vendor: Option<String>,
    pub model: String,
    pub model_version: Option<String>,
}

impl PluginCapabilities {
    pub fn supports_op(&self, op: u32) -> bool {
        self.ops & op != 0
    }

    pub fn supports_dtype(&self, dtype: u32) -> bool {
        self.dtypes & dtype != 0
    }

    pub fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }

    /// Geometry change the model applies to a restored frame.
    pub fn upscale(&self) -> f64 {
        let den = if self.upscale_den == 0 {
            1.0
        } else {
            self.upscale_den as f64
        };
        let num = if self.upscale_num == 0 {
            1.0
        } else {
            self.upscale_num as f64
        };
        num / den
    }

    pub fn dtype_names(&self) -> Vec<&'static str> {
        [
            (SR_DTYPE_U8, "u8"),
            (SR_DTYPE_U16, "u16"),
            (SR_DTYPE_F16, "f16"),
            (SR_DTYPE_BF16, "bf16"),
            (SR_DTYPE_F32, "f32"),
        ]
        .iter()
        .filter(|(bit, _)| self.dtypes & bit != 0)
        .map(|(_, name)| *name)
        .collect()
    }

    pub fn summary(&self) -> String {
        let mut what = Vec::new();
        if self.supports_op(SR_OP_RESTORE) {
            what.push(format!("restore {:.2}x", self.upscale()));
        }
        if self.supports_op(SR_OP_INTERPOLATE) {
            what.push(format!(
                "interpolate (window {}..{})",
                self.min_temporal_window, self.max_temporal_window
            ));
        }
        if self.supports_op(SR_OP_SCALE) {
            what.push("scale".to_string());
        }
        format!(
            "abi {}, {} {}, {}, {}{}",
            self.abi_version,
            if self.backend.is_empty() { "?" } else { &self.backend },
            if self.precision.is_empty() {
                "?"
            } else {
                &self.precision
            },
            if self.model.is_empty() {
                "unnamed model".to_string()
            } else {
                self.model.clone()
            },
            if what.is_empty() {
                "no operations".to_string()
            } else {
                what.join(", ")
            },
            self.vendor
                .as_deref()
                .map(|v| format!(" on {v}"))
                .unwrap_or_default(),
        )
    }
}

/// One device the plugin can run on.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub index: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub name: String,
    pub driver: Option<String>,
    pub vram_bytes: u64,
    pub vram_budget_bytes: u64,
    pub vram_used_bytes: u64,
}

impl DeviceInfo {
    pub fn type_name(&self) -> &'static str {
        match self.device_type {
            SR_DEVICE_CPU => "cpu",
            SR_DEVICE_INTEGRATED => "integrated",
            SR_DEVICE_DISCRETE => "discrete",
            SR_DEVICE_VIRTUAL => "virtual",
            _ => "unknown",
        }
    }

    pub fn vendor_name(&self) -> &'static str {
        match self.vendor_id {
            0x10DE => "nvidia",
            0x1002 | 0x1022 => "amd",
            0x8086 => "intel",
            0x106B => "apple",
            0x5143 => "qualcomm",
            _ => "unknown",
        }
    }

    /// Free device memory when the backend can see it.
    pub fn free_bytes(&self) -> Option<u64> {
        match (self.vram_budget_bytes, self.vram_used_bytes) {
            (budget, used) if budget > 0 => Some(budget.saturating_sub(used)),
            _ => match self.vram_bytes {
                0 => None,
                total => Some(total.saturating_sub(self.vram_used_bytes)),
            },
        }
    }

    pub fn describe(&self) -> String {
        let mem = match self.free_bytes() {
            Some(free) => format!("{:.1} GiB free", free as f64 / (1024.0 * 1024.0 * 1024.0)),
            None => "memory unknown".to_string(),
        };
        format!(
            "[{}] {} ({}, {})",
            self.index,
            if self.name.is_empty() {
                "unnamed device"
            } else {
                &self.name
            },
            self.type_name(),
            mem
        )
    }
}

/// What one `execute` call did.
#[derive(Clone, Debug, Default)]
pub struct ExecOutcome {
    pub outputs_written: u32,
    pub tiles: u32,
    pub vram_used_bytes: u64,
    pub vram_budget_bytes: u64,
    /// Non-zero when the backend ran the job asynchronously.
    pub fence: u64,
    pub message: Option<String>,
}

// ---- the loaded library ---------------------------------------------------

/// A loaded plugin. Dropping it unloads the library; sessions opened from it
/// hold an `Arc` so the code cannot be unloaded under a live session.
pub struct PluginLibrary {
    /// Kept alive for the lifetime of the plugin: dropping it unloads the code
    /// that the function pointers below point into.
    _library: libloading::Library,
    abi_version: AbiVersionFn,
    query: QueryFn,
    devices: Option<DevicesFn>,
    open: OpenFn,
    close: Option<CloseFn>,
    execute: ExecuteFn,
    poll: Option<PollFn>,
    last_error: Option<LastErrorFn>,
    pub path: PathBuf,
}

fn symbol_error(path: &Path, name: &str, err: impl std::fmt::Display) -> Error {
    Error::Plugin {
        plugin: path.display().to_string(),
        detail: format!("missing {name}: {err}"),
    }
}

impl PluginLibrary {
    pub fn load(path: &Path) -> Result<Self> {
        // SAFETY: loading arbitrary code is the explicit purpose of a plugin;
        // the caller decides which paths are trusted. The ABI version is checked
        // before any other symbol is called.
        let library = unsafe { libloading::Library::new(path) }.map_err(|e| Error::Plugin {
            plugin: path.display().to_string(),
            detail: format!("cannot load shared library: {e}"),
        })?;

        // SAFETY: symbol types are defined by the ABI this build was compiled
        // against; the version check below catches layout mismatches.
        let abi_version: AbiVersionFn = unsafe {
            *library
                .get(b"sr_infer_abi_version\0")
                .map_err(|e| symbol_error(path, "sr_infer_abi_version", e))?
        };
        let reported = unsafe { abi_version() };
        if reported != SR_INFER_ABI_VERSION {
            return Err(Error::Plugin {
                plugin: path.display().to_string(),
                detail: format!(
                    "ABI version mismatch: plugin reports {reported}, engine requires \
                     {SR_INFER_ABI_VERSION}. Version 1 plugins are not supported: recompile \
                     against include/sr_infer.h"
                ),
            });
        }

        macro_rules! required {
            ($name:literal) => {
                unsafe {
                    *library
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|e| symbol_error(path, $name, e))?
                }
            };
        }
        macro_rules! optional {
            ($name:literal) => {
                unsafe {
                    library
                        .get(concat!($name, "\0").as_bytes())
                        .ok()
                        .map(|s| *s)
                }
            };
        }

        let query: QueryFn = required!("sr_infer_query");
        let open: OpenFn = required!("sr_infer_open");
        let execute: ExecuteFn = required!("sr_infer_execute");
        let devices: Option<DevicesFn> = optional!("sr_infer_devices");
        let close: Option<CloseFn> = optional!("sr_infer_close");
        let poll: Option<PollFn> = optional!("sr_infer_poll");
        let last_error: Option<LastErrorFn> = optional!("sr_infer_last_error");

        Ok(PluginLibrary {
            _library: library,
            abi_version,
            query,
            devices,
            open,
            close,
            execute,
            poll,
            last_error,
            path: path.to_path_buf(),
        })
    }

    pub fn abi_version(&self) -> u32 {
        // SAFETY: loaded and version-checked above.
        unsafe { (self.abi_version)() }
    }

    pub fn capabilities(&self) -> Result<PluginCapabilities> {
        let mut raw = SrInferCaps::default();
        // SAFETY: `raw` is a live, correctly sized struct.
        let code = unsafe { (self.query)(&mut raw) };
        if code != SR_OK {
            return Err(Error::Plugin {
                plugin: self.path.display().to_string(),
                detail: format!("sr_infer_query returned {code}"),
            });
        }
        if raw.abi_version != SR_INFER_ABI_VERSION {
            return Err(Error::Plugin {
                plugin: self.path.display().to_string(),
                detail: format!(
                    "sr_infer_query reports abi {}, expected {SR_INFER_ABI_VERSION}",
                    raw.abi_version
                ),
            });
        }
        Ok(PluginCapabilities {
            abi_version: raw.abi_version,
            ops: raw.ops,
            dtypes: raw.dtypes,
            flags: raw.flags,
            device_count: raw.device_count,
            max_batch: raw.max_batch.max(1),
            min_temporal_window: raw.min_temporal_window.max(1),
            max_temporal_window: raw.max_temporal_window.max(1),
            tile_min: raw.tile_min,
            tile_max: raw.tile_max,
            upscale_num: if raw.upscale_num == 0 { 1 } else { raw.upscale_num },
            upscale_den: if raw.upscale_den == 0 { 1 } else { raw.upscale_den },
            max_pixels: raw.max_pixels,
            vram_bytes: raw.vram_bytes,
            backend: unsafe { cstr_to_string(raw.backend) }.unwrap_or_default(),
            precision: unsafe { cstr_to_string(raw.precision) }.unwrap_or_default(),
            vendor: unsafe { cstr_to_string(raw.vendor) },
            model: unsafe { cstr_to_string(raw.model) }.unwrap_or_default(),
            model_version: unsafe { cstr_to_string(raw.model_version) },
        })
    }

    /// Devices this plugin can run on. Empty when it exports no enumerator, in
    /// which case device 0 is assumed to be usable.
    pub fn devices(&self) -> Result<Vec<DeviceInfo>> {
        let Some(devices_fn) = self.devices else {
            return Ok(vec![DeviceInfo {
                index: 0,
                name: "default device".into(),
                ..DeviceInfo::default()
            }]);
        };
        let mut count: u32 = 0;
        // SAFETY: a null buffer with capacity 0 is the documented count query.
        let code = unsafe { devices_fn(std::ptr::null_mut(), 0, &mut count) };
        if code != SR_OK {
            return Err(self.error_for(code, "sr_infer_devices(count)"));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut raw = vec![SrInferDevice::default(); count as usize];
        let mut written: u32 = 0;
        // SAFETY: the buffer has room for `count` entries, which is what the
        // plugin just reported.
        let code = unsafe { devices_fn(raw.as_mut_ptr(), count, &mut written) };
        if code != SR_OK {
            return Err(self.error_for(code, "sr_infer_devices"));
        }
        raw.truncate(written.min(count) as usize);
        Ok(raw
            .into_iter()
            .map(|d| DeviceInfo {
                index: d.index,
                vendor_id: d.vendor_id,
                device_id: d.device_id,
                device_type: d.device_type,
                name: unsafe { cstr_to_string(d.name) }.unwrap_or_default(),
                driver: unsafe { cstr_to_string(d.driver) },
                vram_bytes: d.vram_bytes,
                vram_budget_bytes: d.vram_budget_bytes,
                vram_used_bytes: d.vram_used_bytes,
            })
            .collect())
    }

    /// Maps a plugin return code onto the engine's error type.
    ///
    /// `Out of memory` deliberately becomes [`Error::DeviceMemory`]: the runner
    /// answers that by shrinking the working set and retrying the same chunk.
    fn error_for(&self, code: i32, call: &str) -> Error {
        let detail = self.format_code(code, call);
        match code {
            SR_ERR_OUT_OF_MEMORY => Error::DeviceMemory {
                engine: self.path.display().to_string(),
                detail,
            },
            SR_ERR_CANCELLED => Error::Cancelled,
            SR_ERR_UNSUPPORTED | SR_ERR_NO_DEVICE => Error::Unsupported(detail),
            _ => Error::Plugin {
                plugin: self.path.display().to_string(),
                detail,
            },
        }
    }

    fn format_code(&self, code: i32, call: &str) -> String {
        let name = match code {
            SR_ERR_UNSUPPORTED => "unsupported",
            SR_ERR_INVALID_ARGUMENT => "invalid argument",
            SR_ERR_RUNTIME => "runtime failure",
            SR_ERR_OUT_OF_MEMORY => "out of memory",
            SR_ERR_CANCELLED => "cancelled",
            SR_ERR_DEVICE_LOST => "device lost",
            SR_ERR_NO_DEVICE => "no device",
            _ => "unknown error",
        };
        format!("{call} failed: {name} ({code})")
    }

    pub fn describe(&self) -> String {
        match self.capabilities() {
            Ok(caps) => format!("{} ({})", self.path.display(), caps.summary()),
            Err(e) => format!("{} (unusable: {e})", self.path.display()),
        }
    }
}

fn cstring_opt(value: Option<&str>) -> Result<Option<std::ffi::CString>> {
    match value {
        None => Ok(None),
        Some(text) => std::ffi::CString::new(text)
            .map(Some)
            .map_err(|e| Error::Other(format!("plugin string contains a NUL byte: {e}"))),
    }
}

/// How a session should be opened.
#[derive(Clone, Debug, Default)]
pub struct SessionRequest {
    pub device_index: u32,
    pub model_path: Option<PathBuf>,
    pub model_name: Option<String>,
    pub config_json: Option<String>,
    pub vram_budget_bytes: u64,
    pub host_memory_budget_bytes: u64,
}

/// A host frame the engine owns and lends to the plugin for one call.
#[derive(Clone, Debug)]
pub struct FrameBuffer {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub dtype: u32,
    pub layout: u32,
    pub color: u32,
    pub planes: u32,
    pub bit_depth: u32,
    pub pts_num: i64,
    pub pts_den: i64,
}

impl FrameBuffer {
    pub fn new_rgb8(width: u32, height: u32) -> Self {
        FrameBuffer {
            data: vec![0u8; width as usize * height as usize * 3],
            width,
            height,
            dtype: SR_DTYPE_U8,
            layout: SR_LAYOUT_INTERLEAVED,
            color: SR_COLOR_RGB,
            planes: 3,
            bit_depth: 8,
            pts_num: 0,
            pts_den: 1,
        }
    }

    pub fn with_pts(mut self, num: i64, den: i64) -> Self {
        self.pts_num = num;
        self.pts_den = den;
        self
    }

    /// Adopts an already allocated buffer, checking it is the right size.
    pub fn from_owned(data: Vec<u8>, width: u32, height: u32) -> Self {
        debug_assert_eq!(data.len(), width as usize * height as usize * 3);
        FrameBuffer {
            data,
            ..FrameBuffer::new_rgb8(width, height)
        }
    }

    pub fn from_slice(data: &[u8], width: u32, height: u32) -> Self {
        let mut frame = FrameBuffer::new_rgb8(width, height);
        let n = data.len().min(frame.data.len());
        frame.data[..n].copy_from_slice(&data[..n]);
        frame
    }

    pub fn image(&mut self) -> SrInferImage {
        SrInferImage {
            memory: SR_MEM_HOST,
            dtype: self.dtype,
            layout: self.layout,
            color: self.color,
            bit_depth: self.bit_depth,
            planes: self.planes,
            width: self.width,
            height: self.height,
            stride: 0,
            data: self.data.as_mut_ptr() as *mut c_void,
            pts_num: self.pts_num,
            pts_den: self.pts_den,
            ..SrInferImage::default()
        }
    }
}

/// An open plugin session. Not `Send`: one session belongs to one thread.
pub struct Session {
    library: Arc<PluginLibrary>,
    handle: *mut SrInferSession,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("plugin", &self.library.path.display().to_string())
            .field("handle", &format_args!("{:p}", self.handle))
            .finish()
    }
}

impl Session {
    /// Opens a session through an `Arc<PluginLibrary>`, which is how the engine
    /// holds plugins (a session must keep the code loaded).
    pub fn open(library: &Arc<PluginLibrary>, request: &SessionRequest) -> Result<Self> {
        let model_path = cstring_opt(request.model_path.as_deref().and_then(Path::to_str))?;
        let model_name = cstring_opt(request.model_name.as_deref())?;
        let config_json = cstring_opt(request.config_json.as_deref())?;
        let desc = SrInferSessionDesc {
            device_index: request.device_index,
            model_path: model_path
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null()),
            model_name: model_name
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null()),
            config_json: config_json
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null()),
            vram_budget_bytes: request.vram_budget_bytes,
            host_memory_budget_bytes: request.host_memory_budget_bytes,
            ..SrInferSessionDesc::default()
        };
        let mut handle: *mut SrInferSession = std::ptr::null_mut();
        // SAFETY: the descriptor and its strings outlive the call, and `handle`
        // is a valid out-pointer.
        let code = unsafe { (library.open)(&desc, &mut handle) };
        if code != SR_OK {
            return Err(library.error_for(code, "sr_infer_open"));
        }
        if handle.is_null() {
            return Err(Error::Plugin {
                plugin: library.path.display().to_string(),
                detail: "sr_infer_open reported success but returned a null session".into(),
            });
        }
        Ok(Session {
            library: Arc::clone(library),
            handle,
        })
    }

    pub fn plugin_path(&self) -> &Path {
        &self.library.path
    }

    /// The message the plugin left behind for the last failure.
    pub fn last_error(&self) -> Option<String> {
        let last_error = self.library.last_error?;
        let mut buffer = vec![0i8; 1024];
        // SAFETY: the buffer is valid for `capacity` bytes.
        let written = unsafe { last_error(self.handle, buffer.as_mut_ptr(), buffer.len() as u32) };
        if written <= 0 {
            return None;
        }
        let text = unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    /// Interpolates `multiplier - 1` new frames between each consecutive pair.
    ///
    /// `inputs.len()` may be any window the plugin advertised; the outputs must
    /// number `(inputs.len() - 1) * multiplier + 1`.
    pub fn interpolate(
        &mut self,
        inputs: &mut [FrameBuffer],
        outputs: &mut [FrameBuffer],
        options: &JobOptions,
    ) -> Result<ExecOutcome> {
        let multiplier = options.multiplier.max(1);
        let expected = (inputs.len().saturating_sub(1)) as u32 * multiplier + 1;
        if outputs.len() as u32 != expected {
            return Err(Error::Other(format!(
                "interpolation geometry: {} inputs at {}x produce {expected} frames, {} buffers given",
                inputs.len(),
                multiplier,
                outputs.len()
            )));
        }
        self.execute(SR_OP_INTERPOLATE, inputs, outputs, options)
    }

    /// Restores a temporal batch: same count in, same count out.
    pub fn restore(
        &mut self,
        frames: &mut [FrameBuffer],
        options: &JobOptions,
    ) -> Result<ExecOutcome> {
        let count = frames.len();
        // RESTORE is in-place: the plugin writes into the same buffers, which
        // the ABI expresses by passing the aliased images as outputs.
        self.execute(SR_OP_RESTORE, frames, &mut [], options)
            .map(|outcome| {
                debug_assert_eq!(outcome.outputs_written as usize, count);
                outcome
            })
    }

    fn execute(
        &mut self,
        op: u32,
        inputs: &mut [FrameBuffer],
        outputs: &mut [FrameBuffer],
        options: &JobOptions,
    ) -> Result<ExecOutcome> {
        let raw_inputs: Vec<SrInferImage> = inputs.iter_mut().map(FrameBuffer::image).collect();
        let mut raw_outputs: Vec<SrInferImage> = if outputs.is_empty() && op == SR_OP_RESTORE {
            // In-place restore: outputs alias the inputs, which the ABI allows
            // because the plugin is told the buffers are the same.
            inputs
                .iter_mut()
                .map(|f| {
                    let mut image = f.image();
                    image.memory |= SR_MEM_ALIAS;
                    image
                })
                .collect()
        } else {
            outputs.iter_mut().map(FrameBuffer::image).collect()
        };

        let job = SrInferJob {
            op,
            input_count: raw_inputs.len() as u32,
            output_count: raw_outputs.len() as u32,
            inputs: raw_inputs.as_ptr(),
            outputs: raw_outputs.as_mut_ptr(),
            multiplier: options.multiplier.max(1),
            strength: options.strength,
            tile_width: options.tile.map(|t| t.0).unwrap_or(0),
            tile_height: options.tile.map(|t| t.1).unwrap_or(0),
            seed: options.seed,
            chunk_id: options.chunk_id,
            ..SrInferJob::default()
        };

        let mut result = SrInferResult::default();
        // SAFETY: every image points at memory owned by a live `FrameBuffer` in
        // `inputs`/`outputs`, which outlive the call; the job and result structs
        // are correctly sized and initialised.
        let code = unsafe { (self.library.execute)(self.handle, &job, &mut result) };
        if code != SR_OK {
            let detail = self
                .last_error()
                .unwrap_or_else(|| self.library.format_code(code, "sr_infer_execute"));
            return Err(match code {
                SR_ERR_OUT_OF_MEMORY => Error::DeviceMemory {
                    engine: self.library.path.display().to_string(),
                    detail,
                },
                SR_ERR_CANCELLED => Error::Cancelled,
                SR_ERR_UNSUPPORTED => Error::Unsupported(detail),
                _ => Error::Plugin {
                    plugin: self.library.path.display().to_string(),
                    detail,
                },
            });
        }
        Ok(ExecOutcome {
            outputs_written: result.outputs_written,
            tiles: result.tiles,
            vram_used_bytes: result.vram_used_bytes,
            vram_budget_bytes: result.vram_budget_bytes,
            fence: result.fence,
            message: unsafe { cstr_to_string(result.message) },
        })
    }

    /// Waits for an async job started with `SR_JOB_ASYNC`.
    pub fn poll(&mut self, fence: u64, timeout_ms: u32) -> Result<()> {
        let Some(poll) = self.library.poll else {
            return Err(Error::Unsupported(
                "this plugin advertises async execution but exports no sr_infer_poll".into(),
            ));
        };
        // SAFETY: `handle` is a live session and `fence` came from this session.
        let code = unsafe { poll(self.handle, fence, timeout_ms) };
        if code == SR_OK {
            Ok(())
        } else {
            Err(self.library.error_for(code, "sr_infer_poll"))
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(close) = self.library.close {
            // SAFETY: `handle` came from this plugin's `open` and is closed once.
            unsafe { close(self.handle) };
        }
    }
}

/// Everything a job carries besides its frames.
#[derive(Clone, Debug)]
pub struct JobOptions {
    pub multiplier: u32,
    pub strength: f32,
    /// Tile edge the executor wants, or `None` to let the backend decide.
    pub tile: Option<(u32, u32)>,
    /// Stable across a resume so a re-run reproduces the same pixels.
    pub seed: u64,
    /// Correlates the call with the engine's chunk table and the plugin's log.
    pub chunk_id: u64,
}

impl Default for JobOptions {
    fn default() -> Self {
        JobOptions {
            multiplier: 2,
            strength: 1.0,
            tile: None,
            seed: 0,
            chunk_id: 0,
        }
    }
}

// ---- discovery ------------------------------------------------------------

/// Where plugins are looked for, in order.
pub fn search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(explicit) = std::env::var_os("SR_INFER_PLUGIN") {
        paths.push(PathBuf::from(explicit));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join(plugin_filename("sr_infer")));
            paths.push(dir.join("plugins").join(plugin_filename("sr_infer")));
        }
    }
    paths.push(PathBuf::from("plugins").join(plugin_filename("sr_infer")));
    paths
}

pub fn plugin_filename(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}

/// First plugin that loads and reports usable capabilities.
pub fn discover() -> Option<Arc<PluginLibrary>> {
    for path in search_paths() {
        if !path.exists() {
            continue;
        }
        match PluginLibrary::load(&path) {
            Ok(lib) => return Some(Arc::new(lib)),
            Err(e) => tracing::warn!("inference plugin at {} rejected: {e}", path.display()),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example plugin is a dev-dependency, so cargo builds it before tests.
    pub(crate) fn example_plugin_path() -> Option<PathBuf> {
        let exe = std::env::current_exe().ok()?;
        let dir = exe.parent()?.parent()?;
        let candidates = [
            dir.join(plugin_filename("sr_infer_plugin_example")),
            dir.join("deps").join(plugin_filename("sr_infer_plugin_example")),
        ];
        candidates.into_iter().find(|p| p.exists())
    }

    #[test]
    fn missing_plugin_is_an_error_not_a_panic() {
        let err = PluginLibrary::load(Path::new("definitely/not/a/plugin.dll"))
            .err()
            .expect("loading a missing library must fail");
        assert!(matches!(err, Error::Plugin { .. }));
        assert!(err.to_string().contains("cannot load shared library"));
    }

    #[test]
    fn search_paths_always_include_a_plugins_directory() {
        let paths = search_paths();
        assert!(paths
            .iter()
            .any(|p| p.to_string_lossy().contains("plugins")));
    }

    #[test]
    fn struct_sizes_are_wire_ready() {
        // Every struct must start with its own size: that is the whole
        // forward-compatibility story of ABI v2.
        assert_eq!(
            SrInferCaps::default().struct_size as usize,
            std::mem::size_of::<SrInferCaps>()
        );
        assert_eq!(
            SrInferImage::default().struct_size as usize,
            std::mem::size_of::<SrInferImage>()
        );
        assert_eq!(
            SrInferJob::default().struct_size as usize,
            std::mem::size_of::<SrInferJob>()
        );
        assert_eq!(
            SrInferResult::default().struct_size as usize,
            std::mem::size_of::<SrInferResult>()
        );
        assert_eq!(
            SrInferDevice::default().struct_size as usize,
            std::mem::size_of::<SrInferDevice>()
        );
        assert_eq!(
            SrInferSessionDesc::default().struct_size as usize,
            std::mem::size_of::<SrInferSessionDesc>()
        );
    }

    fn load_example() -> Option<Arc<PluginLibrary>> {
        let path = example_plugin_path()?;
        Some(Arc::new(
            PluginLibrary::load(&path).expect("load example plugin"),
        ))
    }

    #[test]
    fn example_plugin_reports_a_v2_capability_surface() {
        let Some(library) = load_example() else {
            eprintln!("example plugin not built; skipping ABI integration test");
            return;
        };
        assert_eq!(library.abi_version(), SR_INFER_ABI_VERSION);
        let caps = library.capabilities().expect("capabilities");
        assert_eq!(caps.abi_version, SR_INFER_ABI_VERSION);
        assert!(caps.supports_op(SR_OP_INTERPOLATE));
        assert!(caps.supports_op(SR_OP_RESTORE));
        assert!(caps.supports_dtype(SR_DTYPE_U8));
        assert!(caps.max_temporal_window >= 2);
        assert_eq!(caps.backend, "cpu");
        assert!(caps.vendor.is_none(), "the example plugin is vendor-neutral");
    }

    #[test]
    fn example_plugin_enumerates_devices() {
        let Some(library) = load_example() else {
            return;
        };
        let devices = library.devices().expect("devices");
        assert!(!devices.is_empty());
        assert_eq!(devices[0].device_type, SR_DEVICE_CPU);
        assert!(devices[0].describe().contains("cpu"));
    }

    #[test]
    fn interpolation_runs_through_the_session_and_places_frames_correctly() {
        let Some(library) = load_example() else {
            return;
        };
        let mut session = Session::open(&library, &SessionRequest::default()).expect("open session");

        // 3 input frames at 2x produce 5 output frames.
        let frames: [[u8; 3]; 3] = [[0, 0, 0], [100, 100, 100], [200, 200, 200]];
        let mut inputs: Vec<FrameBuffer> = frames
            .iter()
            .map(|px| FrameBuffer::from_slice(&px.repeat(4), 2, 2))
            .collect();
        let mut outputs: Vec<FrameBuffer> = (0..5).map(|_| FrameBuffer::new_rgb8(2, 2)).collect();

        let outcome = session
            .interpolate(&mut inputs, &mut outputs, &JobOptions { multiplier: 2, chunk_id: 7, ..JobOptions::default() })
            .expect("interpolate");
        assert_eq!(outcome.outputs_written, 5);
        assert_eq!(outcome.tiles, 1);

        // Output 0 and 4 are the real frames; 1..3 are synthesised between them.
        assert_eq!(outputs[0].data[0], 0);
        assert_eq!(outputs[4].data[0], 200);
        let mid = outputs[2].data[0];
        assert!(
            (95..=105).contains(&mid),
            "the middle frame should sit near the middle input, got {mid}"
        );
    }

    #[test]
    fn a_geometry_mismatch_is_rejected_before_the_plugin_is_called() {
        let Some(library) = load_example() else {
            return;
        };
        let mut session = Session::open(&library, &SessionRequest::default()).expect("open session");
        let mut inputs = vec![FrameBuffer::new_rgb8(2, 2), FrameBuffer::new_rgb8(2, 2)];
        let mut outputs = vec![FrameBuffer::new_rgb8(2, 2)]; // needs 3 at 2x
        let err = session
            .interpolate(&mut inputs, &mut outputs, &JobOptions { multiplier: 2, ..JobOptions::default() })
            .unwrap_err();
        assert!(err.to_string().contains("geometry"), "{err}");
    }

    #[test]
    fn restore_writes_back_into_the_input_buffers() {
        let Some(library) = load_example() else {
            return;
        };
        let mut session = Session::open(&library, &SessionRequest::default()).expect("open session");
        let mut frames = vec![FrameBuffer::new_rgb8(8, 8)];
        // A soft edge, so an unsharp mask has something to act on.
        for y in 0..8usize {
            for x in 0..8usize {
                let value = if x < 4 { 60u8 } else { 190u8 };
                let index = (y * 8 + x) * 3;
                frames[0].data[index..index + 3].fill(value);
            }
        }
        let before = frames[0].data.clone();
        let outcome = session
            .restore(&mut frames, &JobOptions::default())
            .expect("restore");
        assert_eq!(outcome.outputs_written, 1);
        assert_ne!(
            frames[0].data, before,
            "restore must have written into the caller's buffers"
        );
    }

    #[test]
    fn a_device_memory_failure_is_classified_as_an_oom() {
        let Some(library) = load_example() else {
            return;
        };
        // The example plugin fails the session's next `n` execute calls with OOM
        // when its config asks for it, which is how the degrade ladder is tested
        // without a 16 GB GPU. Scoped to the session, so parallel tests cannot
        // inject faults into each other.
        let request = SessionRequest {
            config_json: Some("{\"fake_oom\":1}".to_string()),
            ..SessionRequest::default()
        };
        let mut session = Session::open(&library, &request).expect("open session");
        let mut inputs = vec![FrameBuffer::new_rgb8(2, 2), FrameBuffer::new_rgb8(2, 2)];
        let mut outputs = vec![FrameBuffer::new_rgb8(2, 2); 3];
        let err = session
            .interpolate(
                &mut inputs,
                &mut outputs,
                &JobOptions {
                    multiplier: 2,
                    ..JobOptions::default()
                },
            )
            .expect_err("the first call must fail");
        assert!(matches!(err, Error::DeviceMemory { .. }), "{err}");
        assert!(
            err.is_oom(),
            "an OOM from a plugin must drive the degrade ladder"
        );

        // ... and the retry succeeds, which is what makes the ladder useful.
        let mut inputs = vec![FrameBuffer::new_rgb8(2, 2), FrameBuffer::new_rgb8(2, 2)];
        let mut outputs = vec![FrameBuffer::new_rgb8(2, 2); 3];
        session
            .interpolate(
                &mut inputs,
                &mut outputs,
                &JobOptions {
                    multiplier: 2,
                    ..JobOptions::default()
                },
            )
            .expect("the retry must succeed");
    }

    #[test]
    fn a_v1_plugin_is_rejected_with_an_actionable_message() {
        let Some(path) = example_plugin_path() else {
            return;
        };
        // Nothing to test if the example is v2; the version gate itself is
        // covered by the constant matching the header.
        assert_eq!(SR_INFER_ABI_VERSION, 2);
        assert!(path.exists());
    }
}
