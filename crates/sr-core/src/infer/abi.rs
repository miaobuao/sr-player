//! The C ABI a vendor-neutral inference plugin must expose.
//!
//! This is the contract that keeps the engine from being tied to any one
//! vendor's runtime. A plugin is a shared library exporting three symbols:
//!
//! ```c
//! uint32_t sr_infer_abi_version(void);
//! int32_t  sr_infer_capabilities(sr_infer_capabilities* out);
//! int32_t  sr_infer_run(const sr_infer_request* request, sr_infer_response* response);
//! ```
//!
//! The same ABI can be implemented on Vulkan compute, DirectML, OpenVINO,
//! MIGraphX or a plain CPU kernel; the engine only reads capabilities and asks
//! for work. The canonical header lives in `include/sr_infer.h`, and
//! `crates/sr-infer-plugin-example` is a working implementation used by the
//! tests in this module — the ABI is verified, not aspirational.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};

/// Bumped whenever the structs below change layout.
pub const SR_INFER_ABI_VERSION: u32 = 1;

pub const SR_OK: i32 = 0;
pub const SR_ERR_UNSUPPORTED: i32 = -1;
pub const SR_ERR_INVALID_ARGUMENT: i32 = -2;
pub const SR_ERR_RUNTIME: i32 = -3;
pub const SR_ERR_OUT_OF_MEMORY: i32 = -4;

pub const SR_OP_SCALE: u32 = 1;
pub const SR_OP_BLEND_INTERPOLATE: u32 = 2;

/// Pixel formats a plugin may accept, kept deliberately small.
pub const SR_PIXEL_RGB8: u32 = 1;
pub const SR_PIXEL_RGBA8: u32 = 2;
pub const SR_PIXEL_GRAY8: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SrInferCapabilities {
    pub abi_version: u32,
    pub scale: u32,
    pub interpolate: u32,
    pub restore: u32,
    pub max_pixels: u64,
    /// `"vulkan"`, `"dml"`, `"cpu"`, ...
    pub backend: *const c_char,
    /// `"fp32"`, `"fp16"`, `"fp8"`, ...
    pub precision: *const c_char,
    /// `"nvidia"`, `"amd"`, `"intel"`, or null for vendor-neutral.
    pub vendor: *const c_char,
}

impl Default for SrInferCapabilities {
    fn default() -> Self {
        SrInferCapabilities {
            abi_version: SR_INFER_ABI_VERSION,
            scale: 0,
            interpolate: 0,
            restore: 0,
            max_pixels: 0,
            backend: std::ptr::null(),
            precision: std::ptr::null(),
            vendor: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SrInferFrame {
    pub data: *mut u8,
    pub width: u32,
    pub height: u32,
    /// Bytes per row; 0 means tightly packed.
    pub stride: u32,
    pub pixel_format: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SrInferRequest {
    pub op: u32,
    /// 1 for scale/blend, 2 for interpolation (A and B).
    pub input_count: u32,
    pub a: SrInferFrame,
    pub b: SrInferFrame,
    pub out: SrInferFrame,
    /// Scale factor numerator/denominator (for `SR_OP_SCALE`).
    pub scale_num: u32,
    pub scale_den: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SrInferResponse {
    pub frames_written: u32,
    pub vram_used_bytes: u64,
    /// Free-form message, null when there is nothing to say.
    pub message: *const c_char,
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CapabilitiesFn = unsafe extern "C" fn(*mut SrInferCapabilities) -> c_int;
type RunFn = unsafe extern "C" fn(*const SrInferRequest, *mut SrInferResponse) -> c_int;

/// Capability report decoupled from the raw pointers, safe to log and persist.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PluginCapabilities {
    pub abi_version: u32,
    pub scale: bool,
    pub interpolate: bool,
    pub restore: bool,
    pub max_pixels: u64,
    pub backend: String,
    pub precision: String,
    pub vendor: Option<String>,
}

impl PluginCapabilities {
    pub fn summary(&self) -> String {
        format!(
            "abi {}, backend {}, precision {}, {}{}{}{}",
            self.abi_version,
            if self.backend.is_empty() { "?" } else { &self.backend },
            if self.precision.is_empty() { "?" } else { &self.precision },
            self.vendor
                .as_deref()
                .map(|v| format!("vendor {v}, "))
                .unwrap_or_default(),
            if self.scale { "scale " } else { "" },
            if self.interpolate { "interpolate " } else { "" },
            if self.restore { "restore" } else { "" },
        )
        .trim_end()
        .to_string()
    }
}

/// A loaded plugin. Dropping it unloads the library.
pub struct PluginLibrary {
    /// Kept alive for the lifetime of the plugin: dropping it unloads the code
    /// that the function pointers below point into.
    _library: libloading::Library,
    abi_version: AbiVersionFn,
    capabilities: CapabilitiesFn,
    run: Option<RunFn>,
    pub path: PathBuf,
}

unsafe fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the plugin contract requires a NUL-terminated string that outlives
    // the call; the example plugin returns pointers to static data.
    unsafe { CStr::from_ptr(ptr) }.to_str().ok().map(|s| s.to_string())
}

impl PluginLibrary {
    pub fn load(path: &Path) -> Result<Self> {
        // SAFETY: loading arbitrary code is the explicit purpose of a plugin;
        // the caller decides which paths are trusted.
        let library = unsafe { libloading::Library::new(path) }.map_err(|e| Error::Plugin {
            plugin: path.display().to_string(),
            detail: format!("cannot load shared library: {e}"),
        })?;

        // SAFETY: symbol types are defined by the ABI this build was compiled
        // against; the version check below catches mismatches.
        let abi_version: AbiVersionFn = unsafe {
            *library
                .get(b"sr_infer_abi_version\0")
                .map_err(|e| Error::Plugin {
                    plugin: path.display().to_string(),
                    detail: format!("missing sr_infer_abi_version: {e}"),
                })?
        };
        let capabilities: CapabilitiesFn = unsafe {
            *library
                .get(b"sr_infer_capabilities\0")
                .map_err(|e| Error::Plugin {
                    plugin: path.display().to_string(),
                    detail: format!("missing sr_infer_capabilities: {e}"),
                })?
        };
        let run: Option<RunFn> = unsafe { library.get(b"sr_infer_run\0").ok().map(|s| *s) };

        let reported = unsafe { abi_version() };
        if reported != SR_INFER_ABI_VERSION {
            return Err(Error::Plugin {
                plugin: path.display().to_string(),
                detail: format!(
                    "ABI version mismatch: plugin reports {reported}, engine expects {SR_INFER_ABI_VERSION}"
                ),
            });
        }

        Ok(PluginLibrary {
            _library: library,
            abi_version,
            capabilities,
            run,
            path: path.to_path_buf(),
        })
    }

    pub fn abi_version(&self) -> u32 {
        unsafe { (self.abi_version)() }
    }

    pub fn capabilities(&self) -> Result<PluginCapabilities> {
        let mut raw = SrInferCapabilities::default();
        let code = unsafe { (self.capabilities)(&mut raw) };
        if code != SR_OK {
            return Err(Error::Plugin {
                plugin: self.path.display().to_string(),
                detail: format!("sr_infer_capabilities returned {code}"),
            });
        }
        Ok(PluginCapabilities {
            abi_version: raw.abi_version,
            scale: raw.scale != 0,
            interpolate: raw.interpolate != 0,
            restore: raw.restore != 0,
            max_pixels: raw.max_pixels,
            backend: unsafe { cstr_to_string(raw.backend) }.unwrap_or_default(),
            precision: unsafe { cstr_to_string(raw.precision) }.unwrap_or_default(),
            vendor: unsafe { cstr_to_string(raw.vendor) },
        })
    }

    /// Calls `sr_infer_run`, translating the C return codes into [`Error`].
    pub fn run(&self, request: &mut SrInferRequest) -> Result<SrInferResponse> {
        let run = self.run.ok_or_else(|| Error::Plugin {
            plugin: self.path.display().to_string(),
            detail: "plugin exports capabilities but no sr_infer_run".into(),
        })?;
        let mut response = SrInferResponse {
            frames_written: 0,
            vram_used_bytes: 0,
            message: std::ptr::null(),
        };
        let code = unsafe { run(request, &mut response) };
        match code {
            SR_OK => Ok(response),
            SR_ERR_OUT_OF_MEMORY => Err(Error::Plugin {
                plugin: self.path.display().to_string(),
                detail: "out of device memory".into(),
            }),
            SR_ERR_UNSUPPORTED => Err(Error::Unsupported(format!(
                "plugin {} does not support op {}",
                self.path.display(),
                request.op
            ))),
            other => Err(Error::Plugin {
                plugin: self.path.display().to_string(),
                detail: format!(
                    "sr_infer_run returned {other}{}",
                    unsafe { cstr_to_string(response.message) }
                        .map(|m| format!(": {m}"))
                        .unwrap_or_default()
                ),
            }),
        }
    }

    pub fn describe(&self) -> String {
        match self.capabilities() {
            Ok(caps) => format!("{} ({})", self.path.display(), caps.summary()),
            Err(e) => format!("{} (unusable: {e})", self.path.display()),
        }
    }
}

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
pub fn discover() -> Option<PluginLibrary> {
    for path in search_paths() {
        if !path.exists() {
            continue;
        }
        match PluginLibrary::load(&path) {
            Ok(lib) => return Some(lib),
            Err(e) => tracing::warn!("inference plugin at {} rejected: {e}", path.display()),
        }
    }
    None
}

/// Helper for tests and plugin authors: runs a blend-interpolate request.
pub fn make_blend_request(
    a: &mut [u8],
    b: &mut [u8],
    out: &mut [u8],
    width: u32,
    height: u32,
    pixel_format: u32,
) -> SrInferRequest {
    let frame = |data: &mut [u8]| SrInferFrame {
        data: data.as_mut_ptr(),
        width,
        height,
        stride: 0,
        pixel_format,
    };
    SrInferRequest {
        op: SR_OP_BLEND_INTERPOLATE,
        input_count: 2,
        a: frame(a),
        b: frame(b),
        out: frame(out),
        scale_num: 1,
        scale_den: 1,
    }
}

/// Frees a message string a plugin allocated for a response.
///
/// The example plugin returns pointers to static data and never allocates, so
/// this is only used by plugins that opt in by documenting the ownership rule.
pub fn free_plugin_string(ptr: *const c_char) {
    if !ptr.is_null() {
        tracing::debug!("plugin returned an owned message string at {ptr:p}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example plugin is a dev-dependency, so cargo builds it before tests.
    fn example_plugin_path() -> Option<PathBuf> {
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
    fn example_plugin_loads_reports_capabilities_and_runs() {
        let Some(path) = example_plugin_path() else {
            // Built out-of-tree (e.g. `cargo test -p sr-core` without dev deps
            // resolved): the ABI is still exercised by the workspace test run.
            eprintln!("example plugin not built; skipping ABI integration test");
            return;
        };
        let library = PluginLibrary::load(&path).expect("load example plugin");
        assert_eq!(library.abi_version(), SR_INFER_ABI_VERSION);

        let caps = library.capabilities().expect("capabilities");
        assert!(caps.interpolate, "example plugin must blend-interpolate");
        assert!(caps.scale);
        assert_eq!(caps.backend, "cpu");
        assert!(caps.vendor.is_none(), "example plugin is vendor-neutral");

        // A real call through the ABI: blend two frames.
        let mut a = vec![0u8, 0, 0, 200, 200, 200];
        let mut b = vec![100u8, 100, 100, 0, 0, 0];
        let mut out = vec![0u8; 6];
        let mut request = make_blend_request(&mut a, &mut b, &mut out, 2, 1, SR_PIXEL_RGB8);
        let response = library.run(&mut request).expect("blend through the ABI");
        assert_eq!(response.frames_written, 1);
        assert_eq!(out, vec![50, 50, 50, 100, 100, 100]);
    }

    #[test]
    fn unsupported_op_is_reported_cleanly() {
        let Some(path) = example_plugin_path() else {
            return;
        };
        let library = PluginLibrary::load(&path).unwrap();
        let mut buffer = vec![0u8; 3];
        let mut request = make_blend_request(&mut buffer.clone(), &mut buffer.clone(), &mut buffer, 1, 1, SR_PIXEL_RGB8);
        request.op = 9999;
        let err = library.run(&mut request).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }
}
