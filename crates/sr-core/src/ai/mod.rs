//! The one AI runtime.
//!
//! Everything here is a thin, safe face over `native/sr-native`, which is a C++
//! static library linked into this binary. There is no second implementation, no
//! dynamic loading, no discovery and no feature flag: if the runtime cannot run a
//! model, the honest answer is that the work did not happen, and the callers in
//! [`crate::pipeline`] refuse the job rather than substituting something else.
//!
//! The ABI is deliberately four functions and two constructors — a restorer and an
//! interpolator, each opened against a device and a model directory. There is no
//! tensor, graph, operator or capability negotiation, because an API general
//! enough to describe any network is also general enough to let a caller believe
//! one ran when a filter did.
//!
//! Two model tasks exist and no more:
//!
//! | task | model | call |
//! |---|---|---|
//! | restoration | Real-ESRGAN x4plus | [`Restorer::restore`] |
//! | interpolation | RIFE 4.25, `ensemble = false` | [`Rife::interpolate`] |

pub mod ffi;

use crate::error::Error;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Arc;

/// A failure from the runtime, kept separate from [`crate::Error`] so the tile
/// ladder can tell "the device refused an allocation" from "the model is not
/// there" without matching on strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AiError {
    InvalidArgument(String),
    NoDevice,
    ModelMissing(String),
    OutOfMemory,
    DeviceLost,
    Internal(String),
}

impl std::fmt::Display for AiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AiError::InvalidArgument(m) => write!(f, "the AI runtime rejected an argument: {m}"),
            AiError::NoDevice => write!(f, "no Vulkan device is available"),
            AiError::ModelMissing(m) => write!(f, "{m}"),
            AiError::OutOfMemory => write!(
                f,
                "the device refused an allocation; a smaller tile may fit"
            ),
            AiError::DeviceLost => write!(
                f,
                "the Vulkan device was lost; this context cannot be reused"
            ),
            AiError::Internal(m) => write!(f, "the AI runtime failed: {m}"),
        }
    }
}

impl From<AiError> for Error {
    fn from(value: AiError) -> Self {
        // Both are "the input cannot be handled", which is what `Unsupported`
        // means to the stage runner: report it and do not retry.
        Error::Unsupported(value.to_string())
    }
}

impl AiError {
    pub fn is_out_of_memory(&self) -> bool {
        matches!(self, AiError::OutOfMemory)
    }

    fn from_code(code: i32, ctx: *mut ffi::sr_context, what: &str) -> Self {
        let detail = last_error(ctx);
        match code {
            ffi::SR_ERR_INVALID_ARGUMENT => AiError::InvalidArgument(if detail.is_empty() {
                what.to_string()
            } else {
                detail
            }),
            ffi::SR_ERR_NO_DEVICE => AiError::NoDevice,
            ffi::SR_ERR_MODEL_MISSING => AiError::ModelMissing(if detail.is_empty() {
                what.to_string()
            } else {
                detail
            }),
            ffi::SR_ERR_OUT_OF_MEMORY => AiError::OutOfMemory,
            ffi::SR_ERR_DEVICE_LOST => AiError::DeviceLost,
            _ => AiError::Internal(if detail.is_empty() {
                what.to_string()
            } else {
                detail
            }),
        }
    }
}

type AiResult<T> = std::result::Result<T, AiError>;

/// The runtime's own message for its last failure on this context.
fn last_error(ctx: *mut ffi::sr_context) -> String {
    if ctx.is_null() {
        return String::new();
    }
    // Safety: a live context; the pointer is valid until the next call on it,
    // which is not made here before the string is copied.
    let raw = unsafe { ffi::sr_last_error(ctx) };
    if raw.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

impl DeviceType {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceType::Discrete => "discrete",
            DeviceType::Integrated => "integrated",
            DeviceType::Virtual => "virtual",
            DeviceType::Cpu => "cpu",
            DeviceType::Other => "other",
        }
    }

    fn from_code(code: i32) -> Self {
        match code {
            ffi::SR_DEVICE_DISCRETE => DeviceType::Discrete,
            ffi::SR_DEVICE_INTEGRATED => DeviceType::Integrated,
            ffi::SR_DEVICE_VIRTUAL => DeviceType::Virtual,
            ffi::SR_DEVICE_CPU => DeviceType::Cpu,
            _ => DeviceType::Other,
        }
    }
}

/// A device as the AI runtime sees it.
///
/// This is the authoritative list for anything that will host a model: the
/// runtime is what allocates, and its device numbering need not agree with the
/// Vulkan probe used for telemetry.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
    pub driver: String,
    pub api_version: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: DeviceType,
    /// What the driver will let *this process* allocate. 0 when it will not say.
    pub budget_mib: u64,
    pub unified_memory: bool,
}

impl DeviceInfo {
    /// Whether a model may be placed here.
    ///
    /// An integrated device's "device-local" heap is system RAM shared with
    /// everything else, so a free figure for it does not mean a 12 MB network
    /// plus its activations will fit after a desktop and a browser have taken
    /// their share. It is reported, and it is not used.
    pub fn is_model_capable(&self) -> bool {
        !self.unified_memory && self.device_type == DeviceType::Discrete && self.budget_mib > 0
    }
}

fn decode_fixed(bytes: &[std::os::raw::c_char]) -> String {
    let raw: Vec<u8> = bytes
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8_lossy(&raw).into_owned()
}

/// Every Vulkan device the runtime can see. Empty means it cannot run at all.
pub fn devices() -> Vec<DeviceInfo> {
    let count = unsafe { ffi::sr_device_count() };
    let mut out = Vec::new();
    for index in 0..count {
        let mut info = ffi::sr_device_info_t::default();
        if unsafe { ffi::sr_device_info(index, &mut info) } != ffi::SR_OK {
            continue;
        }
        out.push(DeviceInfo {
            index: info.index,
            name: decode_fixed(&info.name),
            driver: decode_fixed(&info.driver_version),
            api_version: decode_fixed(&info.api_version),
            vendor_id: info.vendor_id,
            device_id: info.device_id,
            device_type: DeviceType::from_code(info.device_type),
            budget_mib: info.budget_mib,
            unified_memory: info.unified_memory != 0,
        });
    }
    out
}

/// The device a model should be placed on: the first discrete one the runtime
/// reports a budget for, or `None` when there is no such device.
///
/// Chosen here rather than by the caller so the choice is one decision in one
/// place. Nothing in this crate branches on the vendor.
pub fn preferred_device() -> Option<DeviceInfo> {
    devices().into_iter().find(DeviceInfo::is_model_capable)
}

/// A device and the Vulkan instance the models on it share.
///
/// One context owns the device. A [`Rife`] and a [`Restorer`] may be opened
/// against the same context, but they are driven one at a time — two networks
/// plus their activation buffers do not fit alongside a desktop on a 16 GB card.
pub struct Runtime {
    ctx: *mut ffi::sr_context,
    device: DeviceInfo,
}

// Safety: sr-native documents that a context is owned by one thread at a time.
// Moving it between threads is fine; sharing it is not, which is why `Sync` is
// deliberately not implemented.
unsafe impl Send for Runtime {}

impl Runtime {
    pub fn open(device_index: i32) -> AiResult<Runtime> {
        let device = devices()
            .into_iter()
            .find(|d| d.index == device_index)
            .ok_or(AiError::NoDevice)?;
        let ctx = unsafe { ffi::sr_context_create(device_index) };
        if ctx.is_null() {
            return Err(AiError::NoDevice);
        }
        Ok(Runtime { ctx, device })
    }

    /// Opens the first device a model may be placed on.
    pub fn open_preferred() -> AiResult<Runtime> {
        let device = preferred_device().ok_or(AiError::NoDevice)?;
        Runtime::open(device.index)
    }

    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    pub fn open_rife(self: &Arc<Self>, model_dir: &Path) -> AiResult<Rife> {
        let dir = CString::new(model_dir.to_string_lossy().as_bytes())
            .map_err(|_| AiError::InvalidArgument("the model path contains a NUL".into()))?;
        let handle = unsafe { ffi::sr_rife_create(self.ctx, dir.as_ptr()) };
        if handle.is_null() {
            return Err(AiError::ModelMissing(format!(
                "RIFE could not be loaded from {}: {}",
                model_dir.display(),
                last_error(self.ctx)
            )));
        }
        Ok(Rife {
            handle,
            _runtime: Arc::clone(self),
        })
    }

    pub fn open_restorer(self: &Arc<Self>, model_dir: &Path) -> AiResult<Restorer> {
        let dir = CString::new(model_dir.to_string_lossy().as_bytes())
            .map_err(|_| AiError::InvalidArgument("the model path contains a NUL".into()))?;
        let handle = unsafe { ffi::sr_restorer_create(self.ctx, dir.as_ptr()) };
        if handle.is_null() {
            return Err(AiError::ModelMissing(format!(
                "the restoration model could not be loaded from {}: {}",
                model_dir.display(),
                last_error(self.ctx)
            )));
        }
        Ok(Restorer {
            handle,
            _runtime: Arc::clone(self),
        })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe { ffi::sr_context_destroy(self.ctx) };
    }
}

/// A frame to hand to, or receive from, the runtime.
///
/// Interleaved 8-bit RGB with an explicit stride, so a caller can pass a
/// row-padded buffer without copying it.
#[derive(Debug)]
pub struct FrameView<'a> {
    pub data: &'a mut [u8],
    pub width: i32,
    pub height: i32,
    pub stride: i32,
}

impl<'a> FrameView<'a> {
    pub fn new(data: &'a mut [u8], width: i32, height: i32) -> Self {
        FrameView {
            data,
            width,
            height,
            stride: width * 3,
        }
    }

    fn as_image(&mut self) -> ffi::sr_image {
        ffi::sr_image {
            data: self.data.as_mut_ptr(),
            width: self.width,
            height: self.height,
            stride: self.stride,
            channels: 3,
        }
    }

    fn as_const_image(&self) -> ffi::sr_image {
        ffi::sr_image {
            data: self.data.as_ptr() as *mut u8,
            width: self.width,
            height: self.height,
            stride: self.stride,
            channels: 3,
        }
    }
}

/// RIFE 4.25, ensemble off.
///
/// The interface is a *pair* and a timestep, and that is the point: the network
/// cannot be handed a window, so it cannot be handed a pair of frames from
/// different shots. The cut-safety guarantee is a property of the call shape
/// rather than a filter setting a future edit could drop.
pub struct Rife {
    handle: *mut ffi::sr_rife,
    /// Owns the context, not merely a copy of its pointer.
    ///
    /// A bare pointer here would be a dangling handle the moment the caller's
    /// `Runtime` went out of scope, and the failure would be a use-after-free
    /// inside ncnn rather than a compile error.
    _runtime: Arc<Runtime>,
}

// Safety: see `Runtime`. The handle is used by one thread at a time.
unsafe impl Send for Rife {}

impl Rife {
    /// Synthesises the frame `timestep` of the way from `previous` to `next`.
    ///
    /// `timestep` is in (0, 1) exclusive — the endpoints are not synthesised,
    /// because a caller that wants frame 0 already has frame 0.
    ///
    /// `previous` and `next` must come from the same shot. This call cannot check
    /// that, and two frames from different shots will produce a confident, wrong,
    /// blended frame; keeping that true is [`crate::pipeline::segments`]'s job.
    pub fn interpolate(
        &mut self,
        previous: &[u8],
        next: &[u8],
        timestep: f32,
        out: &mut FrameView<'_>,
    ) -> AiResult<()> {
        let width = out.width;
        let height = out.height;
        let need = width as usize * height as usize * 3;

        if previous.len() < need || next.len() < need {
            return Err(AiError::InvalidArgument(format!(
                "the pair is {} and {} bytes but a {width}x{height} frame needs {need}",
                previous.len(),
                next.len()
            )));
        }

        // The C ABI takes non-const data pointers for both inputs, but the
        // implementation only ever reads them. Building the views in place rather
        // than copying the frames keeps this a borrow at the boundary.
        let prev_image = ffi::sr_image {
            data: previous.as_ptr() as *mut u8,
            width,
            height,
            stride: width * 3,
            channels: 3,
        };
        let next_image = ffi::sr_image {
            data: next.as_ptr() as *mut u8,
            width,
            height,
            stride: width * 3,
            channels: 3,
        };
        let mut out_image = out.as_image();

        // Safety: all three views are live for the duration of the call, the
        // dimensions match, and the runtime writes only through `out_image`.
        let code = unsafe {
            ffi::sr_rife_process(
                self.handle,
                &prev_image,
                &next_image,
                timestep,
                &mut out_image,
            )
        };
        if code == ffi::SR_OK {
            Ok(())
        } else {
            Err(AiError::from_code(code, self._runtime.ctx, "sr_rife_process"))
        }
    }
}

impl Drop for Rife {
    fn drop(&mut self) {
        unsafe { ffi::sr_rife_destroy(self.handle) };
    }
}

/// Real-ESRGAN x4plus.
///
/// Model-agnostic at the product level: callers ask for "the restorer" and never
/// name a network, so a future proven ncnn-native restoration model can replace
/// this one without touching `sr-core` or reintroducing a framework.
pub struct Restorer {
    handle: *mut ffi::sr_restorer,
    /// See `Rife`: the model owns the context it runs on.
    _runtime: Arc<Runtime>,
}

unsafe impl Send for Restorer {}

impl Restorer {
    /// The size the output must be for a given input.
    pub fn output_size(&self, width: i32, height: i32, scale: i32) -> AiResult<(i32, i32)> {
        let mut out_w = 0i32;
        let mut out_h = 0i32;
        let code = unsafe {
            ffi::sr_restorer_output_size(self.handle, width, height, scale, &mut out_w, &mut out_h)
        };
        if code == ffi::SR_OK {
            Ok((out_w, out_h))
        } else {
            Err(AiError::InvalidArgument(format!(
                "the model does not offer scale {scale}"
            )))
        }
    }

    /// Restores one frame.
    ///
    /// `tile` is the working-set knob: 0 lets the runtime choose, otherwise it is
    /// the tile edge in pixels. [`AiError::OutOfMemory`] is the signal to retry
    /// the same frame with a smaller tile; no other failure is.
    pub fn restore(
        &mut self,
        input: &mut FrameView<'_>,
        output: &mut FrameView<'_>,
        scale: i32,
        tile: i32,
    ) -> AiResult<()> {
        let in_image = input.as_const_image();
        let mut out_image = output.as_image();
        let code =
            unsafe { ffi::sr_restorer_process(self.handle, &in_image, &mut out_image, scale, tile) };
        if code == ffi::SR_OK {
            Ok(())
        } else {
            Err(AiError::from_code(code, self._runtime.ctx, "sr_restorer_process"))
        }
    }
}

impl Drop for Restorer {
    fn drop(&mut self) {
        unsafe { ffi::sr_restorer_destroy(self.handle) };
    }
}

/// The models directory.
///
/// Resolved in three steps, because "next to the executable" is right for a
/// packaged build and wrong for a `cargo test`, whose working directory is the
/// crate root rather than the workspace root:
///
/// 1. `SR_MODELS_DIR`, which is what a deployment and the tests both set;
/// 2. `models/` beside the executable;
/// 3. the workspace's own `models/`, baked in at compile time.
///
/// The order matters: an installed build must never pick up a developer's
/// workspace models by accident.
pub fn models_root() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("SR_MODELS_DIR") {
        if !dir.is_empty() {
            return std::path::PathBuf::from(dir);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join("models");
            if beside.is_dir() {
                return beside;
            }
        }
    }

    // crates/sr-core -> the workspace root.
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
}

pub fn rife_model_dir() -> std::path::PathBuf {
    models_root().join("rife-4.25")
}

pub fn restore_model_dir() -> std::path::PathBuf {
    models_root().join("realesrgan-x4plus")
}

/// Whether both pinned models are present on disk.
///
/// This is a *file* check, not a capability check: the runtime is always linked,
/// and what can be absent is the weights.
pub fn models_installed() -> bool {
    rife_model_dir().join("flownet.bin").exists() && restore_model_dir().join("model.bin").exists()
}

/// A one-line status for the log header and the UI, from the runtime's own view.
pub fn status_line() -> String {
    let devices = devices();
    if devices.is_empty() {
        return "AI runtime: no Vulkan device; restoration and interpolation are refused".into();
    }
    let preferred = preferred_device();
    let target = match &preferred {
        Some(d) => format!("{} (device {}, {:.1} GiB)", d.name, d.index, d.budget_mib as f64 / 1024.0),
        None => "none eligible (every device shares memory with the system)".to_string(),
    };
    let models = if models_installed() {
        "RIFE 4.25 + Real-ESRGAN x4plus"
    } else {
        "no model weights installed"
    };
    format!(
        "AI runtime: ncnn/Vulkan, {} device(s), models on {target}; {models}",
        devices.len()
    )
}

/// Shares one runtime between the stages that need it without opening the device
/// twice.
pub type SharedRuntime = Arc<Runtime>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_runtime_reports_at_least_one_device_on_the_reference_machine() {
        // This is the only test here that depends on the host. It is written to
        // report rather than to pass vacuously: a machine with no Vulkan device
        // says so, and that is the honest outcome rather than a skip.
        let devices = devices();
        for device in &devices {
            println!(
                "device {}: {} [{}] {} MiB{}",
                device.index,
                device.name,
                device.device_type.as_str(),
                device.budget_mib,
                if device.unified_memory { " (unified)" } else { "" }
            );
        }
        if devices.is_empty() {
            eprintln!("no Vulkan device: the AI runtime cannot run on this machine");
        }
    }

    #[test]
    fn an_out_of_range_device_index_is_refused() {
        assert!(Runtime::open(9_999).is_err());
        assert!(Runtime::open(-1).is_err());
    }

    #[test]
    fn an_integrated_device_is_never_chosen_to_host_a_model() {
        assert!(!DeviceInfo {
            index: 1,
            name: "integrated".into(),
            driver: String::new(),
            api_version: String::new(),
            vendor_id: 0,
            device_id: 0,
            device_type: DeviceType::Integrated,
            budget_mib: 30_000,
            unified_memory: true,
        }
        .is_model_capable());

        assert!(DeviceInfo {
            index: 0,
            name: "discrete".into(),
            driver: String::new(),
            api_version: String::new(),
            vendor_id: 0,
            device_id: 0,
            device_type: DeviceType::Discrete,
            budget_mib: 15_227,
            unified_memory: false,
        }
        .is_model_capable());
    }

    #[test]
    fn only_an_allocation_failure_is_treated_as_capacity() {
        assert!(AiError::OutOfMemory.is_out_of_memory());
        assert!(!AiError::DeviceLost.is_out_of_memory());
        assert!(!AiError::NoDevice.is_out_of_memory());
        assert!(!AiError::Internal("x".into()).is_out_of_memory());
    }
}
