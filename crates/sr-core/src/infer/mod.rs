//! Vendor-neutral inference layer.
//!
//! Three executors exist, in order of quality:
//!
//! | engine | interpolation | restoration | requirement |
//! |---|---|---|---|
//! | `plugin` | model-based (RIFE-class) | model-based (SeedVR2-class) | a shared library implementing [`abi`] |
//! | `ffmpeg-minterpolate` | motion compensated | none | FFmpeg with `minterpolate` |
//! | `ffmpeg-baseline` | frame duplication | none (deterministic resample) | FFmpeg |
//!
//! The baseline always exists, so the pipeline always has a correct answer; the
//! better engines are *selected* when available and never *required*. That is
//! what makes "no CUDA, no Python" a property of the product instead of a hope.
//!
//! An engine that reports a capability must also be able to *execute* it:
//! [`InferenceEngine::open_session`] hands back an [`EngineSession`] that the
//! native executor drives frame by frame. An engine whose work happens inside an
//! FFmpeg filter graph has no session, and says so, which is why the runner can
//! tell the difference between "a model ran" and "FFmpeg did something that
//! looks similar".

pub mod abi;
pub mod builtin;

use crate::error::{Error, Result};
use crate::ffmpeg::Ffmpeg;
use abi::{ExecOutcome, FrameBuffer, JobOptions, PluginCapabilities, SessionRequest};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceTask {
    Scale,
    Interpolate,
    Restore,
}

impl InferenceTask {
    pub fn as_str(self) -> &'static str {
        match self {
            InferenceTask::Scale => "scale",
            InferenceTask::Interpolate => "interpolate",
            InferenceTask::Restore => "restore",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capabilities {
    pub scale: bool,
    pub interpolate: bool,
    pub restore: bool,
    /// 0 means "no stated limit".
    pub max_pixels: u64,
    pub backends: Vec<String>,
    pub precision: Vec<String>,
    pub vendor: Option<String>,
    pub notes: Vec<String>,
    /// Frames per `RESTORE` call. 1 for engines with no temporal model.
    pub max_batch: u32,
    /// Frames the interpolator wants at once: 2 for a classic RIFE-class model,
    /// 5+ for a temporal restoration model that needs 4n+1 context.
    pub temporal_window: u32,
    /// How many *new* frames the engine can place between two input frames.
    pub max_multiplier: u32,
    /// Geometry change applied to restored frames (1.0 = same size).
    pub upscale: f64,
    /// True when the engine can be told to use a smaller working set after an
    /// out-of-memory failure instead of failing the chunk.
    pub can_reconfigure: bool,
    pub model: String,
    /// Devices the engine can run on, best first, for the UI.
    pub devices: Vec<String>,
}

impl Default for Capabilities {
    fn default() -> Self {
        Capabilities {
            scale: false,
            interpolate: false,
            restore: false,
            max_pixels: 0,
            backends: Vec::new(),
            precision: Vec::new(),
            vendor: None,
            notes: Vec::new(),
            max_batch: 1,
            temporal_window: 2,
            max_multiplier: 1,
            upscale: 1.0,
            can_reconfigure: false,
            model: String::new(),
            devices: Vec::new(),
        }
    }
}

impl Capabilities {
    pub fn supports(&self, task: InferenceTask) -> bool {
        match task {
            InferenceTask::Scale => self.scale,
            InferenceTask::Interpolate => self.interpolate,
            InferenceTask::Restore => self.restore,
        }
    }

    /// The tasks this engine can really run, for the UI's engine table.
    pub fn task_names(&self) -> Vec<&'static str> {
        [
            (InferenceTask::Restore, self.restore),
            (InferenceTask::Interpolate, self.interpolate),
            (InferenceTask::Scale, self.scale),
        ]
        .iter()
        .filter(|(_, on)| *on)
        .map(|(task, _)| task.as_str())
        .collect()
    }
}

#[derive(Clone, Debug)]
pub enum EngineStatus {
    Ready,
    Unavailable(String),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    /// Ships with the engine: FFmpeg only.
    Baseline,
    /// Better than nothing, still FFmpeg: motion-compensated interpolation.
    Enhanced,
    /// Pluggable model runtime.
    Plugin,
}

/// What the pipeline asks a model backend to shrink to after an out-of-memory
/// failure. Expressed in engine terms so this module does not depend on the
/// pipeline's policy types.
#[derive(Clone, Debug, Default)]
pub struct WorkingSetRequest {
    /// Frames per call.
    pub batch: u32,
    /// Tile edge in pixels; 0 = untiled.
    pub tile: u32,
    /// Where the weights live: `"none"`, `"cpu"`, `"disk"`.
    pub offload: String,
    /// Transformer blocks kept on the host.
    pub block_swap: u32,
    /// `"fp32"`, `"fp16"`, `"bf16"`, `"fp8"`.
    pub precision: String,
}

/// A live model session: the thing frames are actually pushed through.
///
/// Sessions are not `Send` by design — a backend session belongs to the thread
/// that opened it, which keeps device queues and fences simple.
pub trait EngineSession {
    fn engine_id(&self) -> &str;

    /// Places `multiplier - 1` new frames between each consecutive pair.
    ///
    /// The executor guarantees that every frame in `inputs` comes from the same
    /// shot: a model can never be asked to bridge a cut, because it is never
    /// given the pair.
    fn interpolate(
        &mut self,
        inputs: &mut [FrameBuffer],
        outputs: &mut [FrameBuffer],
        options: &JobOptions,
    ) -> Result<ExecOutcome>;

    /// Restores a temporal batch in place: same frames in, same frames out.
    fn restore(
        &mut self,
        frames: &mut [FrameBuffer],
        options: &JobOptions,
    ) -> Result<ExecOutcome>;

    /// Applies a smaller working set after an out-of-memory failure.
    ///
    /// Returning `Err(Unsupported)` is a legitimate answer and means "retrying
    /// with a different budget will not help" — the executor then reports that
    /// honestly instead of pretending the ladder worked.
    fn reconfigure(&mut self, _request: &WorkingSetRequest) -> Result<()> {
        Err(Error::Unsupported(format!(
            "{} does not accept a working-set change",
            self.engine_id()
        )))
    }
}

pub trait InferenceEngine: Send + Sync {
    fn id(&self) -> &str;
    fn display_name(&self) -> String;
    fn kind(&self) -> EngineKind;
    fn capabilities(&self) -> Capabilities;
    fn status(&self) -> EngineStatus;
    /// Honest description of what this engine does *not* do.
    fn quality_note(&self) -> String;

    /// Opens a model session for native frame-by-frame execution.
    ///
    /// An engine whose work lives inside an FFmpeg filter graph cannot do this;
    /// the default implementation says so rather than silently returning
    /// something that would run a different algorithm.
    fn open_session(&self, _request: &SessionRequest) -> Result<Box<dyn EngineSession>> {
        Err(Error::Unsupported(format!(
            "{} runs inside FFmpeg and cannot execute frames natively",
            self.id()
        )))
    }
}

/// Every engine this machine can offer, probed once at startup.
pub struct EngineRegistry {
    engines: Vec<Arc<dyn InferenceEngine>>,
}

impl EngineRegistry {
    /// Builds a registry from an explicit engine list. Used by tests and by
    /// embedders that know exactly which backend they want.
    pub fn from_engines(engines: Vec<Arc<dyn InferenceEngine>>) -> Self {
        EngineRegistry { engines }
    }

    pub fn probe(ff: Arc<Ffmpeg>) -> Self {
        let mut engines: Vec<Arc<dyn InferenceEngine>> = Vec::new();
        if let Some(plugin) = abi::discover() {
            match plugin.capabilities() {
                Ok(caps) => {
                    let devices = plugin.devices().unwrap_or_default();
                    for device in &devices {
                        tracing::info!("inference device {}", device.describe());
                    }
                    engines.push(Arc::new(builtin::PluginEngine::new(plugin, caps, devices)));
                }
                Err(e) => tracing::warn!("inference plugin rejected: {e}"),
            }
        }
        if ff.has_filter("minterpolate") {
            engines.push(Arc::new(builtin::MinterpolateEngine::new(Arc::clone(&ff))));
        }
        engines.push(Arc::new(builtin::FfmpegEngine::new(ff)));
        EngineRegistry { engines }
    }

    pub fn engines(&self) -> &[Arc<dyn InferenceEngine>] {
        &self.engines
    }

    /// Best ready engine for a task: plugin > enhanced > baseline.
    pub fn select(&self, task: InferenceTask) -> Option<Arc<dyn InferenceEngine>> {
        let mut best: Option<(u8, Arc<dyn InferenceEngine>)> = None;
        for engine in &self.engines {
            if !matches!(engine.status(), EngineStatus::Ready) {
                continue;
            }
            if !engine.capabilities().supports(task) {
                continue;
            }
            let rank = match engine.kind() {
                EngineKind::Plugin => 3,
                EngineKind::Enhanced => 2,
                EngineKind::Baseline => 1,
            };
            if best.as_ref().map(|(r, _)| rank > *r).unwrap_or(true) {
                best = Some((rank, Arc::clone(engine)));
            }
        }
        best.map(|(_, engine)| engine)
    }

    /// The model engine, if one is installed. Only these can run natively.
    pub fn model_engine(&self) -> Option<Arc<dyn InferenceEngine>> {
        self.engines
            .iter()
            .find(|e| e.kind() == EngineKind::Plugin)
            .cloned()
    }

    pub fn has_plugin(&self) -> bool {
        self.engines
            .iter()
            .any(|e| e.kind() == EngineKind::Plugin)
    }

    /// Rows for the UI's engine table.
    ///
    /// `id → value`, where the value states what the engine can do *and* whether
    /// it can execute it natively, because "loaded" and "usable" are different
    /// facts and the UI must not conflate them.
    pub fn report_rows(&self) -> Vec<(String, String)> {
        self.engines
            .iter()
            .map(|engine| {
                let status = match engine.status() {
                    EngineStatus::Ready => "ready".to_string(),
                    EngineStatus::Unavailable(reason) => format!("unavailable: {reason}"),
                };
                let caps = engine.capabilities();
                let mut value = format!(
                    "{status} · {} · [{}]",
                    engine.display_name(),
                    caps.task_names().join(", ")
                );
                if !caps.backends.is_empty() {
                    value.push_str(&format!(" · {}", caps.backends.join("/")));
                }
                if !caps.model.is_empty() {
                    value.push_str(&format!(" · {}", caps.model));
                }
                if engine.kind() == EngineKind::Plugin {
                    value.push_str(&format!(
                        " · native executor, window {}, batch {}",
                        caps.temporal_window, caps.max_batch
                    ));
                } else {
                    value.push_str(" · folded into the FFmpeg pass");
                }
                if !caps.devices.is_empty() {
                    value.push_str(&format!(" · {}", caps.devices.join(", ")));
                }
                let note = engine.quality_note();
                if !note.is_empty() {
                    value.push_str(&format!(" · {note}"));
                }
                (engine.id().to_string(), value)
            })
            .collect()
    }

    pub fn summary(&self) -> String {
        let ready = self
            .engines
            .iter()
            .filter(|e| matches!(e.status(), EngineStatus::Ready))
            .count();
        match self.model_engine() {
            Some(engine) => {
                let caps = engine.capabilities();
                format!(
                    "{ready}/{} inference engines ready (model backend: {} {})",
                    self.engines.len(),
                    if caps.model.is_empty() {
                        engine.id()
                    } else {
                        &caps.model
                    },
                    caps.precision.join("/")
                )
            }
            None => format!(
                "{ready}/{} inference engines ready (no model plugin: deterministic FFmpeg path)",
                self.engines.len()
            ),
        }
    }
}

/// Converts a raw plugin capability report into the engine's own shape.
impl From<&PluginCapabilities> for Capabilities {
    fn from(raw: &PluginCapabilities) -> Self {
        Capabilities {
            scale: raw.supports_op(abi::SR_OP_SCALE),
            interpolate: raw.supports_op(abi::SR_OP_INTERPOLATE),
            restore: raw.supports_op(abi::SR_OP_RESTORE),
            max_pixels: raw.max_pixels,
            backends: if raw.backend.is_empty() {
                Vec::new()
            } else {
                vec![raw.backend.clone()]
            },
            precision: if raw.precision.is_empty() {
                Vec::new()
            } else {
                vec![raw.precision.clone()]
            },
            vendor: raw.vendor.clone(),
            notes: vec![format!("ABI {}", raw.abi_version)],
            max_batch: raw.max_batch,
            temporal_window: raw.min_temporal_window.max(2),
            max_multiplier: 4,
            upscale: raw.upscale(),
            can_reconfigure: raw.has(abi::SR_CAP_TILING)
                || raw.supports_op(abi::SR_OP_RESTORE),
            model: if raw.model.is_empty() {
                String::new()
            } else {
                match &raw.model_version {
                    Some(version) => format!("{} {version}", raw.model),
                    None => raw.model.clone(),
                }
            },
            devices: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_names_are_stable() {
        assert_eq!(InferenceTask::Scale.as_str(), "scale");
        assert_eq!(InferenceTask::Restore.as_str(), "restore");
    }

    #[test]
    fn capabilities_supports_matches_flags() {
        let caps = Capabilities {
            scale: true,
            interpolate: false,
            restore: false,
            backends: vec!["ffmpeg".into()],
            ..Capabilities::default()
        };
        assert!(caps.supports(InferenceTask::Scale));
        assert!(!caps.supports(InferenceTask::Interpolate));
        assert!(!caps.supports(InferenceTask::Restore));
        assert_eq!(caps.task_names(), vec!["scale"]);
    }

    #[test]
    fn a_plugin_that_cannot_open_a_session_says_so_instead_of_improvising() {
        // The engine trait's default is the guarantee: an engine without a real
        // execution path must fail loudly rather than run something else.
        struct FilterOnly;
        impl InferenceEngine for FilterOnly {
            fn id(&self) -> &str {
                "filter-only"
            }
            fn display_name(&self) -> String {
                "filter only".into()
            }
            fn kind(&self) -> EngineKind {
                EngineKind::Enhanced
            }
            fn capabilities(&self) -> Capabilities {
                Capabilities {
                    interpolate: true,
                    ..Capabilities::default()
                }
            }
            fn status(&self) -> EngineStatus {
                EngineStatus::Ready
            }
            fn quality_note(&self) -> String {
                String::new()
            }
        }
        let err = match FilterOnly.open_session(&SessionRequest::default()) {
            Ok(_) => panic!("an engine with no native path must not hand back a session"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::Unsupported(_)));
        assert!(err.to_string().contains("cannot execute frames natively"));
    }
}
