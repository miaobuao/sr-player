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

pub mod abi;
pub mod builtin;

use crate::ffmpeg::Ffmpeg;
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
}

impl Capabilities {
    pub fn supports(&self, task: InferenceTask) -> bool {
        match task {
            InferenceTask::Scale => self.scale,
            InferenceTask::Interpolate => self.interpolate,
            InferenceTask::Restore => self.restore,
        }
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

pub trait InferenceEngine: Send + Sync {
    fn id(&self) -> &str;
    fn display_name(&self) -> String;
    fn kind(&self) -> EngineKind;
    fn capabilities(&self) -> Capabilities;
    fn status(&self) -> EngineStatus;
    /// Honest description of what this engine does *not* do.
    fn quality_note(&self) -> String;
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
                Ok(caps) => engines.push(Arc::new(builtin::PluginEngine::new(plugin, caps))),
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

    pub fn has_plugin(&self) -> bool {
        self.engines
            .iter()
            .any(|e| e.kind() == EngineKind::Plugin)
    }

    /// Rows for the UI's engine table.
    pub fn report_rows(&self) -> Vec<(String, String)> {
        self.engines
            .iter()
            .map(|engine| {
                let status = match engine.status() {
                    EngineStatus::Ready => "ready".to_string(),
                    EngineStatus::Unavailable(reason) => format!("unavailable: {reason}"),
                };
                let caps = engine.capabilities();
                let tasks: Vec<&str> = [
                    (InferenceTask::Scale, caps.scale),
                    (InferenceTask::Interpolate, caps.interpolate),
                    (InferenceTask::Restore, caps.restore),
                ]
                .iter()
                .filter(|(_, enabled)| *enabled)
                .map(|(task, _)| task.as_str())
                .collect();
                let mut value = format!(
                    "{status} · {} · [{}]",
                    engine.display_name(),
                    tasks.join(", ")
                );
                if !caps.backends.is_empty() {
                    value.push_str(&format!(" · {}", caps.backends.join("/")));
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
        let plugin = self.has_plugin();
        format!(
            "{ready}/{} inference engines ready{}",
            self.engines.len(),
            if plugin {
                " (model plugin present)"
            } else {
                " (no model plugin: deterministic FFmpeg path)"
            }
        )
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
            max_pixels: 0,
            backends: vec!["ffmpeg".into()],
            precision: vec!["8bit".into()],
            vendor: None,
            notes: vec![],
        };
        assert!(caps.supports(InferenceTask::Scale));
        assert!(!caps.supports(InferenceTask::Interpolate));
        assert!(!caps.supports(InferenceTask::Restore));
    }
}
