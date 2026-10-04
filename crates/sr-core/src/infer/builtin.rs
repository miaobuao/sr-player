//! The engines that ship with the application: FFmpeg, and an optional plugin.

use super::abi::{
    DeviceInfo, ExecOutcome, FrameBuffer, JobOptions, PluginCapabilities, PluginLibrary, Session,
    SessionRequest,
};
use super::{
    Capabilities, EngineKind, EngineSession, EngineStatus, InferenceEngine, WorkingSetRequest,
};
use crate::error::{Error, Result};
use crate::ffmpeg::Ffmpeg;
use std::sync::Arc;

/// Deterministic FFmpeg path: resampling plus frame duplication.
///
/// This is the floor the product stands on. It requires no model, no vendor SDK
/// and no Python, and it cannot hallucinate — at the cost of not inventing detail
/// that was never in the source.
pub struct FfmpegEngine {
    ff: Arc<Ffmpeg>,
}

impl FfmpegEngine {
    pub fn new(ff: Arc<Ffmpeg>) -> Self {
        FfmpegEngine { ff }
    }
}

impl InferenceEngine for FfmpegEngine {
    fn id(&self) -> &str {
        "ffmpeg-baseline"
    }

    fn display_name(&self) -> String {
        format!("FFmpeg baseline ({})", self.ff.version)
    }

    fn kind(&self) -> EngineKind {
        EngineKind::Baseline
    }

    fn capabilities(&self) -> Capabilities {
        let has_scale = self.ff.has_filter("scale");
        let has_framerate = self.ff.has_filter("framerate");
        let mut notes = vec![
            "scaling is a deterministic Lanczos resample: no invented detail".into(),
            "interpolation duplicates frames: motion is not synthesised".into(),
        ];
        if !has_scale {
            notes.push("this build has no `scale` filter; scaling is unavailable".into());
        }
        if !has_framerate {
            notes.push("this build has no `framerate` filter; retiming is unavailable".into());
        }
        Capabilities {
            scale: has_scale,
            interpolate: has_framerate,
            restore: false,
            max_pixels: 0,
            backends: vec!["ffmpeg".into()],
            precision: vec!["8bit".into(), "10bit".into()],
            vendor: None,
            notes,
            max_batch: 1,
            temporal_window: 2,
            max_multiplier: 16,
            upscale: 1.0,
            can_reconfigure: false,
            model: String::new(),
            devices: Vec::new(),
        }
    }

    fn status(&self) -> EngineStatus {
        if self.ff.has_filter("scale") {
            EngineStatus::Ready
        } else {
            EngineStatus::Unavailable("FFmpeg build lacks the scale filter".into())
        }
    }

    fn quality_note(&self) -> String {
        "no model: cannot invent detail, cannot truly interpolate motion".into()
    }
}

/// FFmpeg's motion-compensated interpolator.
///
/// Real motion interpolation, no model required — but slow, and it decides on
/// its own where the cuts are. That is precisely why it is *not* used to satisfy
/// a request for model interpolation: it cannot be told which frame pairs are
/// legal, so the promise "nothing is ever synthesised across a cut" would be
/// FFmpeg's to keep or break.
pub struct MinterpolateEngine {
    ff: Arc<Ffmpeg>,
}

impl MinterpolateEngine {
    pub fn new(ff: Arc<Ffmpeg>) -> Self {
        MinterpolateEngine { ff }
    }

    /// Ready-to-use filter arguments for a 2x interpolation.
    pub fn filter_args(&self) -> String {
        // `scd=quick` enables scene-change detection inside the filter, and
        // `mi_mode=mci` is the motion-compensated mode.
        "minterpolate=fps=0:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=10"
            .to_string()
    }
}

impl InferenceEngine for MinterpolateEngine {
    fn id(&self) -> &str {
        "ffmpeg-minterpolate"
    }

    fn display_name(&self) -> String {
        "FFmpeg minterpolate (motion compensated)".to_string()
    }

    fn kind(&self) -> EngineKind {
        EngineKind::Enhanced
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            scale: self.ff.has_filter("scale"),
            interpolate: self.ff.has_filter("minterpolate"),
            restore: false,
            max_pixels: 0,
            backends: vec!["ffmpeg".into()],
            precision: vec!["8bit".into(), "10bit".into()],
            vendor: None,
            notes: vec![
                "motion estimation in software: expect a large slowdown on 1080p+".into(),
                "no generative detail, so no hallucination artefacts".into(),
                "cut detection is FFmpeg's, not the engine's, so a cut can be crossed".into(),
            ],
            max_batch: 1,
            temporal_window: 2,
            max_multiplier: 16,
            upscale: 1.0,
            can_reconfigure: false,
            model: String::new(),
            devices: Vec::new(),
        }
    }

    fn status(&self) -> EngineStatus {
        if self.ff.has_filter("minterpolate") {
            EngineStatus::Ready
        } else {
            EngineStatus::Unavailable("FFmpeg build lacks minterpolate".into())
        }
    }

    fn quality_note(&self) -> String {
        "slower than a model, safer than a model".into()
    }
}

/// A loaded inference plugin: the only path that can restore or truly
/// interpolate with a model.
pub struct PluginEngine {
    library: Arc<PluginLibrary>,
    capabilities: PluginCapabilities,
    devices: Vec<DeviceInfo>,
}

impl PluginEngine {
    pub fn new(
        library: Arc<PluginLibrary>,
        capabilities: PluginCapabilities,
        devices: Vec<DeviceInfo>,
    ) -> Self {
        PluginEngine {
            library,
            capabilities,
            devices,
        }
    }

    pub fn library(&self) -> &Arc<PluginLibrary> {
        &self.library
    }

    pub fn capabilities_raw(&self) -> &PluginCapabilities {
        &self.capabilities
    }

    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Opens a session on the best device the plugin offers.
    ///
    /// This is the call that turns "the plugin is loaded" into "the model is
    /// resident and frames can be pushed through it".
    pub fn open_session_with(&self, request: &SessionRequest) -> Result<Box<dyn EngineSession>> {
        let session = Session::open(&self.library, request)?;
        Ok(Box::new(PluginSession {
            session,
            capabilities: self.capabilities.clone(),
            working: WorkingSetRequest::default(),
            calls: 0,
        }))
    }
}

impl InferenceEngine for PluginEngine {
    fn id(&self) -> &str {
        "inference-plugin"
    }

    fn display_name(&self) -> String {
        format!("Loaded plugin: {}", self.library.path.display())
    }

    fn kind(&self) -> EngineKind {
        EngineKind::Plugin
    }

    fn capabilities(&self) -> Capabilities {
        let mut caps = Capabilities::from(&self.capabilities);
        caps.devices = self.devices.iter().map(DeviceInfo::describe).collect();
        caps
    }

    fn status(&self) -> EngineStatus {
        let caps = self.capabilities();
        if !caps.scale && !caps.interpolate && !caps.restore {
            return EngineStatus::Unavailable("plugin declares no operation it can run".into());
        }
        if self.devices.is_empty() {
            return EngineStatus::Unavailable("plugin reports no usable device".into());
        }
        EngineStatus::Ready
    }

    fn quality_note(&self) -> String {
        format!(
            "model backend {}{}",
            self.capabilities.backend,
            self.capabilities
                .vendor
                .as_deref()
                .map(|v| format!(" on {v}"))
                .unwrap_or_default()
        )
    }

    fn open_session(&self, request: &SessionRequest) -> Result<Box<dyn EngineSession>> {
        self.open_session_with(request)
    }

    fn preferred_device_index(&self) -> Option<u32> {
        // Ranked by what the device *is* before how much memory it reports: a
        // backend is not obliged to know its own free memory, and on a machine
        // with an integrated and a discrete GPU "the one with the highest number"
        // would otherwise pick whichever came last in a list of zeroes.
        self.devices
            .iter()
            .max_by_key(|device| {
                let rank = match device.device_type {
                    super::abi::SR_DEVICE_DISCRETE => 3u8,
                    super::abi::SR_DEVICE_INTEGRATED => 2,
                    super::abi::SR_DEVICE_VIRTUAL => 1,
                    _ => 0,
                };
                (rank, device.free_bytes().unwrap_or(0))
            })
            .map(|device| device.index)
    }

    fn backend_free_mib(&self) -> Option<u64> {
        self.devices
            .iter()
            .filter_map(|device| device.free_bytes())
            .max()
            .map(|bytes| bytes / (1024 * 1024))
    }
}

/// A live plugin session, adapted to the engine's session trait.
pub struct PluginSession {
    session: Session,
    capabilities: PluginCapabilities,
    working: WorkingSetRequest,
    calls: u64,
}

impl PluginSession {
    /// Working-set change that the *job* can carry: tiling and batch size.
    fn tile(&self) -> Option<(u32, u32)> {
        if self.working.tile >= self.capabilities.tile_min && self.working.tile > 0 {
            Some((self.working.tile, self.working.tile))
        } else {
            None
        }
    }
}

impl EngineSession for PluginSession {
    fn engine_id(&self) -> &str {
        "inference-plugin"
    }

    fn interpolate(
        &mut self,
        inputs: &mut [FrameBuffer],
        outputs: &mut [FrameBuffer],
        options: &JobOptions,
    ) -> Result<ExecOutcome> {
        if options.multiplier < 2 {
            return Err(Error::Unsupported(
                "interpolation with a multiplier below 2 is a copy, not a model call".into(),
            ));
        }
        let window = inputs.len() as u32;
        if window > self.capabilities.max_temporal_window {
            return Err(Error::Unsupported(format!(
                "the model accepts at most {} input frames per call, {} given",
                self.capabilities.max_temporal_window,
                window
            )));
        }
        let mut options = options.clone();
        options.tile = self.tile();
        self.calls += 1;
        self.session.interpolate(inputs, outputs, &options)
    }

    fn restore(
        &mut self,
        frames: &mut [FrameBuffer],
        options: &JobOptions,
    ) -> Result<ExecOutcome> {
        let mut options = options.clone();
        options.tile = self.tile();
        if frames.len() as u32 > self.capabilities.max_batch {
            return Err(Error::Unsupported(format!(
                "the model accepts at most {} frames per restore call, {} given",
                self.capabilities.max_batch,
                frames.len()
            )));
        }
        self.calls += 1;
        self.session.restore(frames, &options)
    }

    fn reconfigure(&mut self, request: &WorkingSetRequest) -> Result<()> {
        // Tiling and batch size travel with each job, so they can change under a
        // live session. Precision, offload and block swapping are properties of
        // the loaded weights: those need the session reopened, and the executor
        // does that rather than pretending the change took effect.
        let needs_reopen = request.precision != self.working.precision
            || request.offload != self.working.offload
            || request.block_swap != self.working.block_swap;
        self.working = request.clone();
        if needs_reopen {
            return Err(Error::Unsupported(
                "precision/offload/block-swap changes require reopening the model session".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_without_plugin(ff: Arc<Ffmpeg>) -> super::super::EngineRegistry {
        // Built directly so the test does not depend on a plugin being present.
        let mut engines: Vec<Arc<dyn InferenceEngine>> = Vec::new();
        engines.push(Arc::new(FfmpegEngine::new(Arc::clone(&ff))));
        super::super::EngineRegistry::from_engines(engines)
    }

    #[test]
    fn baseline_engine_is_honest_about_what_it_cannot_do() {
        let Ok(ff) = Ffmpeg::from_paths("ffmpeg".into(), "ffprobe".into()) else {
            return;
        };
        let ff = Arc::new(ff);
        let engine = FfmpegEngine::new(Arc::clone(&ff));
        let caps = engine.capabilities();
        assert!(caps.scale);
        assert!(!caps.restore, "the baseline must not claim restoration");
        assert!(engine.quality_note().contains("cannot invent detail"));
        assert!(caps.notes.iter().any(|n| n.contains("duplicates frames")));
    }

    #[test]
    fn registry_always_has_a_baseline_to_select() {
        let Ok(ff) = Ffmpeg::from_paths("ffmpeg".into(), "ffprobe".into()) else {
            return;
        };
        let registry = registry_without_plugin(Arc::new(ff));
        let selected = registry.select(super::super::InferenceTask::Scale);
        assert!(selected.is_some(), "scaling must always be available");
        assert_eq!(selected.unwrap().id(), "ffmpeg-baseline");
        assert!(registry
            .select(super::super::InferenceTask::Restore)
            .is_none());
        assert!(registry.model_engine().is_none());
    }
}
