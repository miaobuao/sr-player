//! Turning analysis into a plan a human can read.
//!
//! A plan is not just an argument list: every decision carries the reason it was
//! made, so the UI can show *why* a 29.97 DVD is being field-matched rather than
//! deinterlaced, why interpolation was disabled for a shot, or why the audio was
//! left completely alone. "Zero tuning" only works if the reasoning is visible.

use crate::audio::loudness::AudioAnalysis;
use crate::error::{Error, Result};
use crate::ffmpeg::{EncoderPreference, Ffmpeg, SelectedAudioEncoder, SelectedVideoEncoder, VideoCodec};
use crate::media::classify::{TemporalMode, TemporalReport};
use crate::media::manifest::{MediaManifest, PreservationInventory};
use crate::media::scene::SceneReport;
use crate::pipeline::policy::{plan_vram, VramBudget, WorkingSet};
use crate::pipeline::profile::{
    InterpolationMethod, LoudnessTarget, OutputSettings, RestorationProfile, RestorationSettings,
};
use crate::time::{Rational, Timestamp};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoSourceInfo {
    pub width: u32,
    pub height: u32,
    /// Raster after undoing non-square pixels — what the model actually sees.
    pub square_width: u32,
    pub square_height: u32,
    pub sar: Rational,
    pub fps: Rational,
    pub duration_seconds: f64,
    pub frames: u64,
    pub pix_fmt: Option<String>,
    pub is_hdr: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterpolationPlan {
    pub enabled: bool,
    pub method: InterpolationMethod,
    pub source_fps: Rational,
    pub target_fps: Rational,
    pub multiplier: u32,
    /// Always true in this build: cuts are never crossed.
    pub scene_cuts_respected: bool,
    pub note: String,
}

/// Who turns input frames into output frames.
///
/// This field exists to keep the log honest. [`VideoExecutor::FfmpegSinglePass`]
/// means one FFmpeg process with one filter chain; [`VideoExecutor::Chunked`]
/// means this crate decodes, hands each chunk to its own encoder with its own
/// per-shot filter chain, and concatenates the results. A plan may only claim the
/// second when something in it genuinely needs per-chunk control — today that is
/// per-shot re-grain, and later it is the model stage, which cannot be expressed
/// as an FFmpeg filter at all.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VideoExecutor {
    /// One FFmpeg pass with a filter chain. Deterministic; no model involved.
    FfmpegSinglePass,
    /// Decode, per-chunk encode with a per-shot chain, concatenate.
    Chunked,
}

impl VideoExecutor {
    /// The textual form. `kebab-case` above is not decoration: this value is both
    /// stored in the plan and printed, and two spellings for one value is how a
    /// log line and a stored plan quietly stop agreeing. `the_text_form_matches_
    /// what_serde_writes` holds the two together.
    pub fn as_str(self) -> &'static str {
        match self {
            VideoExecutor::FfmpegSinglePass => "ffmpeg-single-pass",
            VideoExecutor::Chunked => "chunked",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoPlan {
    pub source: VideoSourceInfo,
    pub temporal_mode: TemporalMode,
    pub temporal_note: String,
    /// The cadence decision, shared by the analyser, the decoder and the model.
    pub cadence: CadencePlan,
    pub ivtc: bool,
    pub deinterlace: Option<String>,
    pub target_width: u32,
    pub target_height: u32,
    pub upscale_factor: f32,
    pub interpolation: InterpolationPlan,
    /// Who executes the video work, and therefore what the log may claim.
    pub executor: VideoExecutor,
    /// The restoration settings this plan resolved. Present on every plan: an
    /// executor row that cannot say whether restoration was asked for cannot be
    /// checked against what actually ran.
    pub restoration: RestorationSettings,
    pub regrain_strength: f32,
    /// Per-shot re-grain strength, indexed by shot.
    ///
    /// Empty when re-grain is off. Filled from a measurement of the source, because
    /// a film's grain varies from shot to shot by more than any single setting can
    /// cover — and only the per-chunk executor can apply it, since FFmpeg's `noise`
    /// filter takes one constant for a whole chain. `regrain_strength` stays as the
    /// fallback for the single-pass path and for shots with no measurement.
    pub regrain_per_shot: Vec<f32>,
    /// Filters FFmpeg applies on the encode side. For the native executor this is
    /// the post-model chain only; the model is not an FFmpeg filter.
    pub filter_chain: String,
    /// Filters FFmpeg applies on the decode side before the model sees a frame.
    pub decode_chain: String,
    pub encoder: SelectedVideoEncoder,
    /// Every usable encoder, best first. The runner walks this list when an
    /// encoder turns out to be advertised but unusable (an AMD encoder on an
    /// NVIDIA box is the classic case).
    pub encoder_chain: Vec<SelectedVideoEncoder>,
    pub pix_fmt: String,
    pub estimated_frames: u64,
}

impl VideoPlan {
    /// Encoder-side arguments (input args are returned separately).
    pub fn encoder_args(&self) -> Vec<String> {
        let mut args = vec!["-c:v".to_string(), self.encoder.name.clone()];
        args.extend(self.encoder.quality_args.iter().cloned());
        args.push("-pix_fmt".to_string());
        args.push(self.pix_fmt.clone());
        args
    }

    pub fn input_args(&self) -> Vec<String> {
        self.encoder.input_args.clone()
    }

    /// True when the plan's video work must go through the chunked executor.
    pub fn is_chunked(&self) -> bool {
        self.executor == VideoExecutor::Chunked
    }

    /// Frame rate of the raw frames the chunked executor pushes around.
    ///
    /// With RIFE the pipe already carries the target rate, because the model
    /// produces the extra frames. Without it the pipe carries the decoded rate and
    /// any frame-rate change happens in the post chain — so declaring the target
    /// rate here would silently stretch the output.
    pub fn pipe_fps(&self) -> Rational {
        if self.interpolation.enabled && self.interpolation.method == InterpolationMethod::Rife {
            self.interpolation.target_fps
        } else {
            self.cadence.fps
        }
    }
}

/// Compact, persistable form of the audio analysis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioSummary {
    pub stream_index: usize,
    pub channels: u16,
    pub sample_rate: u32,
    pub programme_lufs: f64,
    pub programme_lufs_ffmpeg: Option<f64>,
    pub dialogue_lufs: Option<f64>,
    pub ldr_lu: Option<f64>,
    pub loudness_range_lu: f64,
    pub true_peak_dbtp: Option<f64>,
    pub speech_ratio: f32,
    pub confidence: f32,
    pub apply: bool,
    pub gain_db: f32,
    pub duck_db: f32,
    pub reason: String,
    pub notes: Vec<String>,
}

impl AudioSummary {
    pub fn from_analysis(analysis: &AudioAnalysis) -> Self {
        AudioSummary {
            stream_index: analysis.stream_index,
            channels: analysis.channels,
            sample_rate: analysis.sample_rate,
            programme_lufs: analysis.programme_lufs,
            programme_lufs_ffmpeg: analysis.programme_lufs_ffmpeg,
            dialogue_lufs: analysis.dialogue_lufs,
            ldr_lu: analysis.ldr_lu,
            loudness_range_lu: analysis.loudness_range_lu,
            true_peak_dbtp: analysis.true_peak_dbtp,
            speech_ratio: analysis.speech_ratio,
            confidence: analysis.dialogue_confidence,
            apply: analysis.decision.apply,
            gain_db: analysis.decision.dialogue_gain_db,
            duck_db: analysis.decision.duck_db,
            reason: analysis.decision.reason.clone(),
            notes: analysis.notes.clone(),
        }
    }

    pub fn summary_line(&self) -> String {
        format!(
            "programme {:.1} LUFS, dialogue {}, LDR {}{}",
            self.programme_lufs,
            self.dialogue_lufs
                .map(|d| format!("{d:.1} LUFS"))
                .unwrap_or_else(|| "n/a".into()),
            self.ldr_lu
                .map(|l| format!("{l:.1} LU"))
                .unwrap_or_else(|| "n/a".into()),
            if self.apply {
                format!(" → {:+.1} dB dialogue, -{:.1} dB masking band", self.gain_db, self.duck_db)
            } else {
                format!(" → left alone ({})", self.reason)
            }
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioPlan {
    pub enabled: bool,
    pub summary: Option<AudioSummary>,
    pub encoder: SelectedAudioEncoder,
    pub loudness: LoudnessTarget,
    pub keep_original: bool,
    pub note: String,
}

impl AudioPlan {
    pub fn describe(&self) -> String {
        if !self.enabled {
            return "audio: copied through untouched".to_string();
        }
        match &self.summary {
            Some(summary) => format!(
                "audio: {} · enhanced track {} ({}){}{}",
                summary.summary_line(),
                self.encoder.name,
                if self.encoder.lossless { "lossless" } else { "lossy" },
                if self.keep_original {
                    " · original tracks preserved"
                } else {
                    ""
                },
                if self.note.is_empty() {
                    String::new()
                } else {
                    format!(" · {}", self.note)
                }
            ),
            None => format!("audio: {} (no analysis)", self.encoder.name),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanRequest {
    pub job_id: String,
    pub input: PathBuf,
    pub output: PathBuf,
    pub profile: RestorationProfile,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversionPlan {
    pub job_id: String,
    pub input: PathBuf,
    pub output: PathBuf,
    pub profile_name: String,
    pub video: VideoPlan,
    pub audio: AudioPlan,
    pub output_settings: OutputSettings,
    /// What the source had, so the mux and the QC stage can prove nothing was
    /// dropped.
    pub inventory: PreservationInventory,
    pub vram: VramBudget,
    pub working_set: WorkingSet,
    pub notes: Vec<String>,
    pub warnings: Vec<String>,
    pub created_ms: u64,
}

impl ConversionPlan {
    pub fn summary_rows(&self) -> Vec<(String, String)> {
        let mut rows = vec![
            ("input".into(), self.input.display().to_string()),
            ("output".into(), self.output.display().to_string()),
            (
                "video".into(),
                format!(
                    "{}x{} {} -> {}x{} @ {:.3} fps",
                    self.video.source.square_width,
                    self.video.source.square_height,
                    self.video.encoder.codec.as_str(),
                    self.video.target_width,
                    self.video.target_height,
                    self.video.interpolation.target_fps.to_f64()
                ),
            ),
            ("cadence".into(), self.video.temporal_note.clone()),
            (
                "executor".into(),
                match self.video.executor {
                    VideoExecutor::Chunked => format!(
                        "chunked · per-shot filter chain over {} measured shot(s) · restore {}",
                        self.video.regrain_per_shot.len(),
                        if self.video.restoration.enabled {
                            self.video.restoration.model.clone()
                        } else {
                            "off".to_string()
                        }
                    ),
                    VideoExecutor::FfmpegSinglePass => {
                        "FFmpeg single pass (one filter chain, no model)".to_string()
                    }
                },
            ),
            (
                "decode filters".into(),
                if self.video.decode_chain.is_empty() {
                    "(none: frames reach the model as stored)".into()
                } else {
                    self.video.decode_chain.clone()
                },
            ),
            (
                "filters".into(),
                if self.video.filter_chain.is_empty() {
                    "(none)".into()
                } else {
                    self.video.filter_chain.clone()
                },
            ),
            ("encoder".into(), self.video.encoder.describe()),
            ("audio".into(), self.audio.describe()),
            ("vram".into(), self.vram.describe()),
            (
                "working set".into(),
                format!(
                    "{} (~{:.1} GiB estimated)",
                    self.working_set.describe(),
                    self.working_set.estimated_mib() as f64 / 1024.0
                ),
            ),
        ];
        for warning in &self.warnings {
            rows.push(("warning".into(), warning.clone()));
        }
        rows
    }

    /// One-line summary for the log header.
    pub fn describe(&self) -> String {
        format!(
            "{}x{} @ {:.3} fps, {}, {}, {}, {}",
            self.video.target_width,
            self.video.target_height,
            self.video.interpolation.target_fps.to_f64(),
            self.video.encoder.describe(),
            self.video.executor.as_str(),
            self.video.interpolation.method.as_str(),
            if self.audio.enabled {
                "audio remaster"
            } else {
                "audio passthrough"
            }
        )
    }
}

/// Target raster: keep the source aspect in square pixels, never exceed
/// `max_upscale`, never round to an odd width (encoders reject it).
pub fn choose_target(source_w: u64, source_h: u64, max_upscale: f32) -> (u32, u32) {
    if source_w == 0 || source_h == 0 {
        return (1920, 1080);
    }
    let max_upscale = max_upscale.max(1.0);
    // "Restore to a resolution that carries real information": up to 1080p for
    // SD sources, keep native resolution above that.
    let desired_h = if source_h < 1080 { 1080u64 } else { source_h };
    let capped_h = ((source_h as f32 * max_upscale).round() as u64).max(source_h);
    let target_h = desired_h.min(capped_h);
    let mut target_w = ((target_h as f64 * source_w as f64 / source_h as f64).round() as u64).max(2);
    if target_w % 2 == 1 {
        target_w += 1;
    }
    let mut target_h = target_h.max(2);
    if target_h % 2 == 1 {
        target_h += 1;
    }
    (target_w as u32, target_h as u32)
}

/// The cadence decision, in one place: which filters run before anything else
/// touches the picture, and what frame rate they leave behind.
///
/// This is the single source of truth for the temporal decision. The scene
/// detector, the FFmpeg encode path and the native executor all build it from
/// here, so shot indices, model input frames and encoded frames cannot drift
/// apart — which is exactly the class of bug that produces a frame interpolated
/// across a cut, or a shot boundary that protects the wrong frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CadencePlan {
    pub chain: Option<String>,
    /// Frame rate after `chain` has been applied.
    pub fps: Rational,
    /// Frame numbering changes: the source frame count is not the decoded one.
    pub renumbers_frames: bool,
    pub note: String,
}

impl CadencePlan {
    pub fn plan(mode: TemporalMode, source_fps: Rational, confident: bool) -> Self {
        let (chain, drops_frames, note) = match mode {
            TemporalMode::Interlaced => (
                Some("bwdif=mode=send_frame:parity=auto:deint=all".to_string()),
                false,
                "motion-adaptive deinterlace (no frame dropping)".to_string(),
            ),
            TemporalMode::Mixed => (
                Some("bwdif=mode=send_frame:parity=auto:deint=interlaced".to_string()),
                false,
                "mixed cadence: deinterlace only the interlaced parts".to_string(),
            ),
            TemporalMode::Telecine => (
                // Field match first, then remove the duplicated frames. A blanket
                // deinterlace here would throw away half the temporal information.
                Some("fieldmatch=order=auto:combmatch=full,decimate".to_string()),
                true,
                if confident {
                    "3:2 pulldown removed (inverse telecine)".to_string()
                } else {
                    "cadence classifier was not confident; inverse telecine applied conservatively"
                        .to_string()
                },
            ),
            _ => (
                None,
                false,
                "progressive or unknown cadence: frames are passed through untouched".to_string(),
            ),
        };
        let fps = if drops_frames {
            post_ivtc_fps(source_fps)
        } else {
            source_fps
        };
        CadencePlan {
            chain,
            fps,
            renumbers_frames: drops_frames,
            note,
        }
    }

    pub fn chain_str(&self) -> Option<&str> {
        self.chain.as_deref()
    }

    /// What the log should say about the cadence.
    pub fn describe(&self, mode: TemporalMode) -> String {
        format!(
            "{} — {}{}",
            mode.as_str(),
            self.note,
            if self.renumbers_frames {
                format!("; decoded at {:.3} fps", self.fps.to_f64())
            } else {
                String::new()
            }
        )
    }
}

/// Frame rate after inverse telecine: 3:2 pulldown collapses to 4/5 of the rate.
pub fn post_ivtc_fps(fps: Rational) -> Rational {
    fps.checked_mul(&Rational::new(4, 5).unwrap_or(Rational::ONE))
        .unwrap_or(fps)
}

/// The cadence decision for a probed file.
///
/// Both the pipeline's scene analysis and its plan go through this, so the shot
/// list and the encoder are guaranteed to be talking about the same frames.
pub fn cadence_for(
    manifest: &MediaManifest,
    temporal: &TemporalReport,
) -> Result<CadencePlan> {
    let fps = manifest
        .primary_video()
        .and_then(|v| v.fps())
        .ok_or_else(|| {
            Error::Unsupported("the video frame rate is unknown".into())
        })?;
    Ok(CadencePlan::plan(
        temporal.mode,
        fps,
        temporal.mode.is_confident(),
    ))
}

/// Geometry filters: the resize that turns a non-square-pixel raster into the
/// target, or the upscale after a restoration model.
fn geometry_filters(video: &VideoPlan, source_square: (u32, u32)) -> Vec<String> {
    if (video.target_width, video.target_height) == source_square {
        Vec::new()
    } else {
        vec![format!(
            "scale={}:{}:flags=lanczos",
            video.target_width, video.target_height
        )]
    }
}

/// Re-grain. Still an FFmpeg noise generator: the per-shot grain estimator that
/// should drive it does not exist yet, which is why the plan only enables this
/// when a profile asks for it explicitly.
fn regrain_filter(video: &VideoPlan) -> Vec<String> {
    if video.regrain_strength > 0.0 {
        vec![format!(
            "noise=alls={:.0}:allf=t+u",
            video.regrain_strength.clamp(1.0, 30.0)
        )]
    } else {
        Vec::new()
    }
}

fn encoder_tail(video: &VideoPlan) -> Vec<String> {
    if video.encoder.requires_hwupload {
        vec![format!("format={},hwupload", video.encoder.pix_fmt)]
    } else {
        Vec::new()
    }
}

/// Builds the `-vf` chain for the single-pass FFmpeg executor. Order matters and
/// is fixed: deinterlace, geometry, grain, encoder upload.
///
/// No interpolation filter appears here, and that is deliberate. The only
/// interpolation this product performs is RIFE, which is not an FFmpeg filter, so
/// a plan that asks for it is not executed by this path at all. The previous
/// build emitted `minterpolate` here when a model was unavailable, which made a
/// request for RIFE silently become a request for a different algorithm.
pub fn build_filter_chain(video: &VideoPlan, source_square: (u32, u32)) -> String {
    let mut filters: Vec<String> = Vec::new();

    if let Some(deinterlace) = &video.deinterlace {
        filters.push(deinterlace.clone());
    }
    filters.extend(geometry_filters(video, source_square));
    filters.extend(regrain_filter(video));
    filters.extend(encoder_tail(video));
    filters.join(",")
}

/// The chain applied to frames after the chunked path produced them.
///
/// The post-model chain: the final resize (Lanczos, because the pixels at that
/// size were not invented by a model), re-grain and the encoder upload.
pub fn build_post_chain(video: &VideoPlan, model_output: (u32, u32)) -> String {
    let mut filters: Vec<String> = Vec::new();
    if (video.target_width, video.target_height) != model_output {
        filters.push(format!(
            "scale={}:{}:flags=lanczos",
            video.target_width, video.target_height
        ));
    }
    filters.extend(regrain_filter(video));
    filters.extend(encoder_tail(video));
    filters.join(",")
}

/// The chain applied before the model sees a frame: cadence only, at source
/// resolution, so the model works on the picture rather than on a resample of it.
pub fn build_decode_chain(video: &VideoPlan) -> String {
    video
        .cadence
        .chain
        .clone()
        .unwrap_or_else(String::new)
}

/// Geometry the chunk encoder must expect from the model stage.
///
/// With restoration enabled the network multiplies the input by its own scale
/// factor; without it frames pass through at source resolution. Encoders reject
/// odd widths, so the result is rounded up to even.
pub fn model_output_geometry(video: &VideoPlan) -> (u32, u32) {
    let even = |value: u32| {
        let rounded = value.max(2);
        if rounded % 2 == 1 {
            rounded + 1
        } else {
            rounded
        }
    };
    let width = video.source.square_width;
    let height = video.source.square_height;
    if video.restoration.enabled {
        let scale = video.restoration.scale.max(1);
        (
            even(width.saturating_mul(scale)),
            even(height.saturating_mul(scale)),
        )
    } else {
        (even(width), even(height))
    }
}

/// Refuses a restoration request that nothing in this binary can serve.
///
/// The alternative that used to be here — carry on, reach the target raster with
/// a Lanczos resample, and describe the result as restored — is the single most
/// misleading thing this project has done, because the output really did get
/// bigger and smoother, so nothing looked wrong.
fn require_restoration_runtime(settings: &RestorationSettings) -> Result<()> {
    if !settings.enabled {
        return Ok(());
    }
    Err(Error::Unsupported(format!(
        "restoration with {} was requested but no restoration network can run it: the native \
         Real-ESRGAN runtime (native/sr-native over ncnn) is not built into this binary yet, and \
         its weights are not installed. Nothing is substituted for it — the previous build \
         resampled with Lanczos and reported it as restoration. Use the `deterministic` or \
         `preview` profile to run the rest of the pipeline.",
        settings.model
    )))
}

/// Resolves the interpolation decision, refusing rather than substituting.
///
/// There is exactly one interpolator this product runs: RIFE, through the native
/// ncnn runtime. It is not wired up yet, so a request for it fails here, before
/// any pixel moves, with a message that says what is missing. The alternative —
/// emitting `minterpolate`, or duplicating frames and calling it interpolation —
/// is the specific dishonesty this redesign exists to remove: the log would say
/// "interpolated" while the picture got something else.
fn resolve_interpolation(
    profile: &RestorationProfile,
    source_fps: Rational,
) -> Result<InterpolationPlan> {
    let multiplier = profile.interpolation.multiplier.max(1);
    let requested = profile.interpolation.method;
    if requested == InterpolationMethod::Off || multiplier <= 1 {
        return Ok(InterpolationPlan {
            enabled: false,
            method: InterpolationMethod::Off,
            source_fps,
            target_fps: source_fps,
            multiplier: 1,
            scene_cuts_respected: true,
            note: "interpolation disabled: source cadence preserved".into(),
        });
    }

    Err(Error::Unsupported(format!(
        "{}x interpolation was requested but no interpolator can run it: the native RIFE runtime \
         (native/sr-native over ncnn) is not built into this binary yet. Nothing is substituted \
         for it — pass `--interpolate off`, or choose a profile whose interpolation is off, if \
         you want the rest of the pipeline to run.",
        multiplier
    )))
}

/// The whole decision, in one place.
pub fn build_plan(
    ff: &Ffmpeg,
    manifest: &MediaManifest,
    temporal: &TemporalReport,
    scenes: &SceneReport,
    audio: Option<&AudioAnalysis>,
    request: &PlanRequest,
) -> Result<ConversionPlan> {
    let profile = &request.profile;
    let mut notes = Vec::new();
    let mut warnings = Vec::new();

    let video_stream = manifest.primary_video().ok_or_else(|| {
        Error::Unsupported("the file has no video stream to convert".into())
    })?;
    let fps = video_stream.fps().ok_or_else(|| {
        Error::Unsupported("the video frame rate is unknown".into())
    })?;
    let (raw_w, raw_h) = video_stream.size().unwrap_or((0, 0));
    let (square_w, square_h) = video_stream.square_pixel_size().unwrap_or((raw_w, raw_h));
    // Content, not container: a leading timestamp offset is not runtime, and every
    // estimate below - frames, progress, the QC comparison - is about content.
    let duration_seconds = manifest.content_duration_seconds();

    // --- temporal -----------------------------------------------------------
    let cadence = cadence_for(manifest, temporal)?;
    let ivtc = temporal.mode.needs_ivtc();
    let deinterlace = cadence.chain.clone();
    if ivtc && !temporal.mode.is_confident() {
        warnings.push(
            "cadence classifier was not confident; IVTC applied conservatively".into(),
        );
    }
    let effective_fps = cadence.fps;
    let temporal_note = cadence.describe(temporal.mode);
    let temporal_note = format!("{} — {}", temporal_note, temporal.summary());

    // --- geometry -----------------------------------------------------------
    let (target_width, target_height) =
        choose_target(square_w.max(2) as u64, square_h.max(2) as u64, profile.restoration.max_upscale);
    let upscale_factor = target_height as f32 / square_h.max(1) as f32;

    // --- model tasks --------------------------------------------------------
    //
    // Both of these are checked before anything else about the video is decided,
    // because they are the two things this build cannot do and must not pretend
    // to. A refusal here is a refusal to produce a file at all: publishing one
    // that quietly skipped the model is the outcome the check exists to prevent.
    require_restoration_runtime(&profile.restoration)?;
    let interpolation = resolve_interpolation(profile, effective_fps)?;
    if interpolation.enabled && interpolation.method != InterpolationMethod::Off {
        notes.push(format!(
            "interpolation: {} ({} cut(s) in this file are shot boundaries the interpolation \
             is never allowed to cross)",
            interpolation.note,
            scenes.cut_count()
        ));
    }

    // --- encoder ------------------------------------------------------------
    let pref = EncoderPreference {
        prefer_hardware: profile.output.prefer_hardware,
        allow_software: true,
        prefer_10bit: profile.output.prefer_10bit,
        codec_order: vec![VideoCodec::Av1, VideoCodec::Hevc, VideoCodec::H264],
        forced: None,
        quality: profile.output.quality,
    };
    let encoder_chain = ff.video_encoder_chain(&pref);
    let encoder = encoder_chain
        .first()
        .cloned()
        .ok_or_else(|| crate::error::Error::NoEncoder {
            tried: pref
                .codec_order
                .iter()
                .map(|c| c.as_str().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        })?;
    notes.push(format!(
        "encoder: {} — {}{}",
        encoder.describe(),
        encoder.note,
        if encoder_chain.len() > 1 {
            format!(
                " ({} fallbacks available)",
                encoder_chain.len() - 1
            )
        } else {
            String::new()
        }
    ));

    let estimated_frames = (duration_seconds * interpolation.target_fps.to_f64()).round() as u64;

    // The executor is decided *after* the grain measurement below, because
    // per-shot re-grain is the thing that forces the chunked path: a filter chain
    // built once cannot give two shots different strengths.

    // ---- grain ------------------------------------------------------------
    //
    // Measured only when re-grain was asked for: it costs a decode pass at native
    // resolution, and a job that is not re-graining should not pay for it. The
    // measurement is what makes per-shot re-grain more than a name — the strength
    // comes from the film rather than from a setting.
    let mut regrain_per_shot: Vec<f32> = Vec::new();
    if profile.output.regrain_strength > 0.0 {
        // A local reporter and flag, because `build_plan` has neither: it is called
        // with a manifest and two analyses, not with the job's channels. The cost is
        // that this pass cannot be cancelled from outside and does not log its
        // measurement - it is bounded to `max_frames` frames, and the amplitudes end
        // up in the plan, which the caller stores. Threading the job's channels into
        // the planner would be the cleaner shape and is a larger change than this
        // pass is worth making here.
        let bus = crate::EventBus::new();
        let reporter = crate::Reporter::new(bus);
        let cancel = std::sync::atomic::AtomicBool::new(false);
        match crate::pipeline::grain::measure_shots(
            ff,
            manifest,
            &scenes.shots,
            cadence.chain_str().as_deref(),
            &crate::pipeline::grain::GrainOptions::default(),
            &reporter,
            &cancel,
        ) {
            Ok(measured) => {
                // Silent here: `build_plan` has no reporter, and the values are in the
                // plan the caller stores and logs. A pass that cannot report is not a
                // reason to give it a channel it has never needed.
                regrain_per_shot = measured
                    .iter()
                    .map(|shot| crate::pipeline::grain::strength_for(&shot.estimate))
                    .collect();
            }
            Err(_) => {
                // A measurement that fails must not fail the job. The fallback is the
                // configured strength, which is exactly what ran before this existed.
            }
        }
    }
    // --- who executes the video work ----------------------------------------
    //
    // Only the chunked executor can give one shot a different filter chain from
    // its neighbour, so a measured per-shot re-grain forces it. The model stage
    // will force it for the same reason — a network is not an FFmpeg filter —
    // but there is no model stage in this build, so nothing else does.
    let executor = if regrain_per_shot.is_empty() {
        VideoExecutor::FfmpegSinglePass
    } else {
        VideoExecutor::Chunked
    };
    let measured_shots = regrain_per_shot.len();

    let mut video = VideoPlan {
        source: VideoSourceInfo {
            width: raw_w as u32,
            height: raw_h as u32,
            square_width: square_w as u32,
            square_height: square_h as u32,
            sar: video_stream.sar(),
            fps,
            duration_seconds,
            frames: video_stream
                .base
                .nb_frames
                .unwrap_or_else(|| (duration_seconds * fps.to_f64()).round() as u64),
            pix_fmt: video_stream.pix_fmt.clone(),
            is_hdr: video_stream.is_hdr(),
        },
        temporal_mode: temporal.mode,
        temporal_note,
        cadence,
        ivtc,
        deinterlace,
        target_width,
        target_height,
        upscale_factor,
        interpolation,
        executor,
        restoration: profile.restoration.clone(),
        regrain_strength: profile.output.regrain_strength,
        regrain_per_shot,
        filter_chain: String::new(),
        decode_chain: String::new(),
        encoder,
        encoder_chain,
        pix_fmt: String::new(),
        estimated_frames,
    };
    video.pix_fmt = video.encoder.pix_fmt.clone();
    video.decode_chain = build_decode_chain(&video);
    video.filter_chain = if video.is_chunked() {
        build_post_chain(&video, model_output_geometry(&video))
    } else {
        build_filter_chain(&video, (square_w as u32, square_h as u32))
    };

    match video.executor {
        VideoExecutor::Chunked => {
            notes.push(format!(
                "video executor: chunked — one encode per chunk with a per-shot filter chain \
                 over {measured_shots} measured shot(s), checkpointed as each one completes"
            ));
        }
        VideoExecutor::FfmpegSinglePass => {
            notes.push(
                "video executor: one FFmpeg pass — no model session is opened, so nothing is \
                 restored or synthesised"
                    .into(),
            );
        }
    }

    if video.source.is_hdr {
        warnings.push(
            "HDR source: the SDR restoration path is bypassed. Tone mapping is deliberately not \
             applied automatically, because it would irreversibly change the master."
                .into(),
        );
    }
    if upscale_factor > 2.5 && video.executor == VideoExecutor::FfmpegSinglePass {
        warnings.push(format!(
            "upscaling by {upscale_factor:.2}x invents no detail: the deterministic path only \
             resamples. A restoration model is required for real detail."
        ));
    }
    if !profile.restoration.enabled {
        notes.push(
            "restoration: off (the `deterministic` configuration) — no detail is invented, and \
             the output raster is reached by a deterministic resize"
                .into(),
        );
    }

    // --- audio --------------------------------------------------------------
    let audio_plan = build_audio_plan(ff, profile, audio, &mut warnings);

    // --- budget -------------------------------------------------------------
    let gpus = crate::gpu::probe();
    let vram = plan_vram(
        profile.gpu.max_ai_vram_mib,
        profile.gpu.reserve_mib,
        &gpus,
    );
    for note in &vram.notes {
        notes.push(note.clone());
    }
    let working_set = WorkingSet::initial(&profile.restoration);
    if profile.restoration.enabled && !vram.fits(working_set.estimated_mib()) {
        warnings.push(format!(
            "the preferred working set (~{:.1} GiB) exceeds the {:.1} GiB budget: the degrade \
             ladder will engage",
            working_set.estimated_mib() as f64 / 1024.0,
            vram.ai_budget_mib as f64 / 1024.0
        ));
    }

    Ok(ConversionPlan {
        job_id: request.job_id.clone(),
        input: request.input.clone(),
        output: request.output.clone(),
        profile_name: profile.name.clone(),
        video,
        audio: audio_plan,
        output_settings: profile.output.clone(),
        inventory: crate::media::manifest::preservation_inventory(manifest),
        vram,
        working_set,
        notes,
        warnings,
        created_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    })
}

fn build_audio_plan(
    ff: &Ffmpeg,
    profile: &RestorationProfile,
    audio: Option<&AudioAnalysis>,
    warnings: &mut Vec<String>,
) -> AudioPlan {
    let encoder = ff.pick_audio_encoder(profile.audio.lossless_enhanced);
    if !profile.audio.enabled {
        return AudioPlan {
            enabled: false,
            summary: audio.map(AudioSummary::from_analysis),
            encoder,
            loudness: profile.audio.loudness.clone(),
            keep_original: profile.audio.keep_original_tracks,
            note: "audio processing disabled in the profile".into(),
        };
    }
    match audio {
        None => AudioPlan {
            enabled: false,
            summary: None,
            encoder,
            loudness: profile.audio.loudness.clone(),
            keep_original: profile.audio.keep_original_tracks,
            note: "no audio analysis available; tracks are copied".into(),
        },
        Some(analysis) => {
            let summary = AudioSummary::from_analysis(analysis);
            if analysis.decision.apply && !analysis.decision.confidence.is_finite() {
                warnings.push("dialogue confidence was not finite: treating as 0".into());
            }
            AudioPlan {
                enabled: true,
                summary: Some(summary),
                encoder,
                loudness: profile.audio.loudness.clone(),
                keep_original: profile.audio.keep_original_tracks,
                note: analysis.decision.summary(),
            }
        }
    }
}

/// Convenience for the CLI and UI: a plan for a file, with analysis disabled.
pub fn plan_without_analysis(
    ff: &Ffmpeg,
    manifest: &MediaManifest,
    request: &PlanRequest,
) -> Result<ConversionPlan> {
    let temporal = TemporalReport {
        mode: TemporalMode::Progressive,
        confidence: 0.0,
        container_field_order: manifest
            .primary_video()
            .and_then(|v| v.field_order.clone()),
        container_says_interlaced: manifest
            .primary_video()
            .map(|v| v.declared_interlaced())
            .unwrap_or(false),
        totals: Default::default(),
        samples: Vec::new(),
        sampled_seconds: 0.0,
        notes: vec!["no cadence analysis was run".into()],
    };
    let scenes = SceneReport {
        shots: Vec::new(),
        frames_analyzed: 0,
        analysis_width: 0,
        analysis_height: 0,
        fps: manifest
            .primary_video()
            .and_then(|v| v.fps())
            .unwrap_or(Rational::from_i64(25)),
        timebase: Rational::from_i64(25),
        accurate_timestamps: false,
        notes: vec!["no shot analysis was run".into()],
    };
    build_plan(ff, manifest, &temporal, &scenes, None, request)
}

/// Timestamp helper for the plan's shot list rendering.
pub fn shot_time_label(t: Timestamp) -> String {
    t.format_hms()
}

/// The container this build writes: Matroska, because it carries multiple audio
/// tracks, ASS subtitles, fonts and chapters without argument.
pub fn output_extension(settings: &OutputSettings) -> &'static str {
    match settings.container.as_str() {
        "mp4" => "mp4",
        _ => "mkv",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_resolution_restores_sd_to_1080p() {
        assert_eq!(choose_target(720, 480, 4.0), (1620, 1080));
        assert_eq!(choose_target(720, 576, 4.0), (1350, 1080));
        assert_eq!(choose_target(1440, 1080, 4.0), (1440, 1080));
    }

    #[test]
    fn target_resolution_never_upscales_beyond_the_cap() {
        // A 2x cap on a 480p source: 960p, not 1080p.
        assert_eq!(choose_target(720, 480, 2.0), (1440, 960));
    }

    #[test]
    fn target_resolution_keeps_large_sources_untouched() {
        assert_eq!(choose_target(3840, 2160, 4.0), (3840, 2160));
        // odd widths are rounded up: encoders reject them
        let (w, h) = choose_target(853, 480, 4.0);
        assert_eq!(w % 2, 0);
        assert_eq!(h % 2, 0);
    }

    #[test]
    fn post_ivtc_rate_is_exactly_24_over_1001() {
        let ntsc = Rational::new(30000, 1001).unwrap();
        assert_eq!(post_ivtc_fps(ntsc), Rational::new(24000, 1001).unwrap());
        let pal = Rational::from_i64(25);
        assert_eq!(post_ivtc_fps(pal), Rational::from_i64(20));
    }

    fn fake_video(method: InterpolationMethod, target_fps: Rational) -> VideoPlan {
        VideoPlan {
            source: VideoSourceInfo {
                width: 720,
                height: 480,
                square_width: 720,
                square_height: 480,
                sar: Rational::ONE,
                fps: Rational::new(24000, 1001).unwrap(),
                duration_seconds: 60.0,
                frames: 1440,
                pix_fmt: Some("yuv420p".into()),
                is_hdr: false,
            },
            temporal_mode: TemporalMode::Progressive,
            temporal_note: String::new(),
            cadence: CadencePlan::plan(
                TemporalMode::Progressive,
                Rational::new(24000, 1001).unwrap(),
                true,
            ),
            ivtc: false,
            deinterlace: None,
            target_width: 1440,
            target_height: 960,
            upscale_factor: 2.0,
            interpolation: InterpolationPlan {
                enabled: method != InterpolationMethod::Off,
                method,
                source_fps: Rational::new(24000, 1001).unwrap(),
                target_fps,
                multiplier: 2,
                scene_cuts_respected: true,
                note: String::new(),
            },
            executor: VideoExecutor::FfmpegSinglePass,
            restoration: RestorationSettings {
                enabled: false,
                ..RestorationSettings::default()
            },
            regrain_strength: 0.0,
            regrain_per_shot: Vec::new(),
            filter_chain: String::new(),
            decode_chain: String::new(),
            encoder: SelectedVideoEncoder {
                name: "av1_nvenc".into(),
                codec: VideoCodec::Av1,
                vendor: Some("nvidia".into()),
                hardware: true,
                pix_fmt: "p010le".into(),
                quality_args: vec!["-cq".into(), "24".into()],
                input_args: vec![],
                requires_hwupload: false,
                note: String::new(),
            },
            encoder_chain: Vec::new(),
            pix_fmt: "p010le".into(),
            estimated_frames: 2880,
        }
    }

    #[test]
    fn the_single_pass_chain_scales_but_never_interpolates() {
        let video = fake_video(
            InterpolationMethod::Rife,
            Rational::new(48000, 1001).unwrap(),
        );
        let chain = build_filter_chain(&video, (720, 480));
        assert!(chain.contains("scale=1440:960:flags=lanczos"), "{chain}");
        // The whole point of the redesign: this function has no way to add frames,
        // so a plan that wants interpolation cannot reach it and be quietly served
        // something else.
        assert!(!chain.contains("framerate="), "{chain}");
        assert!(!chain.contains("minterpolate="), "{chain}");
    }

    #[test]
    fn a_request_for_rife_is_refused_rather_than_answered_with_minterpolate() {
        let profile = RestorationProfile::safe_16gb();
        let err = resolve_interpolation(&profile, Rational::new(24000, 1001).unwrap())
            .expect_err("the default profile asks for RIFE and no runtime can serve it");
        let text = err.to_string();
        assert!(text.contains("RIFE"), "{text}");
        assert!(
            !text.contains("minterpolate") && !text.contains("duplicat"),
            "the refusal must not offer a substitute algorithm: {text}"
        );
    }

    #[test]
    fn interpolation_off_resolves_to_a_plan_that_changes_no_frame_rate() {
        let mut profile = RestorationProfile::safe_16gb();
        profile.interpolation.method = InterpolationMethod::Off;
        let fps = Rational::new(24000, 1001).unwrap();
        let plan = resolve_interpolation(&profile, fps).expect("off is always available");
        assert!(!plan.enabled);
        assert_eq!(plan.multiplier, 1);
        assert_eq!(plan.target_fps, fps);
        assert_eq!(plan.method, InterpolationMethod::Off);
    }

    #[test]
    fn the_text_form_matches_what_serde_writes() {
        // `VideoExecutor` is both persisted in the plan and printed in the log.
        // When those two spellings drifted apart (`ffmpeg_single_pass` on disk,
        // `ffmpeg-single-pass` in the log) a test that hardcoded the readable one
        // failed against the stored JSON — a small bug with a confusing symptom,
        // and exactly the kind of thing that is cheaper to forbid than to debug.
        for value in [VideoExecutor::FfmpegSinglePass, VideoExecutor::Chunked] {
            let json = serde_json::to_string(&value).expect("a unit variant serialises");
            assert_eq!(
                json,
                format!("\"{}\"", value.as_str()),
                "{value:?} has two textual forms"
            );
            let round_tripped: VideoExecutor =
                serde_json::from_str(&json).expect("and reads back");
            assert_eq!(round_tripped, value);
        }
    }

    #[test]
    fn a_request_for_restoration_is_refused_rather_than_resampled() {
        let settings = RestorationProfile::safe_16gb().restoration;
        let err = require_restoration_runtime(&settings)
            .expect_err("the default profile asks for Real-ESRGAN and nothing can serve it");
        let text = err.to_string();
        assert!(text.contains("RealESRGAN_x4plus"), "{text}");
        assert!(
            !text.contains("Lanczos resample to the target"),
            "the refusal must not present a resample as the alternative: {text}"
        );

        // Turning it off is the supported configuration, and it must not error.
        let off = RestorationSettings {
            enabled: false,
            ..settings
        };
        assert!(require_restoration_runtime(&off).is_ok());
    }

    #[test]
    fn filter_chain_puts_deinterlace_first_and_regrain_last() {
        let mut video = fake_video(InterpolationMethod::Off, Rational::from_i64(25));
        video.interpolation.enabled = false;
        video.deinterlace = Some("bwdif=mode=send_frame".into());
        video.regrain_strength = 8.0;
        // source is 720x480, target 1440x960, so a scale step is present
        let chain = build_filter_chain(&video, (720, 480));
        assert!(chain.starts_with("bwdif="), "got {chain}");
        let regrain_at = chain.find("noise=").expect("regrain present");
        let scale_at = chain.find("scale=").expect("scale present");
        assert!(regrain_at > scale_at, "re-grain must follow resampling: {chain}");
        assert!(chain.contains("noise=alls=8:allf=t+u"));
    }

    #[test]
    fn no_scale_filter_when_the_target_matches_the_source() {
        let mut video = fake_video(InterpolationMethod::Off, Rational::from_i64(25));
        video.interpolation.enabled = false;
        video.target_width = 720;
        video.target_height = 480;
        let chain = build_filter_chain(&video, (720, 480));
        assert_eq!(chain, "", "nothing to do means an empty chain");
    }

    #[test]
    fn the_filter_chain_has_no_interpolation_stage_at_all() {
        // Superseded the old `minterpolate_chain_protects_scene_cuts`. The
        // cut-safety contract did not disappear with `minterpolate`; it moved to
        // the segment planner, which is where it belongs, because there it is
        // enforced by never handing the model a pair that straddles a cut rather
        // than by asking a filter to notice one. See
        // `pipeline::segments::a_dissolve_guard_is_held_while_the_cut_is_still_synthesised_up_to_it`.
        for method in [InterpolationMethod::Off, InterpolationMethod::Rife] {
            let video = fake_video(method, Rational::new(48000, 1001).unwrap());
            let chain = build_filter_chain(&video, (720, 480));
            assert!(
                !chain.contains("minterpolate") && !chain.contains("framerate"),
                "the single-pass chain must not be able to invent frames: {chain}"
            );
        }
    }

    #[test]
    fn hwupload_is_appended_for_vaapi_style_encoders() {
        let mut video = fake_video(InterpolationMethod::Off, Rational::from_i64(25));
        video.interpolation.enabled = false;
        video.encoder.requires_hwupload = true;
        video.encoder.pix_fmt = "nv12".into();
        video.pix_fmt = "nv12".into();
        let chain = build_filter_chain(&video, (720, 480));
        assert!(chain.ends_with("format=nv12,hwupload"), "got {chain}");
    }

    #[test]
    fn output_extension_follows_the_container() {
        let mut settings = OutputSettings::default();
        assert_eq!(output_extension(&settings), "mkv");
        settings.container = "mp4".into();
        assert_eq!(output_extension(&settings), "mp4");
    }
}
