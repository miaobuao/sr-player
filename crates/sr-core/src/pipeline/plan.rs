//! Turning analysis into a plan a human can read.
//!
//! A plan is not just an argument list: every decision carries the reason it was
//! made, so the UI can show *why* a 29.97 DVD is being field-matched rather than
//! deinterlaced, why interpolation was disabled for a shot, or why the audio was
//! left completely alone. "Zero tuning" only works if the reasoning is visible.

use crate::audio::loudness::AudioAnalysis;
use crate::error::Result;
use crate::ffmpeg::{EncoderPreference, Ffmpeg, SelectedAudioEncoder, SelectedVideoEncoder, VideoCodec};
use crate::infer::{EngineKind, EngineRegistry, InferenceTask};
use crate::media::classify::{TemporalMode, TemporalReport};
use crate::media::manifest::{MediaManifest, PreservationInventory};
use crate::media::scene::SceneReport;
use crate::pipeline::policy::{plan_vram, VramBudget, WorkingSet};
use crate::pipeline::profile::{
    InterpolationMethod, LoudnessTarget, OutputSettings, RestorationProfile,
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
    pub engine: Option<String>,
    /// Always true in this build: cuts are never crossed.
    pub scene_cuts_respected: bool,
    pub note: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoPlan {
    pub source: VideoSourceInfo,
    pub temporal_mode: TemporalMode,
    pub temporal_note: String,
    pub ivtc: bool,
    pub deinterlace: Option<String>,
    pub target_width: u32,
    pub target_height: u32,
    pub upscale_factor: f32,
    pub interpolation: InterpolationPlan,
    pub regrain_strength: f32,
    pub filter_chain: String,
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
            "{}x{} @ {:.3} fps, {}, {}, {}",
            self.video.target_width,
            self.video.target_height,
            self.video.interpolation.target_fps.to_f64(),
            self.video.encoder.describe(),
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

/// Frame rate after inverse telecine: 3:2 pulldown collapses to 4/5 of the rate.
pub fn post_ivtc_fps(fps: Rational) -> Rational {
    fps.checked_mul(&Rational::new(4, 5).unwrap_or(Rational::ONE))
        .unwrap_or(fps)
}

/// Builds the `-vf` chain. Order matters and is fixed.
pub fn build_filter_chain(video: &VideoPlan, source_square: (u32, u32)) -> String {
    let mut filters: Vec<String> = Vec::new();

    if let Some(deinterlace) = &video.deinterlace {
        filters.push(deinterlace.clone());
    }

    let needs_scale = (video.target_width, video.target_height) != source_square;
    if needs_scale {
        filters.push(format!(
            "scale={}:{}:flags=lanczos",
            video.target_width, video.target_height
        ));
    }

    let interpolation = &video.interpolation;
    if interpolation.enabled && interpolation.multiplier > 1 {
        let fps = format!("{:.6}", interpolation.target_fps.to_f64());
        match interpolation.method {
            InterpolationMethod::Duplicate => filters.push(format!("framerate=fps={fps}")),
            InterpolationMethod::Minterpolate | InterpolationMethod::Plugin => {
                // `scd=fdiff` keeps the interpolator from crossing a cut, which
                // is the one artefact that is impossible to miss.
                filters.push(format!(
                    "minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=10"
                ));
            }
            InterpolationMethod::Off => {}
        }
    }

    if video.regrain_strength > 0.0 {
        filters.push(format!(
            "noise=alls={:.0}:allf=t+u",
            video.regrain_strength.clamp(1.0, 30.0)
        ));
    }

    if video.encoder.requires_hwupload {
        filters.push(format!("format={},hwupload", video.encoder.pix_fmt));
    }

    filters.join(",")
}

/// Chooses the interpolation engine actually used, downgrading honestly when the
/// requested one is not installed.
fn resolve_interpolation(
    profile: &RestorationProfile,
    engines: &EngineRegistry,
    ff: &Ffmpeg,
    source_fps: Rational,
    warnings: &mut Vec<String>,
) -> InterpolationPlan {
    let multiplier = profile.interpolation.multiplier.max(1);
    let requested = profile.interpolation.method;
    if requested == InterpolationMethod::Off || multiplier <= 1 {
        return InterpolationPlan {
            enabled: false,
            method: InterpolationMethod::Off,
            source_fps,
            target_fps: source_fps,
            multiplier: 1,
            engine: None,
            scene_cuts_respected: true,
            note: "interpolation disabled: source cadence preserved".into(),
        };
    }

    let post_ivtc = source_fps;
    let target_fps = post_ivtc
        .checked_mul(&Rational::from_i64(multiplier as i64))
        .unwrap_or(post_ivtc);

    let (method, engine, note) = match requested {
        InterpolationMethod::Plugin => match engines.select(InferenceTask::Interpolate) {
            Some(engine) if engine.kind() == EngineKind::Plugin => (
                InterpolationMethod::Plugin,
                Some(engine.id().to_string()),
                "model interpolation via the installed plugin".to_string(),
            ),
            _ => {
                warnings.push(
                    "no interpolation plugin installed: falling back to frame duplication \
                     (no motion is synthesised)"
                        .into(),
                );
                (
                    InterpolationMethod::Duplicate,
                    None,
                    "frame duplication (no model backend available)".to_string(),
                )
            }
        },
        InterpolationMethod::Minterpolate => {
            if ff.has_filter("minterpolate") {
                (
                    InterpolationMethod::Minterpolate,
                    Some("ffmpeg-minterpolate".to_string()),
                    "motion-compensated interpolation in software; expect a large slowdown".into(),
                )
            } else {
                warnings.push("minterpolate filter missing: using frame duplication".into());
                (
                    InterpolationMethod::Duplicate,
                    None,
                    "frame duplication (minterpolate unavailable)".to_string(),
                )
            }
        }
        InterpolationMethod::Duplicate | InterpolationMethod::Off => (
            InterpolationMethod::Duplicate,
            None,
            "frame duplication: deterministic, invents no motion".to_string(),
        ),
    };

    InterpolationPlan {
        enabled: true,
        method,
        source_fps: post_ivtc,
        target_fps,
        multiplier,
        engine,
        scene_cuts_respected: profile.interpolation.scene_cut_protection,
        note: format!(
            "{note}; {}x {:.3} -> {:.3} fps",
            multiplier,
            post_ivtc.to_f64(),
            target_fps.to_f64()
        ),
    }
}

/// The whole decision, in one place.
pub fn build_plan(
    ff: &Ffmpeg,
    engines: &EngineRegistry,
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
        crate::error::Error::Unsupported("the file has no video stream to convert".into())
    })?;
    let fps = video_stream.fps().ok_or_else(|| {
        crate::error::Error::Unsupported("the video frame rate is unknown".into())
    })?;
    let (raw_w, raw_h) = video_stream.size().unwrap_or((0, 0));
    let (square_w, square_h) = video_stream.square_pixel_size().unwrap_or((raw_w, raw_h));
    let duration_seconds = manifest.duration_seconds();

    // --- temporal -----------------------------------------------------------
    let ivtc = temporal.mode.needs_ivtc();
    let deinterlace = match temporal.mode {
        TemporalMode::Interlaced => Some(
            "bwdif=mode=send_frame:parity=auto:deint=all".to_string(),
        ),
        TemporalMode::Mixed => Some(
            "bwdif=mode=send_frame:parity=auto:deint=interlaced".to_string(),
        ),
        _ => None,
    };
    let temporal_filter = if ivtc {
        // Field match first, then remove the duplicated frames. A blanket
        // deinterlace here would throw away half the temporal information.
        if !temporal.mode.is_confident() {
            warnings.push("cadence classifier was not confident; IVTC applied conservatively".into());
        }
        Some("fieldmatch=order=auto:combmatch=full,decimate".to_string())
    } else {
        None
    };
    let effective_fps = if ivtc { post_ivtc_fps(fps) } else { fps };
    let temporal_note = format!(
        "{} — {}{}",
        temporal.mode.as_str(),
        temporal.summary(),
        if ivtc {
            format!("; IVTC to {:.3} fps", effective_fps.to_f64())
        } else if temporal.mode.needs_deinterlace() {
            "; motion-adaptive deinterlace (no frame dropping)".to_string()
        } else {
            String::new()
        }
    );

    // --- geometry -----------------------------------------------------------
    let (target_width, target_height) =
        choose_target(square_w.max(2) as u64, square_h.max(2) as u64, profile.restoration.max_upscale);
    let upscale_factor = target_height as f32 / square_h.max(1) as f32;

    // --- interpolation ------------------------------------------------------
    let interpolation =
        resolve_interpolation(profile, engines, ff, effective_fps, &mut warnings);
    if interpolation.enabled && interpolation.method != InterpolationMethod::Off {
        notes.push(format!(
            "interpolation: {} across {} cuts is disabled by design",
            interpolation.note, scenes.cut_count()
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
        ivtc,
        deinterlace: deinterlace.or(temporal_filter),
        target_width,
        target_height,
        upscale_factor,
        interpolation,
        regrain_strength: profile.output.regrain_strength,
        filter_chain: String::new(),
        encoder,
        encoder_chain,
        pix_fmt: String::new(),
        estimated_frames,
    };
    video.pix_fmt = video.encoder.pix_fmt.clone();
    video.filter_chain = build_filter_chain(&video, (square_w as u32, square_h as u32));

    if video.source.is_hdr {
        warnings.push(
            "HDR source: the SDR restoration path is bypassed. Tone mapping is deliberately not \
             applied automatically, because it would irreversibly change the master."
                .into(),
        );
    }
    if upscale_factor > 2.5 {
        warnings.push(format!(
            "upscaling by {upscale_factor:.2}x invents no detail: the deterministic path only \
             resamples. A restoration model is required for real detail."
        ));
    }
    if profile.restoration.enabled && !engines.has_plugin() {
        warnings.push(
            "the profile asks for model restoration but no inference plugin is installed: \
             running the deterministic path instead"
                .into(),
        );
    }
    if !profile.restoration.enabled {
        notes.push(
            "restoration: disabled — no model backend, so no detail is invented (this is the \
             honest default)"
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
    engines: &EngineRegistry,
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
    build_plan(ff, engines, manifest, &temporal, &scenes, None, request)
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
            ivtc: false,
            deinterlace: None,
            target_width: 1440,
            target_height: 960,
            upscale_factor: 2.0,
            interpolation: InterpolationPlan {
                enabled: true,
                method,
                source_fps: Rational::new(24000, 1001).unwrap(),
                target_fps,
                multiplier: 2,
                engine: None,
                scene_cuts_respected: true,
                note: String::new(),
            },
            regrain_strength: 0.0,
            filter_chain: String::new(),
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
    fn filter_chain_scales_then_interpolates_in_order() {
        let video = fake_video(
            InterpolationMethod::Duplicate,
            Rational::new(48000, 1001).unwrap(),
        );
        let chain = build_filter_chain(&video, (720, 480));
        let scale_at = chain.find("scale=").expect("scale present");
        let interp_at = chain.find("framerate=").expect("interpolation present");
        assert!(scale_at < interp_at, "scaling must precede interpolation: {chain}");
        assert!(chain.contains("scale=1440:960:flags=lanczos"));
        assert!(chain.contains("framerate=fps=47.952048"));
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
    fn minterpolate_chain_protects_scene_cuts() {
        let video = fake_video(
            InterpolationMethod::Minterpolate,
            Rational::new(48000, 1001).unwrap(),
        );
        let chain = build_filter_chain(&video, (720, 480));
        assert!(chain.contains("minterpolate="));
        assert!(
            chain.contains("scd=fdiff"),
            "scene change detection must be on: {chain}"
        );
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
