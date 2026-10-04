//! Audio: measurement, dialogue analysis and the remaster pass.
//!
//! Division of labour, which is the whole point of this module:
//!
//! | question | answered by | cost |
//! |---|---|---|
//! | how loud is the programme? | FFmpeg `ebur128` (authoritative) + native BS.1770 (cross-check) | one decode, no PCM copies |
//! | how loud is the dialogue? | native band detector + gated loudness over speech-active blocks | shared 10 ms grid |
//! | should anything be done? | EBU R128 S4 decision ([`rider::decide`]) | free |
//! | do it | native rider + mid-band duck, streamed to a float WAV | one re-decode |
//!
//! Everything is streamed: a two hour 5.1 feature never exists in RAM.

pub mod dialogue;
pub mod dsp;
pub mod loudness;
pub mod pcm;
pub mod rider;

use crate::error::{Error, Result};
use crate::events::{Reporter, Stage, StageProgress};
use crate::ffmpeg::Ffmpeg;
use crate::media::manifest::MediaManifest;
use pcm::{DecodeOptions, PcmReader, WavWriter};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

pub use dialogue::{
    dialogue_loudness, ldr_lu, DialogueTrack, SpeechDetector, SpeechDetectorOptions,
};
pub use dsp::{channel_weights, gated_loudness, LoudnessMeter};
pub use loudness::{analyze, AnalyzeOptions, AudioAnalysis, Ebur128Report};
pub use pcm::{DecodeOptions as PcmDecodeOptions, PcmBuffer, WavWriter as FloatWavWriter};
pub use rider::{decide, GainCurve, RemasterDecision, RemasterProcessor, RiderSettings};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemasterStats {
    pub path: PathBuf,
    pub frames: u64,
    pub duration_seconds: f64,
    /// False when the decision was "leave the dynamics alone" and the pass was a
    /// straight copy.
    pub applied: bool,
    pub max_gain_db: f32,
    pub max_duck_db: f32,
}

/// Renders the (optionally remastered) audio to a 32-bit float WAV.
///
/// A WAV intermediate is deliberate: the dialogue rider is stateful DSP that
/// cannot be expressed as an FFmpeg filter chain, and float WAV means the
/// subsequent loudness normalisation and true-peak limiting see exactly what we
/// measured, with no second lossy generation.
pub fn remaster_to_wav(
    ff: &Ffmpeg,
    manifest: &MediaManifest,
    analysis: &AudioAnalysis,
    settings: &RiderSettings,
    out_path: &Path,
    reporter: &Reporter,
    cancel: &AtomicBool,
) -> Result<RemasterStats> {
    let stream_index = analysis.stream_index;
    let channels = analysis.channels.max(1);
    let rate = analysis.sample_rate.max(8_000);
    let expected = manifest.duration().map(|d| d.duration());

    let decode = DecodeOptions {
        stream_index,
        sample_rate: rate,
        channels: Some(channels),
        disable_drc: true,
        expected_duration: expected,
    };

    let mut reader = PcmReader::open(ff, manifest.path.as_path(), &decode, reporter, cancel)?;
    reader.channels = channels;
    let mut writer = WavWriter::create(out_path, rate, channels)?;
    let layout = manifest
        .audio
        .get(stream_index)
        .and_then(|stream| stream.channel_layout.clone());
    let mut processor = if analysis.decision.apply {
        let curve = GainCurve::from_track(&analysis.track, &analysis.decision, settings);
        let processor = RemasterProcessor::new(curve, channels, rate, settings, layout.as_deref());
        // What is done to *which* channel is part of the result, not an
        // implementation detail: a dialogue gain applied to the whole mix is a
        // different (and wrong) operation that happens to share its name.
        reporter.info(
            Some(Stage::AudioProcess),
            format!("dialogue rider: {}", processor.channel_plan().describe()),
        );
        if processor.channel_plan().is_inert() {
            reporter.warn(
                Some(Stage::AudioProcess),
                "no centre channel was identified, so the dialogue rider is leaving the mix \
                 alone rather than lifting every channel and calling it a dialogue boost",
            );
        }
        Some(processor)
    } else {
        None
    };

    // 100 ms blocks: plenty of resolution for a gain curve on a 10 ms grid, and
    // small enough that the WAV writer never buffers meaningfully.
    let block_frames = (rate as usize / 10).max(1);
    let mut frames_written: u64 = 0;
    let mut last_report = std::time::Instant::now();

    loop {
        let mut block = reader.next_block(block_frames, cancel)?;
        if block.is_empty() {
            break;
        }
        if let Some(processor) = processor.as_mut() {
            processor.process_block(&mut block);
        }
        writer.write(&block)?;
        frames_written += (block.len() / channels as usize) as u64;

        if last_report.elapsed() >= Duration::from_millis(400) {
            last_report = std::time::Instant::now();
            reporter.progress(StageProgress {
                stage: Stage::AudioProcess,
                fraction: reader.progress_fraction(),
                detail: format!(
                    "{:.0}s written{}",
                    frames_written as f64 / rate as f64,
                    if analysis.decision.apply {
                        " (remastered)"
                    } else {
                        " (unchanged dynamics)"
                    }
                ),
                frames: Some(frames_written),
                fps: None,
                speed: None,
                out_time: None,
                eta: None,
            });
        }
    }
    reader.finish()?;
    let applied = processor.is_some();
    let (max_gain_db, max_duck_db) = if applied {
        (
            analysis.decision.dialogue_gain_db,
            analysis.decision.duck_db,
        )
    } else {
        (0.0, 0.0)
    };
    writer.finish()?;

    if frames_written == 0 {
        return Err(Error::Stage {
            stage: Stage::AudioProcess.id().into(),
            detail: "the audio decode produced no samples".into(),
        });
    }

    Ok(RemasterStats {
        path: out_path.to_path_buf(),
        frames: frames_written,
        duration_seconds: frames_written as f64 / rate as f64,
        applied,
        max_gain_db,
        max_duck_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaster_stats_reports_what_happened() {
        let stats = RemasterStats {
            path: PathBuf::from("out.wav"),
            frames: 48_000,
            duration_seconds: 1.0,
            applied: false,
            max_gain_db: 0.0,
            max_duck_db: 0.0,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let back: RemasterStats = serde_json::from_str(&json).unwrap();
        assert_eq!(back.frames, 48_000);
        assert!(!back.applied);
    }
}
