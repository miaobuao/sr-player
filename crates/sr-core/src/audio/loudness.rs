//! The audio analysis pass: FFmpeg's `ebur128` as the authoritative meter, the
//! native BS.1770 meter as the cross-check, and the dialogue/ LDR decision that
//! the remaster pass will follow.

use super::dialogue::{
    dialogue_loudness, ldr_lu, DialogueTrack, SpeechDetector, SpeechDetectorOptions,
};
use super::dsp::LoudnessMeter;
use super::pcm::{DecodeOptions, PcmReader};
use super::rider::{decide, RemasterDecision, RiderSettings};
use crate::error::{Error, Result};
use crate::events::{Reporter, Stage, StageProgress};
use crate::ffmpeg::{args, run_tool, Ffmpeg, RunSpec};
use crate::media::manifest::MediaManifest;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Numbers parsed out of FFmpeg's `ebur128` summary.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Ebur128Report {
    pub integrated_lufs: Option<f64>,
    pub loudness_range_lu: Option<f64>,
    pub true_peak_dbtp: Option<f64>,
    pub lra_low_lufs: Option<f64>,
    pub lra_high_lufs: Option<f64>,
    pub threshold_lufs: Option<f64>,
}

impl Ebur128Report {
    pub fn summary(&self) -> String {
        format!(
            "I {} LUFS, LRA {} LU, true peak {} dBTP",
            self.integrated_lufs
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "?".into()),
            self.loudness_range_lu
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "?".into()),
            self.true_peak_dbtp
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "?".into())
        )
    }
}

/// Pulls the first number out of a segment like `   I:   -18.3 LUFS`.
fn number_after(line: &str, marker: &str) -> Option<f64> {
    let rest = line.split(marker).nth(1)?;
    let token = rest.split_whitespace().next()?;
    token.parse::<f64>().ok()
}

/// Parses an `ebur128` log. Last occurrence wins, which is correct because
/// FFmpeg prints its summary after the running series.
pub fn parse_ebur128_line(report: &mut Ebur128Report, line: &str) {
    if line.contains("LRA low:") {
        if let Some(v) = number_after(line, "LRA low:") {
            report.lra_low_lufs = Some(v);
        }
        return;
    }
    if line.contains("LRA high:") {
        if let Some(v) = number_after(line, "LRA high:") {
            report.lra_high_lufs = Some(v);
        }
        return;
    }
    if line.contains("LRA:") {
        if let Some(v) = number_after(line, "LRA:") {
            report.loudness_range_lu = Some(v);
        }
        return;
    }
    if line.contains("Threshold:") {
        if let Some(v) = number_after(line, "Threshold:") {
            report.threshold_lufs = Some(v);
        }
        return;
    }
    if line.contains("Peak:") {
        if let Some(v) = number_after(line, "Peak:") {
            report.true_peak_dbtp = Some(v);
        }
        return;
    }
    if line.contains("I:") && line.contains("LUFS") {
        if let Some(v) = number_after(line, "I:") {
            report.integrated_lufs = Some(v);
        }
    }
}

/// Runs `ebur128` over one audio stream. This is the number that goes into the
/// job report; the native meter only decides *whether* to act.
pub fn measure_ebur128(
    ff: &Ffmpeg,
    manifest: &MediaManifest,
    stream_index: usize,
    reporter: &Reporter,
    cancel: &AtomicBool,
) -> Result<Ebur128Report> {
    let report = Arc::new(Mutex::new(Ebur128Report::default()));
    let sink_report = Arc::clone(&report);
    if !ff.has_filter("ebur128") {
        reporter.warn(
            Some(Stage::AudioAnalysis),
            "this FFmpeg build has no ebur128 filter; relying on the native meter only",
        );
        return Ok(Ebur128Report::default());
    }

    let mut argv = args(&[
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-progress",
        "pipe:1",
    ]);
    if manifest
        .audio
        .get(stream_index)
        .map(|a| a.carries_drc_metadata)
        .unwrap_or(false)
    {
        // Do not let AC-3's own DRC change what we are about to measure.
        argv.extend(args(&["-drc_scale", "0"]));
    }
    argv.push("-i".into());
    argv.push(manifest.path.display().to_string());
    argv.push("-map".into());
    argv.push(format!("0:a:{stream_index}"));
    argv.extend(args(&[
        "-af",
        "ebur128=peak=true",
        "-vn",
        "-sn",
        "-dn",
        "-f",
        "null",
    ]));
    argv.push(if cfg!(windows) { "NUL".into() } else { "-".into() });

    let mut spec = RunSpec::new(Stage::AudioAnalysis, "ebur128")
        .quiet("Parsed_ebur128")
        .stderr_level(crate::events::Level::Debug)
        .with_sink(move |line: &str| {
            let mut guard = sink_report.lock().expect("ebur128 report");
            parse_ebur128_line(&mut guard, line);
        });
    if let Some(duration) = manifest.duration() {
        spec = spec.with_duration(duration.duration());
    }

    run_tool(&ff.ffmpeg, &argv, reporter, cancel, &spec)?;
    let parsed = report.lock().expect("ebur128 report").clone();
    reporter.info(
        Some(Stage::AudioAnalysis),
        format!("FFmpeg ebur128: {}", parsed.summary()),
    );
    Ok(parsed)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioAnalysis {
    pub stream_index: usize,
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_seconds: f64,
    /// Native BS.1770 measurement.
    pub programme_lufs: f64,
    /// FFmpeg's independent measurement of the same thing.
    pub programme_lufs_ffmpeg: Option<f64>,
    pub loudness_range_lu: f64,
    pub sample_peak_dbfs: f32,
    pub true_peak_dbtp: Option<f64>,
    pub dialogue_lufs: Option<f64>,
    pub ldr_lu: Option<f64>,
    pub speech_ratio: f32,
    pub dialogue_confidence: f32,
    pub decision: RemasterDecision,
    pub track: DialogueTrack,
    pub notes: Vec<String>,
}

impl AudioAnalysis {
    pub fn summary(&self) -> String {
        format!(
            "programme {:.1} LUFS, dialogue {}, LDR {} — {}",
            self.programme_lufs,
            self.dialogue_lufs
                .map(|d| format!("{d:.1} LUFS"))
                .unwrap_or_else(|| "not measurable".into()),
            self.ldr_lu
                .map(|l| format!("{l:.1} LU"))
                .unwrap_or_else(|| "n/a".into()),
            self.decision.summary()
        )
    }
}

#[derive(Clone, Debug)]
pub struct AnalyzeOptions {
    pub stream_index: usize,
    pub sample_rate: u32,
    pub dialogue: SpeechDetectorOptions,
    pub rider: RiderSettings,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        AnalyzeOptions {
            stream_index: 0,
            sample_rate: 48_000,
            dialogue: SpeechDetectorOptions::default(),
            rider: RiderSettings::default(),
        }
    }
}

/// Decodes the stream once, measuring programme and dialogue loudness on the
/// same 10 ms grid, then decides what the remaster pass may do.
pub fn analyze(
    ff: &Ffmpeg,
    manifest: &MediaManifest,
    opts: &AnalyzeOptions,
    reporter: &Reporter,
    cancel: &AtomicBool,
) -> Result<AudioAnalysis> {
    let stream = manifest.audio.get(opts.stream_index).ok_or_else(|| {
        Error::Unsupported(format!(
            "audio stream {} does not exist (the file has {})",
            opts.stream_index,
            manifest.audio.len()
        ))
    })?;
    let channels = stream.channels.unwrap_or(2).clamp(1, 16) as u16;
    let layout = stream.channel_layout.clone();
    let duration = manifest.duration().map(|d| d.duration());

    if stream.carries_drc_metadata {
        reporter.info(
            Some(Stage::AudioAnalysis),
            format!(
                "{} carries dynamic range control metadata; decoding with drc_scale=0 so the source's DRC is not applied twice",
                stream.base.codec_name.as_deref().unwrap_or("audio")
            ),
        );
    }

    // Authoritative measurement first (cheap: no PCM leaves FFmpeg).
    let ebur128 = measure_ebur128(ff, manifest, opts.stream_index, reporter, cancel)?;

    // Native pass: one decode feeding both meters.
    let decode = DecodeOptions {
        stream_index: opts.stream_index,
        sample_rate: opts.sample_rate,
        channels: Some(channels),
        disable_drc: true,
        expected_duration: duration,
    };
    let mut reader = PcmReader::open(ff, manifest.path.as_path(), &decode, reporter, cancel)?;
    reader.channels = channels;
    let mut meter = LoudnessMeter::new(opts.sample_rate, channels, layout.as_deref());
    let mut detector = SpeechDetector::new(opts.sample_rate, opts.dialogue.clone());

    let block_frames = (opts.sample_rate as usize / 10).max(1) * 10; // 100 ms
    let mut mono_scratch: Vec<f32> = Vec::with_capacity(block_frames);
    let mut total_frames: u64 = 0;
    let mut last_report = std::time::Instant::now();

    loop {
        let block = reader.next_block(block_frames, cancel)?;
        if block.is_empty() {
            break;
        }
        meter.push(&block);
        total_frames += (block.len() / channels as usize) as u64;

        // Dialogue analysis input: the centre channel on surround mixes (that is
        // where film dialogue lives), otherwise the downmix.
        mono_scratch.clear();
        if channels >= 3 {
            let centre = 2usize;
            mono_scratch.extend(
                block
                    .chunks_exact(channels as usize)
                    .map(|frame| frame[centre]),
            );
        } else {
            let ch = channels as usize;
            mono_scratch.extend(
                block
                    .chunks_exact(ch)
                    .map(|frame| frame.iter().sum::<f32>() / ch as f32),
            );
        }
        detector.push(&mono_scratch);

        if last_report.elapsed() >= Duration::from_millis(400) {
            last_report = std::time::Instant::now();
            reporter.progress(StageProgress {
                stage: Stage::AudioAnalysis,
                fraction: reader.progress_fraction(),
                detail: format!(
                    "{:.0}s analysed · {} speech",
                    total_frames as f64 / opts.sample_rate as f64,
                    detector.chunk_count()
                ),
                frames: None,
                fps: None,
                speed: None,
                out_time: None,
                eta: None,
            });
        }
    }
    meter.flush();
    detector.flush();
    reader.finish()?;

    let programme_lufs = meter.integrated_lufs();
    let block_loudness = meter.block_loudness();
    let track = detector.finish(programme_lufs, &block_loudness);
    let dialogue_lufs = dialogue_loudness(&block_loudness, &track.mask);
    let ldr = ldr_lu(programme_lufs, dialogue_lufs);
    let decision = decide(
        programme_lufs,
        dialogue_lufs,
        track.confidence,
        track.speech_ratio,
        &opts.rider,
    );

    let mut notes = track.notes.clone();
    if let Some(ffmpeg_lufs) = ebur128.integrated_lufs {
        let delta = (ffmpeg_lufs - programme_lufs).abs();
        if delta > 0.5 {
            notes.push(format!(
                "native loudness {programme_lufs:.1} LUFS disagrees with FFmpeg's {ffmpeg_lufs:.1} LUFS by {delta:.2} LU: investigate before trusting the LDR"
            ));
            reporter.warn(
                Some(Stage::AudioAnalysis),
                notes.last().cloned().unwrap_or_default(),
            );
        }
    } else {
        notes.push(
            "FFmpeg ebur128 produced no integrated value; using the native measurement only"
                .into(),
        );
    }
    if meter.non_finite_samples() > 0 {
        notes.push(format!(
            "{} non-finite samples were replaced with silence during measurement",
            meter.non_finite_samples()
        ));
    }

    let analysis = AudioAnalysis {
        stream_index: opts.stream_index,
        channels,
        sample_rate: opts.sample_rate,
        duration_seconds: total_frames as f64 / opts.sample_rate as f64,
        programme_lufs,
        programme_lufs_ffmpeg: ebur128.integrated_lufs,
        loudness_range_lu: meter.loudness_range_lu(),
        sample_peak_dbfs: meter.sample_peak_dbfs(),
        true_peak_dbtp: ebur128.true_peak_dbtp,
        dialogue_lufs,
        ldr_lu: ldr,
        speech_ratio: track.speech_ratio,
        dialogue_confidence: track.confidence,
        decision,
        track,
        notes,
    };

    reporter.info(Some(Stage::AudioAnalysis), analysis.summary());
    for note in &analysis.notes {
        reporter.debug(Some(Stage::AudioAnalysis), note.clone());
    }
    Ok(analysis)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUMMARY: &str = "\
[Parsed_ebur128_0 @ 0x1] t: 0.4 M: -30.0 S: -120.0 I: -30.0 LUFS LRA: 0.0 LU
[Parsed_ebur128_0 @ 0x1] t: 120.0 M: -18.0 S: -19.0 I: -18.3 LUFS LRA: 6.2 LU
[Parsed_ebur128_0 @ 0x1] Integrated loudness:
[Parsed_ebur128_0 @ 0x1]   I:         -18.3 LUFS
[Parsed_ebur128_0 @ 0x1]   Threshold: -28.3 LUFS
[Parsed_ebur128_0 @ 0x1] Loudness range:
[Parsed_ebur128_0 @ 0x1]   LRA:        12.4 LU
[Parsed_ebur128_0 @ 0x1]   Threshold:  -38.3 LUFS
[Parsed_ebur128_0 @ 0x1]   LRA low:   -24.0 LUFS
[Parsed_ebur128_0 @ 0x1]   LRA high:  -11.6 LUFS
[Parsed_ebur128_0 @ 0x1] True peak:
[Parsed_ebur128_0 @ 0x1]   Peak:       -2.60 dBFS
";

    #[test]
    fn parses_the_ebur128_summary_and_prefers_it_over_the_running_series() {
        let mut report = Ebur128Report::default();
        for line in SUMMARY.lines() {
            parse_ebur128_line(&mut report, line);
        }
        assert_eq!(report.integrated_lufs, Some(-18.3));
        assert_eq!(report.loudness_range_lu, Some(12.4));
        assert_eq!(report.lra_low_lufs, Some(-24.0));
        assert_eq!(report.lra_high_lufs, Some(-11.6));
        assert_eq!(report.true_peak_dbtp, Some(-2.60));
        assert_eq!(report.threshold_lufs, Some(-38.3));
    }

    #[test]
    fn lra_low_and_high_do_not_clobber_the_range() {
        let mut report = Ebur128Report::default();
        parse_ebur128_line(&mut report, "  LRA:        12.4 LU");
        parse_ebur128_line(&mut report, "  LRA low:   -24.0 LUFS");
        parse_ebur128_line(&mut report, "  LRA high:  -11.6 LUFS");
        assert_eq!(report.loudness_range_lu, Some(12.4));
        assert_eq!(report.lra_low_lufs, Some(-24.0));
        assert_eq!(report.lra_high_lufs, Some(-11.6));
    }

    #[test]
    fn unrelated_lines_are_ignored() {
        let mut report = Ebur128Report::default();
        parse_ebur128_line(&mut report, "frame= 120 fps=0.0 q=-0.0 size=N/A");
        parse_ebur128_line(&mut report, "[Parsed_ebur128_0 @ 0x1] Summary:");
        assert_eq!(report.integrated_lufs, None);
        assert_eq!(report.true_peak_dbtp, None);
    }

    #[test]
    fn summary_string_is_human_readable_even_with_missing_values() {
        let report = Ebur128Report::default();
        let text = report.summary();
        assert!(text.contains("I ? LUFS"));
        assert!(text.contains("true peak ? dBTP"));
    }
}
