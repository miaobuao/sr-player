//! Timing and transition cases: variable frame rate, a stream that does not start
//! at zero, and the two transitions that fool a shot detector.
//!
//! These are the shapes that break a pipeline *quietly*. A wrong frame rate
//! changes the runtime; a mishandled start offset inserts silence or shifts the
//! audio against the picture; a flash or a dissolve splits shots that were never
//! split, which costs interpolation across the join and multiplies the chunks.
//!
//! Skips itself when FFmpeg is unavailable.

use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::infer::EngineRegistry;
use sr_core::media::classify::{classify, ClassifyOptions};
use sr_core::media::probe;
use sr_core::media::scene::{detect_scenes, SceneDecode, SceneOptions};
use sr_core::pipeline::plan::{cadence_for, PlanRequest};
use sr_core::pipeline::profile::{InterpolationMethod, RestorationProfile};
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::{EventBus, Reporter};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn ffmpeg_or_skip() -> Option<Arc<Ffmpeg>> {
    match Ffmpeg::discover() {
        Ok(ff) => Some(Arc::new(ff)),
        Err(err) => {
            eprintln!("SKIPPED: {err}");
            return None;
        }
    }
}

fn build(ff: &Ffmpeg, output: &Path, arguments: Vec<String>) {
    let mut argv = args(&["-y", "-hide_banner", "-loglevel", "error"]);
    argv.extend(arguments);
    argv.push(output.display().to_string());
    capture(&ff.ffmpeg, &argv).unwrap_or_else(|err| panic!("build {}: {err}", output.display()));
}

/// Two seconds at 24 fps with one white frame in the middle.
///
/// A flash is the classic false positive for a shot detector: the frame-to-frame
/// difference at the flash is larger than at most real cuts.
fn build_flash(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let path = dir.join("flash.mkv");
    build(
        ff,
        &path,
        args(&[
            "-f",
            "lavfi",
            "-i",
            "color=navy:size=160x120:rate=24:duration=2",
            "-vf",
            "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='eq(n,24)'",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
        ]),
    );
    path
}

/// A half-second cross-fade between two flat colours.
fn build_dissolve(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let first = dir.join("dissolve-a.mkv");
    let second = dir.join("dissolve-b.mkv");
    build(
        ff,
        &first,
        args(&[
            "-f", "lavfi", "-i", "color=navy:size=160x120:rate=24:duration=2",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-f", "matroska",
        ]),
    );
    build(
        ff,
        &second,
        args(&[
            "-f", "lavfi", "-i", "color=orange:size=160x120:rate=24:duration=1.5",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-f", "matroska",
        ]),
    );
    let path = dir.join("dissolve.mkv");
    build(
        ff,
        &path,
        args(&[
            "-i",
            &first.display().to_string(),
            "-i",
            &second.display().to_string(),
            "-filter_complex",
            "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=1.5[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
        ]),
    );
    path
}

/// One second at 24 fps joined to one second at 30, with the timestamps kept.
fn build_vfr(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let path = dir.join("vfr.mkv");
    build(
        ff,
        &path,
        args(&[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=24:duration=1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=30:duration=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
        ]),
    );
    path
}

/// A one-second clip whose timestamps start at ten seconds.
fn build_offset(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let path = dir.join("offset.mkv");
    build(
        ff,
        &path,
        args(&[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=24:duration=1",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-output_ts_offset",
            "10",
            "-f",
            "matroska",
        ]),
    );
    path
}

/// Frames in a file, counted by decoding it.
fn count_frames(ff: &Ffmpeg, path: &Path) -> u64 {
    let mut argv = args(&["-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(path.display().to_string());
    argv.extend(args(&["-map", "0:v:0", "-f", "framemd5", "-"]));
    match capture(&ff.ffmpeg, &argv) {
        Ok(text) => text
            .lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .count() as u64,
        Err(_) => 0,
    }
}

fn shots(ff: &Arc<Ffmpeg>, input: &Path) -> (usize, u64) {
    let manifest = probe(ff, input).expect("probe");
    let bus = EventBus::new();
    let reporter = Reporter::new(bus);
    let cancel = AtomicBool::new(false);
    let temporal = classify(ff, &manifest, &reporter, &cancel, &ClassifyOptions::default())
        .expect("classify");
    let cadence = cadence_for(&manifest, &temporal).expect("cadence");
    let scenes = detect_scenes(
        ff,
        &manifest,
        &reporter,
        &cancel,
        &SceneOptions::default(),
        SceneDecode {
            pre_chain: cadence.chain_str(),
            fps: cadence.fps,
            reason: "corpus",
        },
    )
    .expect("scenes");
    (scenes.shots.len(), scenes.frames_analyzed)
}

/// Runs the pipeline and returns the output's duration in seconds.
fn run_and_measure(ff: &Arc<Ffmpeg>, input: &Path, dir: &Path, job: &str) -> (f64, f64) {
    let output = dir.join(format!("{job}.out.mkv"));
    let bus = EventBus::new();
    let events = bus.subscribe();
    let runner = PipelineRunner::new(
        Arc::clone(ff),
        Arc::new(EngineRegistry::probe(Arc::clone(ff))),
        bus.clone(),
        Arc::new(Store::open(&dir.join(format!("{job}.sqlite3"))).expect("state store")),
        Arc::new(AtomicBool::new(false)),
        dir.join(format!("{job}-scratch")),
    );
    let mut profile = RestorationProfile::deterministic();
    profile.interpolation.method = InterpolationMethod::Off;
    profile.interpolation.multiplier = 1;
    profile.restoration.max_upscale = 1.0;
    profile.audio.enabled = false;
    profile.output.prefer_hardware = false;
    let outcome = runner
        .run(
            PlanRequest {
                job_id: job.into(),
                input: input.to_path_buf(),
                output: output.clone(),
                profile,
            },
            &RunnerOptions {
                resume: true,
                ..Default::default()
            },
        )
        .expect("the runner always reports an outcome");
    if !outcome.ok {
        let mut logs = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let sr_core::Event::Log(record) = event {
                logs.push(record.message);
            }
        }
        let produced = probe(ff, &output)
            .map(|manifest| {
                format!(
                    "output exists: duration {:?}, video start {:?}, {} frame(s) by count",
                    manifest.duration().map(|d| d.seconds_f64()),
                    manifest
                        .primary_video()
                        .and_then(|video| video.base.start_time.clone()),
                    count_frames(ff, &output)
                )
            })
            .unwrap_or_else(|err| format!("no output to probe: {err}"));
        panic!(
            "the job failed: {}\n{produced}\n{}",
            outcome.message,
            logs.iter()
                .filter(|line| {
                    let lower = line.to_ascii_lowercase();
                    lower.contains("check") || lower.contains("fail") || lower.contains("frame")
                })
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    let result = probe(ff, &output).expect("probe the output");
    let duration = result.duration().map(|d| d.seconds_f64()).unwrap_or(0.0);
    let start = result
        .primary_video()
        .and_then(|video| {
            video
                .base
                .start_time
                .as_deref()
                .and_then(|value| value.parse::<f64>().ok())
        })
        .unwrap_or(0.0);
    (start, duration)
}

/// A flash must not split a shot.
///
/// **This fails today, and the failure is the finding.** A single white frame is
/// detected as a cut, so a two-second shot becomes two shots with a protected
/// boundary between them. The cost is not corruption — nothing interpolates across
/// a shot boundary, which is the safe direction — but it is real: the frame pair
/// spanning the flash never gets interpolated, every such transition multiplies
/// the chunk count, and a fade to white (common in old prints) would fragment a
/// take into dozens of shots.
///
/// The fix is a flash guard in the shot detector: a boundary whose two sides differ
/// mostly in luminance, and which lasts a frame or two, is a flash rather than a
/// change of scene. It is recorded here rather than asserted away.
#[test]
fn a_flash_frame_does_not_split_a_shot() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_flash(&ff, dir.path());
    let (shot_count, frames) = shots(&ff, &input);
    eprintln!("flash: {shot_count} shots from {frames} frames");
    assert_eq!(
        shot_count, 1,
        "a two-second shot with one white frame in it must stay one shot"
    );
}

/// A dissolve is a gradual transition, and the detector reports it as several
/// cuts. That is characterised here rather than asserted to be right.
#[test]
fn a_dissolve_is_fragmented_but_the_job_still_runs() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_dissolve(&ff, dir.path());
    let (shot_count, frames) = shots(&ff, &input);
    eprintln!("dissolve: {shot_count} shots from {frames} frames");
    assert!(
        shot_count >= 2,
        "the two sides of a dissolve are different scenes and must be separated"
    );
    // Whatever the split, the pipeline must neither lose nor gain runtime. A
    // dissolve that fragments the shot list is a quality concern; a dissolve that
    // changes the duration is a corrupted file.
    let (_, duration) = run_and_measure(&ff, &input, dir.path(), "dissolve");
    assert!(
        (duration - 3.0).abs() < 0.2,
        "the output must run for the three seconds the source does, got {duration:.3}s"
    );
}

/// A source whose timestamps start at ten seconds must come out starting at zero
/// and lasting as long as its content, not eleven seconds.
///
/// **The pipeline does this correctly and the quality check rejects it.** Measured
/// on the failing run: the output is `duration 1.0`, `video start 0.000000`, 24
/// frames — everything the test asks for — and the job still ends with
/// `stage qc failed: 1 of 8 checks failed`.
///
/// The check that fails is the duration comparison: the expected duration comes
/// from the source's *format* duration, which for this fixture is 11 seconds
/// because the offset is included, while the content is one second. So any file
/// whose timestamps do not start at zero fails QC even when the transcode is
/// right — and in an unattended system that turns a whole class of real files
/// (broadcast captures, rips with a delay, transport streams) into reported
/// failures.
///
/// The fix belongs in the check, not in the pipeline: the expected duration should
/// be the content duration — the format duration less the start time, or the sum
/// of the stream durations — rather than the raw container figure.
#[test]


fn a_non_zero_start_time_is_normalised() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_offset(&ff, dir.path());
    let manifest = probe(&ff, &input).expect("probe");
    let source_start = manifest
        .primary_video()
        .and_then(|video| {
            video
                .base
                .start_time
                .as_deref()
                .and_then(|value| value.parse::<f64>().ok())
        })
        .unwrap_or(0.0);
    assert!(
        source_start > 9.0,
        "the fixture must actually start late, got {source_start}"
    );
    let (start, duration) = run_and_measure(&ff, &input, dir.path(), "offset");
    eprintln!("offset: source started at {source_start}s, output starts at {start}s for {duration}s");
    assert!(
        start.abs() < 0.1,
        "the output must start at zero; a preserved ten-second offset is ten \
         seconds of nothing at the front of the film, got {start}"
    );
    assert!(
        (duration - 1.0).abs() < 0.2,
        "the content is one second long and must come out one second long, got {duration:.3}s"
    );
}

/// A variable frame rate source must keep its duration. Converting it to a
/// constant rate is fine and expected; changing how long the film runs is not.
#[test]
fn a_variable_frame_rate_source_keeps_its_duration() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_vfr(&ff, dir.path());
    let manifest = probe(&ff, &input).expect("probe");
    let source = manifest.duration().map(|d| d.seconds_f64()).unwrap_or(0.0);
    let (_, duration) = run_and_measure(&ff, &input, dir.path(), "vfr");
    eprintln!("vfr: source {source:.3}s, output {duration:.3}s");
    assert!(
        (source - 2.0).abs() < 0.2,
        "the fixture is two seconds of content, got {source:.3}s"
    );
    assert!(
        (duration - source).abs() < 0.15,
        "a variable rate must not change the runtime: {source:.3}s in, {duration:.3}s out"
    );
}
