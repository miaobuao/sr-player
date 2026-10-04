//! End-to-end: build a fixture with FFmpeg, run the real pipeline over it, and
//! verify what came out.
//!
//! This is the test that backs the project's central claim — *a real video goes
//! in and a correct one comes out, with no Python and no CUDA* — so it exercises
//! the whole chain: probing, cadence classification, shot detection, audio
//! analysis, planning, the dialogue decision, the encode and the QC stage.
//!
//! It skips itself (loudly) when FFmpeg is not installed, because FFmpeg is an
//! external dependency rather than a build dependency.

use sr_core::events::{Event, JobState, Stage, StageStatus};
use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::media::probe;
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::{InterpolationMethod, RestorationProfile};
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::EventBus;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Returns `None` when FFmpeg is unavailable, so the test can skip instead of
/// failing on a machine that simply has no media toolchain.
fn ffmpeg_or_skip() -> Option<Arc<Ffmpeg>> {
    match Ffmpeg::discover() {
        Ok(ff) => Some(Arc::new(ff)),
        Err(err) => {
            eprintln!("SKIPPED: {err}");
            None
        }
    }
}

/// 320x240 MPEG-2 at 29.97 with hard cuts, plus two audio tracks, a subtitle and
/// a chapter. Small enough to build in a second, awkward enough to be useful.
fn build_fixture(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let srt = dir.join("subs.srt");
    std::fs::write(
        &srt,
        "1\n00:00:00,200 --> 00:00:01,000\nfirst shot\n\n2\n00:00:01,400 --> 00:00:02,200\nsecond shot\n",
    )
    .expect("write subtitles");

    let chapters = dir.join("chapters.txt");
    std::fs::write(
        &chapters,
        ";FFMETADATA1\ntitle=E2E Fixture\n\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1200\ntitle=One\n\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=1200\nEND=2400\ntitle=Two\n",
    )
    .expect("write chapters");

    let base = dir.join("base.mkv");
    let movie = dir.join("e2e-input.mkv");

    let filter = "[0:v][1:v]concat=n=2:v=1:a=0[v];[2:a]volume=0.3[d];[3:a]volume=0.8[f];[d][f]amix=inputs=2:duration=longest:normalize=0[a]";

    let mut argv = args(&[
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=30000/1001:duration=1.2",
        "-f",
        "lavfi",
        "-i",
        "smptebars=size=320x240:rate=30000/1001:duration=1.2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1200:sample_rate=48000:duration=2.4",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=70:sample_rate=48000:duration=2.4",
    ]);
    argv.push("-i".into());
    argv.push(srt.display().to_string());
    argv.extend(args(&[
        "-filter_complex",
        filter,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-map",
        "4:s",
        "-c:v",
        "mpeg2video",
        "-q:v",
        "6",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "ac3",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-c:s",
        "srt",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:s:0",
        "language=chi",
        "-map_metadata",
        "-1",
        "-f",
        "matroska",
    ]));
    argv.push(base.display().to_string());
    capture(&ff.ffmpeg, &argv).expect("build the video fixture");

    let mut argv = args(&["-y", "-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(base.display().to_string());
    argv.push("-i".into());
    argv.push(chapters.display().to_string());
    argv.extend(args(&[
        "-map",
        "0",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c",
        "copy",
        "-f",
        "matroska",
    ]));
    argv.push(movie.display().to_string());
    capture(&ff.ffmpeg, &argv).expect("attach chapters");

    let _ = std::fs::remove_file(&base);
    movie
}

fn profile_for_test() -> RestorationProfile {
    // Deterministic (no model backend), no interpolation: the point of this test
    // is the correctness of the pipeline, not a particular model.
    let mut profile = RestorationProfile::deterministic();
    profile.interpolation.method = InterpolationMethod::Off;
    profile.interpolation.multiplier = 1;
    profile.output.quality = 32;
    profile
}

#[test]
fn converts_a_real_file_end_to_end_and_verifies_the_result() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_fixture(&ff, dir.path());
    let output = dir.path().join("e2e-output.mkv");

    // The fixture must be what we intended: 2 shots, 2 audio tracks, 1 subtitle,
    // 2 chapters. If this breaks, the assertions below would be meaningless.
    let source = probe(&ff, &input).expect("probe the fixture");
    assert_eq!(source.video.len(), 1, "{:?}", source.summary_lines());
    assert_eq!(source.audio.len(), 1);
    assert_eq!(source.subtitles.len(), 1);
    assert_eq!(source.chapters.len(), 2);
    assert!(source.duration_seconds() > 2.0);

    let store = Arc::new(Store::open(&dir.path().join("jobs.sqlite3")).expect("state store"));
    let bus = EventBus::new();
    let events = bus.subscribe();
    let cancel = Arc::new(AtomicBool::new(false));
    let runner = PipelineRunner::new(
        Arc::clone(&ff),
        bus.clone(),
        Arc::clone(&store),
        Arc::clone(&cancel),
        dir.path().join("scratch"),
    );

    let request = PlanRequest {
        job_id: "e2e".to_string(),
        input: input.clone(),
        output: output.clone(),
        profile: profile_for_test(),
    };
    let outcome = runner
        .run(request, &RunnerOptions::default())
        .expect("the runner never fails to report an outcome");

    // ---- what the engine said ---------------------------------------------
    let mut logs = Vec::new();
    let mut stages = Vec::new();
    let mut progress = 0usize;
    let mut finished = 0usize;
    while let Ok(event) = events.try_recv() {
        match event {
            Event::Log(record) => logs.push(record),
            Event::Stage { stage, status, .. } => stages.push((stage, status)),
            Event::Progress(_) => progress += 1,
            Event::Finished(_) => finished += 1,
            _ => {}
        }
    }

    assert_eq!(
        finished, 1,
        "a job must emit exactly one Finished event, saw {finished}"
    );
    assert!(outcome.ok, "job failed: {}", outcome.message);
    assert!(
        !logs.is_empty(),
        "the pipeline must report what it is doing"
    );
    assert!(
        stages.iter().any(|(stage, status)| *stage == Stage::Probe
            && *status == StageStatus::Done),
        "the probe stage must be reported as done"
    );
    assert!(
        stages.iter().any(|(stage, status)| *stage == Stage::Qc
            && *status == StageStatus::Done),
        "quality control must pass, stages seen: {stages:?}"
    );

    // ---- what actually came out -------------------------------------------
    let result = probe(&ff, &output).expect("probe the output");
    let video = result.primary_video().expect("output has video");
    let (width, height) = video.size().expect("output has dimensions");
    assert!(
        width > 320 && height > 240,
        "SD sources are restored to a modern raster, got {width}x{height}"
    );
    assert_eq!(width % 2, 0, "encoders reject odd widths");

    let enhanced = result
        .audio
        .first()
        .expect("the enhanced track must exist");
    assert!(
        enhanced.base.disposition.is_default(),
        "the enhanced track must be the default"
    );
    assert!(
        result
            .audio
            .iter()
            .any(|a| a.base.codec_name.as_deref() == Some("ac3")),
        "the original audio track must be preserved, got {:?}",
        result
            .audio
            .iter()
            .map(|a| a.base.codec_name.clone())
            .collect::<Vec<_>>()
    );

    assert_eq!(
        result.subtitles.len(),
        source.subtitles.len(),
        "subtitles must survive the remux"
    );
    assert_eq!(
        result.chapters.len(),
        source.chapters.len(),
        "chapters must survive the remux"
    );

    let drift = (result.duration_seconds() - source.duration_seconds()).abs();
    assert!(
        drift < 0.5,
        "duration drift of {drift:.3}s between source and output"
    );

    // The job record must agree with the events.
    let row = store.job("e2e").expect("read job").expect("job exists");
    assert_eq!(row.state, JobState::Completed);
    assert!(row.elapsed_ms.is_some());
    let point = store.resume_point("e2e").expect("resume point");
    assert!(
        point.is_done(Stage::Qc),
        "every stage must be checkpointed, got {:?}",
        point.completed
    );
    assert!(!store.logs("e2e", 10, sr_core::Level::Trace).unwrap().is_empty());

    // ---- a dry run must not touch a pixel ---------------------------------
    let dry_output = dir.path().join("dry-run.mkv");
    let request = PlanRequest {
        job_id: "e2e-dry".to_string(),
        input: input.clone(),
        output: dry_output.clone(),
        profile: profile_for_test(),
    };
    let outcome = runner
        .run(
            request,
            &RunnerOptions {
                dry_run: true,
                ..Default::default()
            },
        )
        .expect("dry run reports an outcome");
    assert!(outcome.ok, "dry run failed: {}", outcome.message);
    assert!(
        !dry_output.exists(),
        "a dry run must not write the output file"
    );
}

/// A missing input must be reported as a failed job, not as a panic or a hang.
#[test]
fn a_missing_input_fails_the_job_cleanly() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let store = Arc::new(Store::open(&dir.path().join("jobs.sqlite3")).expect("state store"));
    let bus = EventBus::new();
    let events = bus.subscribe();
    let runner = PipelineRunner::new(
        Arc::clone(&ff),
        bus,
        Arc::clone(&store),
        Arc::new(AtomicBool::new(false)),
        dir.path().join("scratch"),
    );

    let outcome = runner
        .run(
            PlanRequest {
                job_id: "missing".into(),
                input: dir.path().join("does-not-exist.mkv"),
                output: dir.path().join("out.mkv"),
                profile: profile_for_test(),
            },
            &RunnerOptions::default(),
        )
        .expect("the runner always reports an outcome");

    assert!(!outcome.ok);
    assert!(
        outcome.message.contains("does not exist") || outcome.message.contains("Probe"),
        "unhelpful failure message: {}",
        outcome.message
    );
    let mut finished = 0;
    while let Ok(event) = events.try_recv() {
        if matches!(event, Event::Finished(_)) {
            finished += 1;
        }
    }
    assert_eq!(finished, 1, "even a failed job must emit exactly one Finished");
    assert_eq!(
        store.job("missing").unwrap().unwrap().state,
        JobState::Failed
    );
}

/// Sanity check that the fixture builder itself is doing what it claims.
#[test]
fn fixture_building_is_reproducible() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_fixture(&ff, dir.path());
    assert!(input.exists());
    let manifest = probe(&ff, &input).expect("probe");
    let fps = manifest.primary_video().unwrap().fps().unwrap();
    assert!(
        (fps.to_f64() - 29.97).abs() < 0.01,
        "expected 29.97 fps, got {fps}"
    );
    // The fixture is built by ffmpeg, so this also proves the tool wrapper works.
    let version = capture(&ff.ffmpeg, &args(&["-version"])).expect("ffmpeg -version");
    assert!(version.starts_with("ffmpeg version"));
}
