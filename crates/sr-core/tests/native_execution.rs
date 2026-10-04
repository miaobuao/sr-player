//! Proof that the model backend really executes.
//!
//! The gap this test closes was not a missing feature but a missing *connection*:
//! the engine could load a plugin, read its capabilities, select it during
//! planning and then hand the whole job to a single FFmpeg pass, so the log said
//! "model interpolation" while `minterpolate` did the work. A test that only
//! checked "the job succeeded" would pass in both worlds, so this one checks the
//! things that are only true when frames really went through the backend:
//!
//! 1. the reference plugin logged a call for every shot — it only writes that log
//!    from inside `sr_infer_execute`;
//! 2. **no call ever spanned a cut**, which is the promise the whole shot-splitting
//!    machinery exists to keep;
//! 3. the output frame count is exactly `2*(n-1)+1`, so the executor's accounting
//!    survived chunking;
//! 4. consecutive output frames are *not* duplicated, which is what a frame
//!    duplication fallback would produce;
//! 5. an injected out-of-memory fault degrades and still completes the job;
//! 6. a second run resumes from the committed chunks instead of redoing the work.
//!
//! It skips itself when FFmpeg or the example plugin is missing.

use sr_core::events::{Event, Stage, StageStatus};
use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::infer::EngineRegistry;
use sr_core::media::probe;
use sr_core::pipeline::native::ChunkEncoding;
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::RestorationProfile;
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::EventBus;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn ffmpeg_or_skip() -> Option<Arc<Ffmpeg>> {
    match Ffmpeg::discover() {
        Ok(ff) => Some(Arc::new(ff)),
        Err(err) => {
            eprintln!("SKIPPED: {err}");
            None
        }
    }
}

/// The example plugin is a dev-dependency of this crate, so cargo builds it
/// before the tests run.
fn example_plugin() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?.parent()?;
    let name = if cfg!(windows) {
        "sr_infer_plugin_example.dll"
    } else if cfg!(target_os = "macos") {
        "libsr_infer_plugin_example.dylib"
    } else {
        "libsr_infer_plugin_example.so"
    };
    [dir.join(name), dir.join("deps").join(name)]
        .into_iter()
        .find(|p| p.exists())
}

/// Three one-second shots at 24 fps with hard cuts at 1.0 s and 2.0 s.
///
/// Progressive on purpose: this test is about the model path, not cadence
/// handling, so nothing may insert or drop frames before the model sees them.
/// Every shot moves: a static pattern would make "did the interpolator produce
/// anything new" unanswerable.
fn build_fixture(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let input = dir.join("native-input.mkv");
    let mut argv = args(&[
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        "1",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=24",
        "-t",
        "1",
        "-f",
        "lavfi",
        "-i",
        "mandelbrot=size=320x240:rate=24",
        "-t",
        "1",
        "-f",
        "lavfi",
        "-i",
        "gradients=size=320x240:rate=24",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
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
    ]);
    argv.push(input.display().to_string());
    capture(&ff.ffmpeg, &argv).expect("build the fixture");
    input
}

/// Per-frame hashes of a file, in presentation order.
fn frame_hashes(ff: &Ffmpeg, path: &Path) -> Vec<String> {
    let mut argv = args(&[
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
    ]);
    argv.push(path.display().to_string());
    argv.extend(args(&["-map", "0:v:0", "-f", "framemd5", "-"]));
    let text = capture(&ff.ffmpeg, &argv).expect("framemd5");
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| line.rsplit(',').next().map(|h| h.trim().to_string()))
        .collect()
}

#[derive(Debug)]
struct ModelCall {
    op: u32,
    chunk: u64,
    first_pts: f64,
    last_pts: f64,
    inputs: usize,
}

fn read_calls(path: &Path) -> Vec<ModelCall> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| {
            // Deliberately hand-parsed: the log is the plugin's, and the test
            // should fail if its shape changes rather than silently pass.
            let field = |key: &str| -> Option<&str> {
                let needle = format!("\"{key}\":");
                let start = line.find(&needle)? + needle.len();
                Some(line[start..].split(',').next()?.trim())
            };
            Some(ModelCall {
                op: field("op")?.parse().ok()?,
                chunk: field("chunk")?.parse().ok()?,
                first_pts: field("first_pts")?.parse().ok()?,
                last_pts: field("last_pts")?.parse().ok()?,
                inputs: field("in")?.parse().ok()?,
            })
        })
        .collect()
}

#[test]
fn a_model_backend_executes_the_frames_and_never_sees_a_cut() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let Some(plugin) = example_plugin() else {
        eprintln!("SKIPPED: the example inference plugin was not built");
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let call_log = dir.path().join("model-calls.jsonl");

    // These are read by `abi::discover` and by the plugin's own logging. This is
    // the only test in this binary, so there is no parallel test to disturb.
    std::env::set_var("SR_INFER_PLUGIN", &plugin);
    std::env::set_var("SR_INFER_CALL_LOG", &call_log);
    // Fail the first two model calls. The executor must walk the degrade ladder
    // and finish the job rather than reporting a failure.
    std::env::set_var("SR_INFER_CONFIG_EXTRA", "{\"fake_oom\":2}");

    let input = build_fixture(&ff, dir.path());
    let output = dir.path().join("native-output.mkv");
    let source = probe(&ff, &input).expect("probe the fixture");
    let source_video = source.primary_video().expect("fixture has video");
    let source_frames = source_video.base.nb_frames.unwrap_or(72);
    assert_eq!(source_frames, 72, "the fixture must be three 24-frame shots");

    let store = Arc::new(Store::open(&dir.path().join("jobs.sqlite3")).expect("state store"));
    let bus = EventBus::new();
    let events = bus.subscribe();
    let engines = Arc::new(EngineRegistry::probe(Arc::clone(&ff)));
    assert!(
        engines.has_plugin(),
        "the example plugin at {} must be discovered",
        plugin.display()
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let runner = PipelineRunner::new(
        Arc::clone(&ff),
        Arc::clone(&engines),
        bus.clone(),
        Arc::clone(&store),
        Arc::clone(&cancel),
        dir.path().join("scratch"),
    );

    let mut profile = RestorationProfile::safe_16gb();
    // Keep the output at source resolution and prefer software so the test does
    // not depend on which hardware encoder this machine happens to have.
    profile.restoration.max_upscale = 1.0;
    profile.output.prefer_hardware = false;
    profile.output.quality = 30;
    assert!(
        profile.restoration.enabled,
        "the safe profile must ask for model restoration"
    );

    let request = PlanRequest {
        job_id: "native".to_string(),
        input: input.clone(),
        output: output.clone(),
        profile,
    };
    let options = RunnerOptions {
        resume: true,
        keep_intermediates: true,
        dry_run: false,
        max_chunk_retries: 1,
                max_degrade_retries: 4,
        chunk_encoding: ChunkEncoding::LosslessIntermediate,
    };
    let outcome = runner
        .run(request.clone(), &options)
        .expect("the runner always reports an outcome");

    let mut logs = Vec::new();
    let mut stages = Vec::new();
    while let Ok(event) = events.try_recv() {
        match event {
            Event::Log(record) => logs.push(record.message),
            Event::Stage { stage, status, .. } => stages.push((stage, status)),
            _ => {}
        }
    }

    assert!(
        outcome.ok,
        "the job failed: {}\ninteresting log lines:\n{}",
        outcome.message,
        logs.iter()
            .filter(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("native")
                    || lower.contains("chunk")
                    || lower.contains("fail")
                    || lower.contains("model")
                    || lower.contains("degrad")
                    || lower.contains("memory")
            })
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );

    // ---- 1. the backend ran at all ----------------------------------------
    let calls = read_calls(&call_log);
    assert!(
        !calls.is_empty(),
        "the plugin logged no calls: the model path did not execute. Log tail:\n{}",
        logs.iter().rev().take(25).cloned().collect::<Vec<_>>().join("\n")
    );
    let interpolations: Vec<&ModelCall> = calls.iter().filter(|c| c.op == 2).collect();
    let restores: Vec<&ModelCall> = calls.iter().filter(|c| c.op == 1).collect();
    assert!(
        !interpolations.is_empty(),
        "no interpolation call reached the backend, only {:?}",
        calls.iter().map(|c| c.op).collect::<Vec<_>>()
    );
    assert!(
        !restores.is_empty(),
        "the plan asked for restoration but the backend was never asked to restore"
    );

    // ---- 2. no call ever straddled a cut ----------------------------------
    // The fixture's cuts are at 1.0 s and 2.0 s.
    for cut in [1.0f64, 2.0] {
        for call in &interpolations {
            assert!(
                !(call.first_pts < cut - 1e-6 && call.last_pts > cut + 1e-6),
                "a model call covered the cut at {cut}s: frames {:.4}s..{:.4}s (chunk {})",
                call.first_pts,
                call.last_pts,
                call.chunk
            );
        }
    }
    // ... and the cuts were genuinely respected rather than the file happening to
    // have none: some call must end before 1.0s and another start at or after it.
    assert!(
        interpolations.iter().any(|c| c.last_pts <= 1.0 + 1e-6),
        "no call finished before the first cut, so the shot split is not being exercised"
    );
    assert!(
        interpolations.iter().any(|c| c.first_pts >= 1.0 - 1e-6),
        "no call started at or after the first cut"
    );
    // Three shots must produce model work in three distinct chunks.
    let chunks: HashSet<u64> = interpolations.iter().map(|c| c.chunk).collect();
    assert!(
        chunks.len() >= 3,
        "expected at least one chunk per shot, saw chunk ids {chunks:?}"
    );

    // ---- 3. the degrade ladder answered the injected OOM -------------------
    assert!(
        stages
            .iter()
            .any(|(_, status)| *status == StageStatus::Degraded),
        "an injected out-of-memory fault must be answered by the degrade ladder, \
         stages seen: {stages:?}"
    );
    assert!(
        logs.iter().any(|line| line.contains("out of memory")),
        "the log must say what happened: {:?}",
        logs.iter().filter(|l| l.contains("memory")).collect::<Vec<_>>()
    );

    // ---- 4. the output has exactly the frames the plan promised ------------
    let result = probe(&ff, &output).expect("probe the output");
    let video = result.primary_video().expect("output has video");
    let hashes = frame_hashes(&ff, &output);
    let expected = 2 * (source_frames - 1) + 1;
    assert_eq!(
        hashes.len() as u64,
        expected,
        "2x interpolation of {source_frames} frames must emit exactly {expected}"
    );
    let fps = video.fps().map(|f| f.to_f64()).unwrap_or(0.0);
    assert!(
        (fps - 48.0).abs() < 0.1,
        "the output must run at twice the source rate, got {fps:.3} fps"
    );

    // ---- 5. frames are synthesised, not duplicated ------------------------
    //
    // Duplication would produce an output made *only* of source frames, each
    // appearing `multiplier` times. Anything beyond the source frame count is a
    // frame the source never contained, so this is the property that separates
    // "the model interpolated" from "the fallback repeated frames".
    let distinct: HashSet<&String> = hashes.iter().collect();
    assert!(
        distinct.len() > source_frames as usize,
        "the output has {} distinct frames for {source_frames} source frames: every frame \
         could have come from the source, so nothing was synthesised",
        distinct.len()
    );
    assert!(
        distinct.len() >= source_frames as usize + source_frames as usize / 4,
        "only {} distinct frames from {source_frames} source frames: too few frames were \
         synthesised to believe the model ran",
        distinct.len()
    );
    // A frame repeated immediately is normal at a held boundary, but most of the
    // file must not look like an exact 2x duplication.
    let duplicated = hashes.windows(2).filter(|pair| pair[0] == pair[1]).count();
    assert!(
        duplicated * 4 < hashes.len(),
        "{duplicated} of {} frame pairs are identical: the output looks like frame \
         duplication",
        hashes.len() - 1
    );

    // ---- 6. the second run resumes instead of redoing the work -------------
    std::env::remove_var("SR_INFER_CONFIG_EXTRA");
    let before = std::fs::metadata(&call_log).map(|m| m.len()).unwrap_or(0);
    let bus = EventBus::new();
    let events = bus.subscribe();
    let runner = PipelineRunner::new(
        Arc::clone(&ff),
        engines,
        bus.clone(),
        Arc::clone(&store),
        cancel,
        dir.path().join("scratch"),
    );
    let outcome = runner
        .run(request, &options)
        .expect("the resumed run reports an outcome");
    assert!(outcome.ok, "the resumed run failed: {}", outcome.message);
    let mut resumed_logs = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let Event::Log(record) = event {
            resumed_logs.push(record.message);
        }
    }
    assert!(
        resumed_logs
            .iter()
            .any(|line| line.contains("already committed")),
        "the resumed run must reuse the committed chunks, log tail:\n{}",
        resumed_logs.iter().rev().take(20).cloned().collect::<Vec<_>>().join("\n")
    );
    let after = std::fs::metadata(&call_log).map(|m| m.len()).unwrap_or(0);
    assert_eq!(
        before, after,
        "a fully resumed run must not call the model again"
    );
    let chunks = store.chunks("native").expect("chunk rows");
    assert!(
        chunks.len() >= 3,
        "every chunk must be recorded for resume, found {}",
        chunks.len()
    );
    assert!(chunks
        .iter()
        .all(|row| row.stage == Stage::Encode && row.status == "committed"));
}
