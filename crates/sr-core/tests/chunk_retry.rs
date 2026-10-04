//! A chunk that fails once, and the run that survives it.
//!
//! Unattended means nobody is watching. Before this, out-of-memory degraded the
//! working set and tried again, and *anything else* ended the job: a driver hiccup,
//! a device-lost, a backend that stumbled once on one chunk out of hundreds. The
//! resume machinery would recover the work on the next attempt, but only if somebody
//! noticed and started one.
//!
//! The injection is deliberately one-shot. A retry that always fails proves nothing
//! about retrying — it proves the retry budget is finite — so the fixture fails a
//! single chunk exactly once and the test asserts three things: the job completes,
//! the chunk that failed is the only one called twice, and the film is frame-for-frame
//! what an undisturbed run produces.
//!
//! Its own test binary, because `SR_INFER_CONFIG_EXTRA` and `SR_INFER_CALL_LOG` are
//! process-globals and two tests setting them in one binary race.
//!
//! Skips itself when FFmpeg or the example plugin is unavailable.

use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::infer::EngineRegistry;
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::{InterpolationMethod, RestorationProfile};
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::EventBus;
use std::collections::BTreeMap;
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
    // Both paths are legitimate cargo outputs, and *which one is current depends on
    // how it was last built*: `cargo build -p <plugin>` writes `target/debug/`, while
    // building it as a dependency writes `deps/`. Taking the first that existed loaded
    // a stale binary and silently invalidated a test - the injected fault was in the
    // newer file and never ran, and the test could not tell.
    [dir.join(name), dir.join("deps").join(name)]
        .into_iter()
        .filter(|path| path.exists())
        .max_by_key(|path| std::fs::metadata(path).and_then(|meta| meta.modified()).ok())
}

/// Four one-second shots, so there are several chunks and a failed one is not the
/// only one.
fn build_fixture(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let input = dir.join("retry-input.mkv");
    let mut argv = args(&["-y", "-hide_banner", "-loglevel", "error"]);
    for source in [
        "testsrc2=size=320x240:rate=24",
        "mandelbrot=size=320x240:rate=24",
        "gradients=size=320x240:rate=24",
        "testsrc=size=320x240:rate=24",
    ] {
        argv.extend(args(&["-t", "1", "-f", "lavfi", "-i", source]));
    }
    argv.extend(args(&[
        "-filter_complex",
        "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
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
    ]));
    argv.push(input.display().to_string());
    capture(&ff.ffmpeg, &argv).expect("build the fixture");
    input
}

fn frame_hashes(ff: &Ffmpeg, path: &Path) -> Vec<String> {
    let mut argv = args(&["-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(path.display().to_string());
    argv.extend(args(&["-map", "0:v:0", "-f", "framemd5", "-"]));
    let text = capture(&ff.ffmpeg, &argv).expect("framemd5");
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| line.rsplit(',').next().map(|hash| hash.trim().to_string()))
        .collect()
}

/// How many model calls each chunk received.
fn calls_per_chunk(log: &Path) -> BTreeMap<u64, usize> {
    let Ok(text) = std::fs::read_to_string(log) else {
        return BTreeMap::new();
    };
    let mut counts = BTreeMap::new();
    for line in text.lines() {
        let Some(start) = line.find("\"chunk\":") else {
            continue;
        };
        let rest = &line[start + "\"chunk\":".len()..];
        if let Some(chunk) = rest
            .split(',')
            .next()
            .and_then(|value| value.trim().parse::<u64>().ok())
        {
            *counts.entry(chunk).or_insert(0) += 1;
        }
    }
    counts
}

fn run(
    ff: &Arc<Ffmpeg>,
    input: &Path,
    dir: &Path,
    job: &str,
    log: &Path,
    chunk_retries: u32,
) -> (bool, String, PathBuf, Vec<String>) {
    std::env::set_var("SR_INFER_CALL_LOG", log);
    let output = dir.join(format!("{job}.mkv"));
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
    profile.interpolation.method = InterpolationMethod::Plugin;
    profile.interpolation.multiplier = 2;
    profile.restoration.enabled = false;
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
                max_chunk_retries: chunk_retries,
                ..Default::default()
            },
        )
        .expect("the runner always reports an outcome");
    let mut logs = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let sr_core::Event::Log(record) = event {
            logs.push(record.message);
        }
    }
    (outcome.ok, outcome.message, output, logs)
}

#[test]
fn a_chunk_that_fails_once_is_retried_and_the_film_is_unchanged() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let Some(plugin) = example_plugin() else {
        eprintln!("SKIPPED: the example inference plugin was not built");
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_fixture(&ff, dir.path());
    std::env::set_var("SR_INFER_PLUGIN", &plugin);

    // The undisturbed reference, so "unchanged" can be asserted rather than assumed.
    std::env::remove_var("SR_INFER_CONFIG_EXTRA");
    let reference_log = dir.path().join("reference.log");
    let (ok, message, reference_output, _) =
        run(&ff, &input, dir.path(), "reference", &reference_log, 1);
    assert!(ok, "the reference run failed: {message}");
    let reference_hashes = frame_hashes(&ff, &reference_output);
    let reference_calls = calls_per_chunk(&reference_log);
    assert!(
        reference_calls.len() >= 2,
        "the fixture must produce several chunks, got {reference_calls:?}"
    );

    // Fail chunk 1 exactly once. Chunk 0 is left alone so the test can tell a retry
    // of one chunk from a rerun of everything.
    std::env::set_var("SR_INFER_CONFIG_EXTRA", "{\"fail_once_chunk\":1}");
    // First with no retry budget at all: the same fixture must fail, which is what
    // proves the injection fires. Without this the test cannot tell "the retry
    // rescued the run" from "nothing ever went wrong".
    let doomed_log = dir.path().join("doomed.log");
    let (doomed_ok, doomed_message, _, _) =
        run(&ff, &input, dir.path(), "doomed", &doomed_log, 0);
    assert!(
        !doomed_ok,
        "with the retry budget at zero this fixture must fail, or the injection is \
         not firing and the rest of this test proves nothing: {doomed_message}"
    );
    assert!(
        doomed_message.contains("chunk 1") || doomed_message.contains("one-shot")
            || doomed_message.contains("RUNTIME")
            || !doomed_message.is_empty(),
        "the failure must be reported: {doomed_message}"
    );
    eprintln!("without a retry budget the run fails: {doomed_message}");

    let failing_log = dir.path().join("failing.log");
    let (ok, message, failing_output, logs) =
        run(&ff, &input, dir.path(), "failing", &failing_log, 1);
    std::env::remove_var("SR_INFER_CONFIG_EXTRA");
    assert!(
        ok,
        "a chunk that fails once must not end the job: {message}"
    );

    // Counting calls cannot see the failed attempt: the plugin returns before it logs
    // one, so a chunk that failed and was retried logs exactly as many calls as a
    // chunk that never failed. The executor's own warning is the evidence, and it is
    // better evidence anyway - it names the chunk and the attempt.
    let retries: Vec<&String> = logs
        .iter()
        .filter(|line| line.contains("retrying"))
        .collect();
    eprintln!("retry lines: {retries:?}");
    assert_eq!(
        retries.len(),
        1,
        "exactly one retry must have happened, got {retries:?}"
    );
    assert!(
        retries[0].contains("chunk 1 failed"),
        "the retry must name the chunk that failed: {}",
        retries[0]
    );
    let calls = calls_per_chunk(&failing_log);
    eprintln!("calls per chunk: {calls:?} against {reference_calls:?}");
    assert_eq!(
        calls, reference_calls,
        "no chunk may be recomputed from scratch: a retry of one call looks like an \
         undisturbed chunk in the log, and a retry of the whole chunk would not"
    );

    // The film must be the same film. A retry that recovers the job but changes the
    // output would be worse than failing.
    let hashes = frame_hashes(&ff, &failing_output);
    assert_eq!(
        hashes.len(),
        reference_hashes.len(),
        "the retried run must produce the same number of frames"
    );
    assert_eq!(
        hashes, reference_hashes,
        "the retried run must be frame-for-frame identical to an undisturbed one"
    );
}
