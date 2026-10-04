//! Interrupting a job and resuming it.
//!
//! "Unattended" means a run that is stopped halfway — by a crash, a power cut, an
//! operator killing it — finishes correctly on the next attempt without redoing the
//! work it already completed and without producing a different film. The chunks
//! table exists for this and nothing had ever interrupted a run to prove it.
//!
//! The interruption is a cancellation rather than a `kill`, and that is the weaker
//! of the two: it lets the pipeline reach an orderly stop. What it cannot fake is
//! the property being tested — that committed chunks survive and are not recomputed
//! — because the commit happens per chunk, on disk, before the run continues.
//!
//! Self-contained rather than sharing helpers with `native_execution.rs`: this is a
//! separate test binary because `SR_INFER_CALL_LOG` is a process-global and two
//! tests setting it in one binary race.
//!
//! Skips itself when FFmpeg or the example plugin is unavailable.

use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::infer::EngineRegistry;
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::{InterpolationMethod, RestorationProfile};
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::EventBus;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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
        .find(|path| path.exists())
}

/// Five one-second shots, so an interruption has somewhere to land that is not the
/// first or the last chunk.
fn build_fixture(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let input = dir.join("resume-input.mkv");
    let mut argv = args(&["-y", "-hide_banner", "-loglevel", "error"]);
    for (index, source) in [
        "testsrc2=size=320x240:rate=24",
        "mandelbrot=size=320x240:rate=24",
        "gradients=size=320x240:rate=24",
        "testsrc=size=320x240:rate=24",
        "smptebars=size=320x240:rate=24",
    ]
    .iter()
    .enumerate()
    {
        let _ = index;
        argv.extend(args(&["-t", "1", "-f", "lavfi", "-i", source]));
    }
    argv.extend(args(&[
        "-filter_complex",
        "[0:v][1:v][2:v][3:v][4:v]concat=n=5:v=1:a=0[v]",
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

/// The chunk ids a plugin call log mentions, in order.
fn chunks_called(log: &Path) -> Vec<u64> {
    let Ok(text) = std::fs::read_to_string(log) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| {
            let needle = "\"chunk\":";
            let start = line.find(needle)? + needle.len();
            line[start..].split(',').next()?.trim().parse().ok()
        })
        .collect()
}

struct Job<'a> {
    ff: &'a Arc<Ffmpeg>,
    input: &'a Path,
    dir: &'a Path,
    job_id: &'a str,
}

impl Job<'_> {
    fn run(&self, output: &Path, cancel: Arc<AtomicBool>, resume: bool) -> (bool, String) {
        let runner = PipelineRunner::new(
            Arc::clone(self.ff),
            Arc::new(EngineRegistry::probe(Arc::clone(self.ff))),
            EventBus::new(),
            Arc::new(Store::open(&self.dir.join("resume.sqlite3")).expect("state store")),
            cancel,
            self.dir.join("scratch"),
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
                    job_id: self.job_id.into(),
                    input: self.input.to_path_buf(),
                    output: output.to_path_buf(),
                    profile,
                },
                &RunnerOptions {
                    resume,
                    keep_intermediates: true,
                    ..Default::default()
                },
            )
            .expect("the runner always reports an outcome");
        (outcome.ok, outcome.message)
    }
}

#[test]
fn an_interrupted_job_resumes_without_recomputing_committed_chunks() {
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

    // ---- the reference run: uninterrupted, in its own job directory --------
    let reference_dir = dir.path().join("reference");
    std::fs::create_dir_all(&reference_dir).expect("reference dir");
    let reference_log = dir.path().join("reference.log");
    std::env::set_var("SR_INFER_CALL_LOG", &reference_log);
    let reference = Job {
        ff: &ff,
        input: &input,
        dir: &reference_dir,
        job_id: "resume",
    };
    let reference_output = reference_dir.join("out.mkv");
    let (ok, message) = reference.run(&reference_output, Arc::new(AtomicBool::new(false)), true);
    assert!(ok, "the reference run failed: {message}");
    let reference_hashes = frame_hashes(&ff, &reference_output);
    let reference_chunks: BTreeSet<u64> = chunks_called(&reference_log).into_iter().collect();
    assert!(
        reference_chunks.len() >= 3,
        "the fixture must produce several chunks for an interruption to mean \
         anything, got {:?}",
        reference_chunks
    );

    // ---- the interrupted run, in the job directory the resume will use -----
    let job_dir = dir.path().join("job");
    std::fs::create_dir_all(&job_dir).expect("job dir");
    let interrupted_log = dir.path().join("interrupted.log");
    std::env::set_var("SR_INFER_CALL_LOG", &interrupted_log);
    let job = Job {
        ff: &ff,
        input: &input,
        dir: &job_dir,
        job_id: "resume",
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let watcher = {
        let cancel = Arc::clone(&cancel);
        let log = interrupted_log.clone();
        std::thread::spawn(move || {
            // Cancel once two chunks have been worked on: far enough in that the
            // first is committed, early enough that the job cannot finish.
            for _ in 0..4_000 {
                let distinct: BTreeSet<u64> = chunks_called(&log).into_iter().collect();
                if distinct.len() >= 2 {
                    cancel.store(true, Ordering::Relaxed);
                    return true;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            false
        })
    };
    let interrupted_output = job_dir.join("out.mkv");
    let (ok, message) = job.run(&interrupted_output, Arc::clone(&cancel), true);
    let cancelled_in_time = watcher.join().unwrap_or(false);
    assert!(
        !ok,
        "the run was supposed to be interrupted, but it finished: {message}"
    );
    assert!(
        cancelled_in_time,
        "the watcher never saw two chunks; the test cannot conclude anything"
    );

    // What the job actually committed before it was stopped, from the store rather
    // than from the log: the log says what was *attempted*, the table says what
    // survived, and resume follows the table.
    let store = Store::open(&job_dir.join("resume.sqlite3")).expect("reopen the store");
    let rows = store.chunks("resume").expect("read the chunks");
    for row in &rows {
        eprintln!(
            "  chunk row: stage {:?} index {} status {} artifact {:?}",
            row.stage, row.chunk_index, row.status, row.artifact
        );
    }
    let committed: BTreeSet<u64> = rows
        .iter()
        .filter(|chunk| chunk.status == "committed")
        .map(|chunk| chunk.chunk_index as u64)
        .collect();
    eprintln!(
        "interrupted after {} chunk(s) committed: {committed:?}",
        committed.len()
    );
    assert!(
        !committed.is_empty(),
        "an interrupted run must leave its finished chunks behind, or resume has \
         nothing to resume from"
    );

    // ---- the resumed run ---------------------------------------------------
    let resumed_log = dir.path().join("resumed.log");
    std::env::set_var("SR_INFER_CALL_LOG", &resumed_log);
    let resumed_output = job_dir.join("out.mkv");
    let (ok, message) = job.run(&resumed_output, Arc::new(AtomicBool::new(false)), true);
    assert!(ok, "the resumed run failed: {message}");

    let resumed_calls: Vec<u64> = chunks_called(&resumed_log);
    let resumed_chunks: BTreeSet<u64> = resumed_calls.iter().copied().collect();
    eprintln!(
        "reference chunks {reference_chunks:?}, committed before the interruption \
         {committed:?}, recomputed after {resumed_chunks:?}"
    );

    // The heart of it: nothing that was already committed may be computed again.
    let repeated: Vec<u64> = committed.intersection(&resumed_chunks).copied().collect();
    assert!(
        repeated.is_empty(),
        "the resumed run recomputed chunks {repeated:?}, which were already \
         committed; resume is not resuming"
    );
    // And every chunk must have been done exactly once across the two runs.
    let done: BTreeSet<u64> = committed.union(&resumed_chunks).copied().collect();
    assert_eq!(
        done, reference_chunks,
        "the interrupted and resumed runs together must cover exactly the chunks a \
         single run covers"
    );

    // The film must be the same film: a resume that produces different frames is
    // worse than one that redoes the work.
    let resumed_hashes = frame_hashes(&ff, &resumed_output);
    assert_eq!(
        resumed_hashes.len(),
        reference_hashes.len(),
        "the resumed output must have the same number of frames"
    );
    assert_eq!(
        resumed_hashes, reference_hashes,
        "the resumed output must be frame-for-frame identical to an uninterrupted run"
    );
}
