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
//! ## Why this test asks for re-grain
//!
//! Per-chunk resume only exists on the chunked executor, and the chunked executor is
//! now selected by a *measured* per-shot re-grain: with `output.regrain_strength > 0`
//! the planner measures each shot and builds one chunk per shot, which is the only
//! thing in this build that needs a per-chunk filter chain. The old selector was the
//! model stage, and the model stage is gone. So the profile below switches re-grain
//! on, and the executor is asserted to be `chunked` before anything is concluded from
//! the run; the test has nothing to say about grain itself.
//!
//! ## What is observed, now that the plugin call log is gone
//!
//! The progress signal used to be the example plugin's call log (`SR_INFER_CALL_LOG`
//! plus `SR_INFER_PLUGIN`): it counted chunk calls from inside the inference ABI, and
//! with that ABI deleted there is nothing left to read it. The durable record is
//! watched instead, and it is the stronger evidence of the two:
//!
//! * the `chunks` table of the job's state database, which the executor commits per
//!   chunk before it moves on — so it says what *survived*, not what was attempted;
//! * the length and modification time of every committed chunk artifact. A chunk that
//!   was re-encoded rewrites its file, so "not recomputed" becomes falsifiable rather
//!   than a claim about a log line;
//! * the plan's own `executor` field, read back from the stored plan.
//!
//! Skips itself when FFmpeg is unavailable.

use sr_core::events::Stage;
use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::RestorationProfile;
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::EventBus;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// The job id both the reference run and the interrupted run use. It is only a key
/// into each run's own state database.
const JOB: &str = "resume";

fn ffmpeg_or_skip() -> Option<Arc<Ffmpeg>> {
    match Ffmpeg::discover() {
        Ok(ff) => Some(Arc::new(ff)),
        Err(err) => {
            eprintln!("SKIPPED: {err}");
            None
        }
    }
}

/// Five one-second shots, so an interruption has somewhere to land that is not the
/// first or the last chunk.
fn build_fixture(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let input = dir.join("resume-input.mkv");
    let mut argv = args(&["-y", "-hide_banner", "-loglevel", "error"]);
    for source in [
        "testsrc2=size=320x240:rate=24",
        "mandelbrot=size=320x240:rate=24",
        "gradients=size=320x240:rate=24",
        "testsrc=size=320x240:rate=24",
        "smptebars=size=320x240:rate=24",
    ] {
        // Two seconds each, ten in total. The interrupted run below is cancelled by a
        // watcher polling the store, and at one second per source the whole job
        // finished before the poll saw two committed chunks whenever the workspace
        // suite was running concurrently. That is a race in the harness rather than in
        // the engine, and the fix is more work rather than a retry -- the same one
        // applied to cli_resume.rs for the same reason.
        argv.extend(args(&["-t", "2", "-f", "lavfi", "-i", source]));
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

/// The chunks the store says are durable, by index, with the file each one wrote.
fn committed_chunks(store: &Store, job: &str) -> BTreeMap<u32, PathBuf> {
    store
        .chunks(job)
        .expect("read the chunk table")
        .into_iter()
        .filter(|row| row.stage == Stage::Encode && row.status == "committed")
        .filter_map(|row| row.artifact.map(|artifact| (row.chunk_index, artifact)))
        .collect()
}

/// Chunk index → the input frames it covers. A resume has to reproduce the plan, not
/// merely a file of the right length: a different split would put the chunk
/// boundaries somewhere else and the output would differ at the joins.
fn chunk_spans(store: &Store, job: &str) -> BTreeMap<u32, (Option<i64>, Option<i64>)> {
    store
        .chunks(job)
        .expect("read the chunk table")
        .into_iter()
        .filter(|row| row.stage == Stage::Encode && row.status == "committed")
        .map(|row| (row.chunk_index, (row.start_frame, row.end_frame)))
        .collect()
}

/// The executor the stored plan names. Read back from the plan rather than assumed,
/// because everything this test concludes depends on the chunked path having run.
fn executor_of(store: &Store, job: &str) -> String {
    let plan_json = store
        .job(job)
        .expect("read the job")
        .and_then(|row| row.plan_json)
        .unwrap_or_else(|| panic!("job {job} has no plan"));
    let plan: serde_json::Value = serde_json::from_str(&plan_json).expect("the plan is JSON");
    plan.pointer("/video/executor")
        .and_then(|value| value.as_str())
        .unwrap_or("<absent>")
        .to_string()
}

/// Length and modification time: the two things a re-encode cannot leave alone.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Artifact {
    len: u64,
    modified: Option<SystemTime>,
}

fn artifact(path: &Path) -> Artifact {
    let meta = std::fs::metadata(path)
        .unwrap_or_else(|err| panic!("chunk artifact {} is unreadable: {err}", path.display()));
    Artifact {
        len: meta.len(),
        modified: meta.modified().ok(),
    }
}

struct Job<'a> {
    ff: &'a Arc<Ffmpeg>,
    input: &'a Path,
    dir: &'a Path,
    job_id: &'a str,
}

impl Job<'_> {
    fn store_path(&self) -> PathBuf {
        self.dir.join("resume.sqlite3")
    }

    fn run(&self, output: &Path, cancel: Arc<AtomicBool>, resume: bool) -> (bool, String) {
        let runner = PipelineRunner::new(
            Arc::clone(self.ff),
            EventBus::new(),
            Arc::new(Store::open(&self.store_path()).expect("state store")),
            cancel,
            self.dir.join("scratch"),
        );
        let mut profile = RestorationProfile::deterministic();
        // Not about grain: a measured per-shot re-grain is what selects the chunked
        // executor, and per-chunk resume exists only there.
        profile.output.regrain_strength = 8.0;
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
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_fixture(&ff, dir.path());

    // ---- the reference run: uninterrupted, in its own job directory --------
    let reference_dir = dir.path().join("reference");
    std::fs::create_dir_all(&reference_dir).expect("reference dir");
    let reference = Job {
        ff: &ff,
        input: &input,
        dir: &reference_dir,
        job_id: JOB,
    };
    let reference_output = reference_dir.join("out.mkv");
    let (ok, message) = reference.run(&reference_output, Arc::new(AtomicBool::new(false)), true);
    assert!(ok, "the reference run failed: {message}");
    let reference_hashes = frame_hashes(&ff, &reference_output);
    let reference_store = Store::open(&reference.store_path()).expect("reference store");
    assert_eq!(
        executor_of(&reference_store, JOB),
        "chunked",
        "the fixture must run through the chunked executor, or there is no per-chunk \
         resume for this test to interrupt"
    );
    let reference_chunks = committed_chunks(&reference_store, JOB);
    assert!(
        reference_chunks.len() >= 3,
        "the fixture must produce several chunks for an interruption to mean \
         anything, got {:?}",
        reference_chunks.keys().collect::<Vec<_>>()
    );
    let reference_spans = chunk_spans(&reference_store, JOB);

    // ---- the interrupted run, in the job directory the resume will use -----
    let job_dir = dir.path().join("job");
    std::fs::create_dir_all(&job_dir).expect("job dir");
    let job = Job {
        ff: &ff,
        input: &input,
        dir: &job_dir,
        job_id: JOB,
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let watcher = {
        let cancel = Arc::clone(&cancel);
        let state = job.store_path();
        std::thread::spawn(move || {
            // The store is opened once and reused: the database is in WAL mode, so a
            // reader neither blocks the writer nor is blocked by it, and a poll every
            // couple of milliseconds costs nothing.
            let Ok(store) = Store::open(&state) else {
                return false;
            };
            // Cancel once two chunks are committed: far enough in that the first is
            // durable, early enough that the job cannot finish.
            for _ in 0..6_000 {
                if committed_chunks(&store, JOB).len() >= 2 {
                    cancel.store(true, Ordering::Relaxed);
                    return true;
                }
                std::thread::sleep(Duration::from_millis(2));
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
        "the watcher never saw two committed chunks; the test cannot conclude anything"
    );

    // What the job actually committed before it was stopped. This is the table, not a
    // log: the commit happens before the run continues, so a row here means the file
    // it names was complete on disk when the process stopped.
    let store = Store::open(&job.store_path()).expect("reopen the store");
    let committed = committed_chunks(&store, JOB);
    for (index, path) in &committed {
        eprintln!("  chunk {index:04} committed: {}", path.display());
    }
    assert!(
        !committed.is_empty(),
        "an interrupted run must leave its finished chunks behind, or resume has \
         nothing to resume from"
    );
    let before: BTreeMap<u32, Artifact> = committed
        .iter()
        .map(|(index, path)| (*index, artifact(path)))
        .collect();

    // ---- the resumed run ---------------------------------------------------
    let resumed_output = job_dir.join("out.mkv");
    let (ok, message) = job.run(&resumed_output, Arc::new(AtomicBool::new(false)), true);
    assert!(ok, "the resumed run failed: {message}");

    let after = committed_chunks(&store, JOB);
    eprintln!(
        "reference chunks {:?}, committed before the interruption {:?}, committed \
         after the resumed run {:?}",
        reference_chunks.keys().collect::<Vec<_>>(),
        committed.keys().collect::<Vec<_>>(),
        after.keys().collect::<Vec<_>>()
    );

    // The heart of it: nothing that was already committed may be computed again. A
    // re-encoded chunk rewrites its artifact, so the file that was durable before the
    // interruption must be byte-length-identical and untouched afterwards.
    for (index, was) in &before {
        let path = after.get(index).unwrap_or_else(|| {
            panic!("chunk {index} was committed before the interruption and is missing from the table now")
        });
        assert_eq!(
            path,
            committed.get(index).expect("same index"),
            "chunk {index} must still name the artifact it committed"
        );
        assert_eq!(
            artifact(path),
            *was,
            "chunk {index} was recomputed by the resumed run ({} changed), which was \
             already committed before the interruption; resume is not resuming",
            path.display()
        );
    }

    // And the resumed run must have produced the chunks it had not reached: the whole
    // plan, with the same boundaries, or the film would join differently.
    assert_eq!(
        after.keys().copied().collect::<BTreeSet<u32>>(),
        reference_chunks.keys().copied().collect::<BTreeSet<u32>>(),
        "the interrupted and resumed runs together must cover exactly the chunks a \
         single run covers"
    );
    assert_eq!(
        chunk_spans(&store, JOB),
        reference_spans,
        "the resumed run must split the film where the reference run did"
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
