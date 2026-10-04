//! Killing the program, and starting it again.
//!
//! Everything verified about resume so far went through the library. This goes
//! through `sr-cli`, because that is what an unattended user runs, and the two are
//! not the same thing: the command line derived its job id from the clock, so the
//! chunks table was keyed by a name that never recurred and resume was unreachable
//! from it entirely. That is fixed, and these are the tests that would have caught
//! it.
//!
//! The interruption is `Child::kill`, which on Windows is `TerminateProcess` and on
//! Unix a `SIGKILL`: no unwinding, no destructors, no flush. The engine gets no
//! chance to tidy up, which is the point — a power cut does not ask permission. What
//! survives is whatever was already durable on disk, and that is the property under
//! test.
//!
//! ## One test per executor
//!
//! The command line can reach both of the pipeline's video paths, and each has its
//! own test because they resume from different things:
//!
//! * [`a_killed_cli_run_resumes_instead_of_starting_over`] runs the **single-pass**
//!   executor (no `--regrain`). There are no chunks on that path, so what resume
//!   means there is the stage results: the job id must be a function of the
//!   conversion rather than of the clock, and the second run must find the first
//!   run's committed analysis and skip it.
//! * [`a_killed_cli_run_reuses_its_committed_chunks`] runs the **chunked** executor —
//!   `--regrain` is what selects it, because a measured per-shot re-grain is the one
//!   thing in this build that needs a per-chunk filter chain. There, resume means the
//!   chunks table: a chunk committed before the kill must not be recomputed, and the
//!   chunks that were *not* committed are the ones the second run has to produce.
//!
//! The old version of the first test drove the example inference plugin
//! (`--interpolate plugin` plus `SR_INFER_PLUGIN`/`SR_INFER_CALL_LOG`) and compared
//! the chunks the two runs called the model for. Both are gone with the inference
//! layer, and the plugin call log is replaced in both tests by the durable record:
//! the `chunks` table the executor commits to, plus the length and modification time
//! of each chunk artifact. A re-encoded chunk rewrites its file, so "not recomputed"
//! is falsifiable rather than a claim about a log line.
//!
//! `--chunk-encoding ffv1` is passed explicitly in the chunked test: the lossless
//! intermediates are what make the per-chunk artifacts on disk worth measuring, and
//! the published file is then a single encode of them, which is also what makes a
//! resumed run reproducible frame for frame.
//!
//! Skips itself when FFmpeg is unavailable.

use sr_core::events::{Stage, StageStatus};
use sr_core::pipeline::plan::VideoExecutor;
use sr_core::state::{JobRow, Store};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn cli() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sr-cli"))
}

/// Four one-second shots, so the job has several chunks and a kill has somewhere to
/// land that is neither the first nor the last.
fn build_fixture(dir: &Path) -> PathBuf {
    build_fixture_repeated(dir, 1)
}

/// The same fixture, `repeats` times over.
///
/// The single-pass test needs a run long enough to interrupt. Four seconds at
/// 320x240 converts faster than a database poll round-trip when the whole workspace
/// suite is running, so the process could exit before the watcher ever saw its
/// committed plan — a race in the test harness, not in the engine, and one that made
/// the test fail only under load. Giving it enough work removes the race instead of
/// papering over it with a retry.
fn build_fixture_repeated(dir: &Path, repeats: usize) -> PathBuf {
    const SOURCES: [&str; 4] = [
        "testsrc2=size=320x240:rate=24",
        "mandelbrot=size=320x240:rate=24",
        "gradients=size=320x240:rate=24",
        "testsrc=size=320x240:rate=24",
    ];
    let input = dir.join("cli-input.mkv");
    let mut command = Command::new("ffmpeg");
    command.args(["-y", "-hide_banner", "-loglevel", "error"]);
    let mut labels = String::new();
    for repeat in 0..repeats {
        for (offset, source) in SOURCES.iter().enumerate() {
            command.args(["-t", "1", "-f", "lavfi", "-i", source]);
            labels.push_str(&format!("[{}:v]", repeat * SOURCES.len() + offset));
        }
    }
    let inputs = repeats * SOURCES.len();
    let status = command
        .args([
            "-filter_complex",
            &format!("{labels}concat=n={inputs}:v=1:a=0[v]"),
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
        ])
        .arg(&input)
        .status()
        .expect("build the fixture");
    assert!(status.success(), "ffmpeg could not build the fixture");
    input
}

/// The conversion as the command line runs it, with no model task: asking for
/// restoration or interpolation is refused before any pixel moves.
fn convert(input: &Path, output: &Path, state: &Path) -> Command {
    let mut command = Command::new(cli());
    command
        .arg("--state-db")
        .arg(state)
        .arg("convert")
        .arg(input)
        .arg("-o")
        .arg(output)
        .args(["--profile", "deterministic", "--interpolate", "off"])
        .arg("--no-audio")
        .arg("--keep-intermediates");
    command
}

/// The same conversion on the chunked executor.
fn convert_chunked(input: &Path, output: &Path, state: &Path) -> Command {
    let mut command = convert(input, output, state);
    command
        .args(["--regrain", "8"])
        .args(["--chunk-encoding", "ffv1"]);
    command
}

fn frame_hashes(ffmpeg: &str, path: &Path) -> Vec<String> {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-map", "0:v:0", "-f", "framemd5", "-"])
        .output()
        .expect("framemd5");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| line.rsplit(',').next().map(|hash| hash.trim().to_string()))
        .collect()
}

/// The one job the state database knows about, once it knows about one.
fn only_job(store: &Store) -> Option<JobRow> {
    let jobs = store.list_jobs(10).ok()?;
    match jobs.len() {
        1 => jobs.into_iter().next(),
        _ => None,
    }
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
/// merely a file of the right length.
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
/// because what each test concludes depends on which path actually ran.
fn executor_of(store: &Store, job: &str) -> VideoExecutor {
    let plan_json = store
        .job(job)
        .expect("read the job")
        .and_then(|row| row.plan_json)
        .unwrap_or_else(|| panic!("job {job} has no plan"));
    let plan: serde_json::Value = serde_json::from_str(&plan_json).expect("the plan is JSON");
    let value = plan
        .pointer("/video/executor")
        .cloned()
        .unwrap_or_else(|| panic!("job {job} has a plan with no executor in it"));
    serde_json::from_value(value).expect("the plan names an executor this build knows")
}

/// Length and modification time: the two things a re-encode cannot leave alone.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ChunkArtifact {
    len: u64,
    modified: Option<SystemTime>,
}

fn chunk_artifact(path: &Path) -> ChunkArtifact {
    let meta = std::fs::metadata(path)
        .unwrap_or_else(|err| panic!("chunk artifact {} is unreadable: {err}", path.display()));
    ChunkArtifact {
        len: meta.len(),
        modified: meta.modified().ok(),
    }
}

/// Waits until the plan has been committed, then kills the process.
///
/// Returns the job the killed run had recorded, or `None` if the process exited on
/// its own first — a kill before any durable stage existed would leave the test
/// unable to say anything about resume, so that is reported rather than assumed.
fn kill_once_planned(child: &mut Child, store: &Store) -> Option<JobRow> {
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        if let Some(job) = only_job(store) {
            let planned = store
                .resume_point(&job.id)
                .map(|point| point.is_done(Stage::Plan))
                .unwrap_or(false);
            if planned {
                let _ = child.kill();
                let _ = child.wait();
                return Some(job);
            }
        }
        if let Ok(Some(_)) = child.try_wait() {
            return None;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Waits until some — but deliberately not all — of the plan's chunks are committed,
/// then kills the process.
///
/// `total` is what an uninterrupted run produced, read from that run's own table.
/// Requiring `committed < total` is what guarantees the kill landed with work still to
/// do: a kill after the last chunk would leave the test asserting that nothing was
/// recomputed when there was nothing left to recompute.
fn kill_once_chunks(child: &mut Child, store: &Store, target: usize, total: usize) -> Option<JobRow> {
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if let Some(job) = only_job(store) {
            let committed = committed_chunks(store, &job.id).len();
            if committed >= target && committed < total {
                let _ = child.kill();
                let _ = child.wait();
                return Some(job);
            }
        }
        if let Ok(Some(_)) = child.try_wait() {
            return None;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// The single-pass path: no chunks, so resume is the stage results.
#[test]
fn a_killed_cli_run_resumes_instead_of_starting_over() {
    if !ffmpeg_available() {
        eprintln!("SKIPPED: no FFmpeg");
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    // Six times over: ~24 s of video, so the conversion takes long enough that the
    // kill reliably lands while it is working.
    let input = build_fixture_repeated(dir.path(), 6);

    // ---- a clean run, for the frame-for-frame comparison -------------------
    let reference_output = dir.path().join("reference.mkv");
    let reference_state = dir.path().join("reference.sqlite3");
    let status = convert(&input, &reference_output, &reference_state)
        .stdout(Stdio::null())
        .status()
        .expect("run the reference");
    assert!(status.success(), "the reference conversion failed");
    let reference_hashes = frame_hashes("ffmpeg", &reference_output);
    assert!(
        !reference_hashes.is_empty(),
        "the reference run produced no frames to compare against"
    );

    // ---- the run that is killed -------------------------------------------
    let output = dir.path().join("out.mkv");
    let state = dir.path().join("jobs.sqlite3");
    let store = Store::open(&state).expect("open the state database");
    let mut command = convert(&input, &output, &state);
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = command.spawn().expect("spawn the conversion");
    let Some(killed) = kill_once_planned(&mut child, &store) else {
        panic!(
            "the process was never killed with durable work behind it, so there is \
             nothing to resume from"
        );
    };
    assert_eq!(
        executor_of(&store, &killed.id),
        VideoExecutor::FfmpegSinglePass,
        "this test is about the path with no chunks in it; without --regrain the plan \
         must not claim the chunked executor"
    );

    // What survived the kill, from the database rather than from the console: the
    // stages whose result was committed are exactly what the next run may skip.
    // (Skipped stages are not counted here: `resume_point` treats them as done for
    // its own purposes, but nothing was ever computed for them to resume.)
    let done_before: Vec<Stage> = store
        .stage_rows(&killed.id)
        .expect("read the stage rows")
        .into_iter()
        .filter(|row| row.status == StageStatus::Done && row.result_json.is_some())
        .map(|row| row.stage)
        .collect();
    eprintln!(
        "killed after {} stage result(s) had been committed: {done_before:?}",
        done_before.len()
    );
    assert!(
        done_before.contains(&Stage::Plan),
        "the kill must land after the plan was committed, or the test proves nothing \
         about resume: {done_before:?}"
    );
    assert_eq!(
        killed.input, input,
        "the job row must name the input being converted"
    );

    // ---- and the run that finishes ----------------------------------------
    let resumed_log = dir.path().join("resumed.stdout");
    let log_file = std::fs::File::create(&resumed_log).expect("create the log file");
    let status = convert(&input, &output, &state)
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::null())
        .status()
        .expect("run the resumed conversion");
    assert!(
        status.success(),
        "the resumed conversion failed; a killed run must leave a job that can be \
         finished"
    );

    // The job id is a function of the conversion, not of the clock: the second run
    // must have found the first one's job rather than starting a second.
    let jobs = store.list_jobs(10).expect("list the jobs");
    assert_eq!(
        jobs.len(),
        1,
        "the CLI must key a job by what it is, not by when it started; the second run \
         created a job of its own: {:?}",
        jobs.iter().map(|job| job.id.clone()).collect::<Vec<_>>()
    );
    assert_eq!(jobs[0].id, killed.id, "the job id must not change between runs");

    // And it must have skipped what the first run had already committed, which is the
    // CLI-visible form of "it did not start over": every stage that was done before
    // the kill is reported as resumed.
    let console = std::fs::read_to_string(&resumed_log).expect("read the console log");
    let resumed_stages = console
        .lines()
        .filter(|line| line.contains("resumed"))
        .collect::<Vec<_>>();
    eprintln!("the resumed run reported:\n{}", resumed_stages.join("\n"));
    for stage in &done_before {
        assert!(
            console
                .lines()
                .any(|line| line.contains("resumed") && line.contains(stage.label())),
            "stage {} was committed before the kill and must have been resumed, but \
             the run's console output never says so:\n{console}",
            stage.label()
        );
    }

    // And the film must be the film: a resume that finishes but produces something
    // else is worse than one that redoes the work.
    let hashes = frame_hashes("ffmpeg", &output);
    assert_eq!(
        hashes, reference_hashes,
        "the film finished after a kill must be frame-for-frame what an undisturbed \
         run produces"
    );
}

/// The chunked path: resume means the chunks table, not the stage results.
#[test]
fn a_killed_cli_run_reuses_its_committed_chunks() {
    if !ffmpeg_available() {
        eprintln!("SKIPPED: no FFmpeg");
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_fixture(dir.path());

    // ---- a clean run: the film to compare against, and the chunk plan ------
    //
    // Its own directory, because the scratch tree is derived from the output path and
    // the reference run must not share chunk files with the run under test.
    let reference_dir = dir.path().join("reference");
    std::fs::create_dir_all(&reference_dir).expect("reference dir");
    let reference_output = reference_dir.join("out.mkv");
    let reference_state = reference_dir.join("jobs.sqlite3");
    let status = convert_chunked(&input, &reference_output, &reference_state)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("run the reference");
    assert!(status.success(), "the reference conversion failed");
    let reference_hashes = frame_hashes("ffmpeg", &reference_output);
    let reference_store = Store::open(&reference_state).expect("reference store");
    let reference_job = only_job(&reference_store).expect("the reference run recorded one job");
    assert_eq!(
        executor_of(&reference_store, &reference_job.id),
        VideoExecutor::Chunked,
        "--regrain must select the chunked executor, or there are no per-chunk \
         checkpoints for this test to interrupt"
    );
    let reference_chunks = committed_chunks(&reference_store, &reference_job.id);
    assert!(
        reference_chunks.len() >= 3,
        "the fixture must produce several chunks for an interruption to mean \
         anything, got {:?}",
        reference_chunks.keys().collect::<Vec<_>>()
    );
    let reference_spans = chunk_spans(&reference_store, &reference_job.id);

    // ---- the run that is killed partway -----------------------------------
    let job_dir = dir.path().join("job");
    std::fs::create_dir_all(&job_dir).expect("job dir");
    let output = job_dir.join("out.mkv");
    let state = job_dir.join("jobs.sqlite3");
    let store = Store::open(&state).expect("open the state database");
    let mut command = convert_chunked(&input, &output, &state);
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = command.spawn().expect("spawn the conversion");
    let Some(killed) = kill_once_chunks(&mut child, &store, 2, reference_chunks.len()) else {
        panic!(
            "the process was never killed with at least two chunks committed and at \
             least one still to do, so there is nothing to conclude about reusing them"
        );
    };

    // What survived the kill. This is the table, not a log: the commit happens before
    // the run continues, so a row here means the file it names was complete on disk
    // when the process stopped.
    let committed_before = committed_chunks(&store, &killed.id);
    for (index, path) in &committed_before {
        eprintln!(
            "  killed with chunk {index:04} committed: {}",
            path.display()
        );
    }
    assert!(
        committed_before.len() >= 2,
        "the kill must leave more than one committed chunk, or reusing one file \
         proves nothing: {:?}",
        committed_before.keys().collect::<Vec<_>>()
    );
    assert!(
        committed_before.len() < reference_chunks.len(),
        "the kill landed after the last chunk ({} of {}), so nothing was left to \
         recompute and the test could not tell reuse from a fresh run",
        committed_before.len(),
        reference_chunks.len()
    );
    let before: BTreeMap<u32, ChunkArtifact> = committed_before
        .iter()
        .map(|(index, path)| (*index, chunk_artifact(path)))
        .collect();

    // Everything the resumed run writes has to be written after this instant.
    let resumed_started = SystemTime::now();

    // ---- and the run that finishes ----------------------------------------
    let status = convert_chunked(&input, &output, &state)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("run the resumed conversion");
    assert!(
        status.success(),
        "the resumed conversion failed; a killed run must leave a job that can be \
         finished"
    );

    // The job id is still a function of the conversion: the resumed run has to find
    // the killed run's chunks, which are keyed by job id and live in its scratch tree.
    let jobs = store.list_jobs(10).expect("list the jobs");
    assert_eq!(
        jobs.len(),
        1,
        "the chunked resume also depends on the job id recurring; the second run \
         created a job of its own: {:?}",
        jobs.iter().map(|job| job.id.clone()).collect::<Vec<_>>()
    );

    let after = committed_chunks(&store, &killed.id);
    let after_keys: BTreeSet<u32> = after.keys().copied().collect();
    let reference_keys: BTreeSet<u32> = reference_chunks.keys().copied().collect();
    eprintln!(
        "reference chunks {:?}, committed before the kill {:?}, committed after the \
         resumed run {:?}",
        reference_keys.iter().collect::<Vec<_>>(),
        committed_before.keys().collect::<Vec<_>>(),
        after_keys.iter().collect::<Vec<_>>()
    );

    // (1) A chunk that was already committed must not be computed again. A re-encoded
    // chunk rewrites its artifact, so the file that was durable before the kill must
    // still name the same path and still have the same length and modification time.
    for (index, was) in &before {
        let path = after.get(index).unwrap_or_else(|| {
            panic!(
                "chunk {index} was committed before the kill and is missing from the \
                 table now"
            )
        });
        assert_eq!(
            Some(path),
            committed_before.get(index),
            "chunk {index} must still name the artifact it committed"
        );
        assert_eq!(
            chunk_artifact(path),
            *was,
            "chunk {index} was recomputed by the resumed run ({} changed), which was \
             already committed before the kill; the chunks are not being reused",
            path.display()
        );
    }

    // (2) The chunks that were *not* committed are exactly the ones the resumed run had
    // to produce. Each of their artifacts must therefore have been written by it, which
    // is what the modification time says; and the set must be the whole plan, so
    // nothing was silently skipped.
    assert_eq!(
        after_keys, reference_keys,
        "the killed and resumed runs together must cover exactly the chunks a single \
         run covers"
    );
    assert_eq!(
        chunk_spans(&store, &killed.id),
        reference_spans,
        "the resumed run must split the film where an uninterrupted run did"
    );
    let recomputed: Vec<u32> = after_keys
        .difference(&committed_before.keys().copied().collect())
        .copied()
        .collect();
    assert!(
        !recomputed.is_empty(),
        "the kill left every chunk committed, so this run recomputed nothing and the \
         test proved nothing"
    );
    for index in &recomputed {
        let path = after.get(index).expect("index is in the table");
        let artifact = chunk_artifact(path);
        assert!(
            artifact.modified.map(|at| at >= resumed_started).unwrap_or(false),
            "chunk {index} was not committed before the kill, so the resumed run had \
             to write it, but {} predates the resumed run ({:?} < {resumed_started:?})",
            path.display(),
            artifact.modified
        );
    }
    eprintln!("the resumed run had to produce chunks {recomputed:?}");

    // (3) And the film must be the film: a resume that finishes but produces something
    // else is worse than one that redoes the work.
    let hashes = frame_hashes("ffmpeg", &output);
    assert_eq!(
        hashes.len(),
        reference_hashes.len(),
        "the resumed output must have the same number of frames"
    );
    assert_eq!(
        hashes, reference_hashes,
        "the film finished after a kill must be frame-for-frame what an undisturbed \
         run produces"
    );
}
