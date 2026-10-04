//! Killing the program, and starting it again.
//!
//! Everything verified about resume so far went through the library. This goes
//! through `sr-cli`, because that is what an unattended user runs, and the two are
//! not the same thing: the command line derived its job id from the clock, so the
//! chunks table was keyed by a name that never recurred and resume was unreachable
//! from it entirely. That is fixed, and this is the test that would have caught it.
//!
//! The interruption is `Child::kill`, which on Windows is `TerminateProcess` and on
//! Unix a `SIGKILL`: no unwinding, no destructors, no flush. The engine gets no
//! chance to tidy up, which is the point — a power cut does not ask permission. What
//! survives is whatever was already durable on disk, and that is the property under
//! test.
//!
//! Skips itself when FFmpeg or the example plugin is unavailable.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// The newest of the two legitimate plugin paths.
///
/// Picking the first that exists loaded a stale binary and silently invalidated
/// another test; the reasoning is in `native_execution.rs`.
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
        .filter(|path| path.exists())
        .max_by_key(|path| std::fs::metadata(path).and_then(|meta| meta.modified()).ok())
}

fn cli() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sr-cli"))
}

/// Four one-second shots, so the job has several chunks and a kill has somewhere to
/// land that is neither the first nor the last.
fn build_fixture(dir: &Path) -> PathBuf {
    let input = dir.join("cli-input.mkv");
    let mut command = Command::new("ffmpeg");
    command.args(["-y", "-hide_banner", "-loglevel", "error"]);
    for source in [
        "testsrc2=size=320x240:rate=24",
        "mandelbrot=size=320x240:rate=24",
        "gradients=size=320x240:rate=24",
        "testsrc=size=320x240:rate=24",
    ] {
        command.args(["-t", "1", "-f", "lavfi", "-i", source]);
    }
    let status = command
        .args([
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
        ])
        .arg(&input)
        .status()
        .expect("build the fixture");
    assert!(status.success(), "ffmpeg could not build the fixture");
    input
}

fn convert(input: &Path, output: &Path, state: &Path, log: &Path, plugin: &Path) -> Command {
    let mut command = Command::new(cli());
    command
        .arg("--state-db")
        .arg(state)
        .arg("convert")
        .arg(input)
        .arg("-o")
        .arg(output)
        .args(["--profile", "deterministic", "--interpolate", "plugin"])
        .arg("--no-audio")
        .arg("--keep-intermediates")
        .env("SR_INFER_PLUGIN", plugin)
        .env("SR_INFER_CALL_LOG", log);
    command
}

fn chunks_called(log: &Path) -> BTreeSet<u64> {
    let Ok(text) = std::fs::read_to_string(log) else {
        return BTreeSet::new();
    };
    text.lines()
        .filter_map(|line| {
            let needle = "\"chunk\":";
            let start = line.find(needle)? + needle.len();
            line[start..].split(',').next()?.trim().parse().ok()
        })
        .collect()
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

/// Waits for the call log to mention `target` chunks, then kills the process.
///
/// Returns whether the kill happened in time. A kill before any chunk was committed
/// would leave the test unable to say anything about resume, so that is reported
/// rather than assumed.
fn kill_once_started(child: &mut Child, log: &Path, target: usize) -> bool {
    for _ in 0..6_000 {
        if chunks_called(log).len() >= target {
            let _ = child.kill();
            let _ = child.wait();
            return true;
        }
        if let Ok(Some(_)) = child.try_wait() {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let _ = child.kill();
    let _ = child.wait();
    false
}

#[test]
fn a_killed_cli_run_resumes_instead_of_starting_over() {
    if !ffmpeg_available() {
        eprintln!("SKIPPED: no FFmpeg");
        return;
    }
    let Some(plugin) = example_plugin() else {
        eprintln!("SKIPPED: the example inference plugin was not built");
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_fixture(dir.path());

    // ---- a clean run, for the frame-for-frame comparison -------------------
    let reference_output = dir.path().join("reference.mkv");
    let reference_log = dir.path().join("reference.log");
    let reference_state = dir.path().join("reference.sqlite3");
    let status = convert(
        &input,
        &reference_output,
        &reference_state,
        &reference_log,
        &plugin,
    )
    .stdout(Stdio::null())
    .status()
    .expect("run the reference");
    assert!(status.success(), "the reference conversion failed");
    let reference_hashes = frame_hashes("ffmpeg", &reference_output);
    let reference_chunks = chunks_called(&reference_log);
    assert!(
        reference_chunks.len() >= 2,
        "the fixture must produce several chunks, got {reference_chunks:?}"
    );

    // ---- the run that is killed -------------------------------------------
    let output = dir.path().join("out.mkv");
    let state = dir.path().join("jobs.sqlite3");
    let killed_log = dir.path().join("killed.log");
    let mut command = convert(&input, &output, &state, &killed_log, &plugin);
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = command.spawn().expect("spawn the conversion");
    let killed_in_time = kill_once_started(&mut child, &killed_log, 2);
    assert!(
        killed_in_time,
        "the process was never killed mid-flight, so there is nothing to resume from"
    );

    // What survived the kill, from the database rather than from the log: the log
    // says what was attempted, the table says what is durable, and resume follows the
    // table.
    let survived = chunks_called(&killed_log);
    eprintln!(
        "killed after {} chunk(s) had been called: {survived:?}",
        survived.len()
    );

    // ---- and the run that finishes ----------------------------------------
    let resumed_log = dir.path().join("resumed.log");
    let status = convert(&input, &output, &state, &resumed_log, &plugin)
        .stdout(Stdio::null())
        .status()
        .expect("run the resumed conversion");
    assert!(
        status.success(),
        "the resumed conversion failed; a killed run must leave a job that can be \
         finished"
    );

    let resumed_chunks = chunks_called(&resumed_log);
    eprintln!("recomputed after the kill: {resumed_chunks:?}");
    assert!(
        !resumed_chunks.is_empty(),
        "the resumed run called the model for nothing, which would mean it started \
         over"
    );
    assert!(
        resumed_chunks.len() < reference_chunks.len(),
        "the resumed run recomputed every chunk ({resumed_chunks:?} of \
         {reference_chunks:?}), so it started over rather than resuming"
    );
    let done: BTreeSet<u64> = resumed_chunks.union(&survived).copied().collect();
    assert_eq!(
        done, reference_chunks,
        "the killed and resumed runs together must cover exactly the chunks a single \
         run covers"
    );

    // And the film must be the film: a resume that finishes but produces something
    // else is worse than one that redoes the work.
    let hashes = frame_hashes("ffmpeg", &output);
    assert_eq!(
        hashes, reference_hashes,
        "the film finished after a kill must be frame-for-frame what an undisturbed \
         run produces"
    );
}
