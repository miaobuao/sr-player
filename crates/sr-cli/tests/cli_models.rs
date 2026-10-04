//! The model stages, end to end, through the real command line.
//!
//! `docs/verification.md` records that these runs happened. This asserts that they
//! still do. Without it, every claim about Phase 2 and Phase 3 rests on a document
//! and a refactor could regress both silently — which is exactly what a frame count
//! and a duration would fail to notice.
//!
//! Two properties are checked, and both are chosen because a *plausible* wrong
//! implementation satisfies the obvious ones:
//!
//! * **the frame count is exactly `m*(n-1)+1`.** A pass that dropped or duplicated a
//!   frame at a segment boundary would be off by one here, and the boundary is the
//!   place the segment planner and the frame carry are most likely to disagree.
//! * **no two adjacent output frames are identical.** A 2x pass that repeated frames
//!   instead of synthesising them produces the right count, the right duration and a
//!   clean decode. Only comparing consecutive frames tells interpolation from
//!   duplication, and that is the regression worth a test.
//!
//! Skips itself, loudly, when FFmpeg, the model weights or a Vulkan device are
//! absent — a test that cannot run should say why rather than pass.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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

/// Why this test cannot run here, or `None` if it can.
fn unavailable() -> Option<String> {
    if !ffmpeg_available() {
        return Some("ffmpeg is not on PATH".into());
    }
    if !sr_core::ai::models_installed() {
        return Some(format!(
            "the model weights are not installed (looked in {})",
            sr_core::ai::models_root().display()
        ));
    }
    if sr_core::ai::preferred_device().is_none() {
        return Some("no Vulkan device can host a model".into());
    }
    None
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sr-cli-models-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the scratch directory");
    dir
}

/// A short clip with real motion, so RIFE has something to interpolate.
///
/// Deliberately small: this test runs two models, and its job is to prove the path
/// works rather than to measure it. `docs/verification.md` carries the full-size
/// numbers.
fn build_fixture(dir: &Path) -> PathBuf {
    let input = dir.join("input.mkv");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=24",
            "-t",
            "0.5",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .expect("build the fixture");
    assert!(status.success(), "ffmpeg could not build the fixture");
    input
}

/// Per-frame hashes, which give both the frame count and the ability to tell
/// whether two frames are the same picture.
fn frame_hashes(path: &Path) -> Vec<String> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-map", "0:v:0", "-f", "framemd5", "-"])
        .output()
        .expect("run framemd5");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| line.rsplit(',').next().map(|hash| hash.trim().to_string()))
        .collect()
}

fn run(args: &[&str]) -> (bool, String) {
    let output = Command::new(cli())
        .args(args)
        .output()
        .expect("run sr-cli");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), text)
}

/// Runs a conversion and asserts the two properties above.
fn convert_and_check(dir: &Path, input: &Path, name: &str, profile: &str) {
    let output = dir.join(format!("{name}.mkv"));
    let state = dir.join(format!("{name}.db"));
    let (ok, log) = run(&[
        "--state-db",
        state.to_str().unwrap(),
        "convert",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "--profile",
        profile,
        "--interpolate",
        "rife",
        "--no-audio",
        "--no-resume",
    ]);
    assert!(ok, "[{name}] the conversion failed:\n{log}");
    assert!(output.exists(), "[{name}] no output was written");

    let before = frame_hashes(input);
    let after = frame_hashes(&output);
    assert!(!before.is_empty(), "[{name}] the input has no frames");

    // m*(n-1)+1, with m = 2. An off-by-one at a segment boundary shows up here and
    // almost nowhere else.
    let expected = 2 * (before.len() - 1) + 1;
    assert_eq!(
        after.len(),
        expected,
        "[{name}] {} input frames at 2x must give {expected} output frames, got {}",
        before.len(),
        after.len()
    );

    // The property that separates interpolation from duplication.
    let repeated = after.windows(2).filter(|pair| pair[0] == pair[1]).count();
    assert_eq!(
        repeated,
        0,
        "[{name}] {repeated} of {} adjacent frame pairs are identical, so frames were \
         duplicated rather than synthesised",
        after.len() - 1
    );
}

#[test]
fn rife_interpolation_runs_through_the_command_line() {
    if let Some(reason) = unavailable() {
        eprintln!("SKIPPED rife_interpolation_runs_through_the_command_line: {reason}");
        return;
    }
    let dir = scratch("rife");
    let input = build_fixture(&dir);
    convert_and_check(&dir, &input, "rife", "deterministic");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn restoration_and_interpolation_run_together() {
    if let Some(reason) = unavailable() {
        eprintln!("SKIPPED restoration_and_interpolation_run_together: {reason}");
        return;
    }
    let dir = scratch("both");
    let input = build_fixture(&dir);
    // `safe_16gb` is the profile that asks for both models.
    convert_and_check(&dir, &input, "both", "safe-16gb");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A longer fixture, because a resume needs several chunks and a kill that lands.
///
/// Four *distinct* sources, three seconds each. Distinct because chunks are cut at
/// shot boundaries: a single twelve-second source is one shot and therefore one
/// chunk, which would make the kill land after all the work was already done. The
/// first version of this test did exactly that and reported "1 chunk reused of 1",
/// which looks like a pass and tests nothing.
///
/// The model path is the point. Resume has been tested without a model since the
/// beginning; the completion bar asks for it *with* real restoration and
/// interpolation, where the segment boundaries and the frame carry actually live.
fn build_long_fixture(dir: &Path) -> PathBuf {
    const SOURCES: [&str; 4] = [
        "testsrc2=size=320x240:rate=24",
        "mandelbrot=size=320x240:rate=24",
        "gradients=size=320x240:rate=24",
        "testsrc=size=320x240:rate=24",
    ];
    let input = dir.join("long-input.mkv");
    let mut command = Command::new("ffmpeg");
    command.args(["-y", "-hide_banner", "-loglevel", "error"]);
    let mut labels = String::new();
    for (index, source) in SOURCES.iter().enumerate() {
        command.args(["-t", "3", "-f", "lavfi", "-i", source]);
        labels.push_str(&format!("[{index}:v]"));
    }
    let status = command
        .args([
            "-filter_complex",
            &format!("{labels}concat=n={}:v=1:a=0[v]", SOURCES.len()),
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .expect("build the long fixture");
    assert!(status.success(), "ffmpeg could not build the long fixture");
    input
}

/// The job the state database knows about, once it knows about exactly one.
fn only_job(store: &sr_core::state::Store) -> Option<sr_core::state::JobRow> {
    let jobs = store.list_jobs(10).ok()?;
    match jobs.len() {
        1 => jobs.into_iter().next(),
        _ => None,
    }
}

fn committed_chunk_count(store: &sr_core::state::Store, job: &str) -> usize {
    store
        .chunks(job)
        .map(|rows| {
            rows.into_iter()
                .filter(|row| {
                    row.stage == sr_core::events::Stage::Encode && row.status == "committed"
                })
                .count()
        })
        .unwrap_or(0)
}

/// The completion bar's forced-termination requirement, with a model in the path.
///
/// Three runs: one uninterrupted, one killed once some but not all chunks are
/// committed, and one that resumes. The resumed output must be byte-identical to the
/// uninterrupted one — a resume that silently produced a different file would survive
/// a test that only checked the file existed.
#[test]
fn a_killed_model_run_resumes_to_the_same_bytes() {
    if let Some(reason) = unavailable() {
        eprintln!("SKIPPED a_killed_model_run_resumes_to_the_same_bytes: {reason}");
        return;
    }
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    let dir = scratch("resume");
    let input = build_long_fixture(&dir);
    let convert = |name: &str| {
        let mut command = Command::new(cli());
        command
            .arg("--state-db")
            .arg(dir.join(format!("{name}.db")))
            .arg("convert")
            .arg(&input)
            .arg("-o")
            .arg(dir.join(format!("{name}.mkv")))
            .args(["--profile", "deterministic", "--interpolate", "rife"])
            .args(["--no-audio", "--no-resume"]);
        command
    };

    // ---- the uninterrupted reference -------------------------------------
    let (ok, log) = run(&[
        "--state-db",
        dir.join("ref.db").to_str().unwrap(),
        "convert",
        input.to_str().unwrap(),
        "-o",
        dir.join("ref.mkv").to_str().unwrap(),
        "--profile",
        "deterministic",
        "--interpolate",
        "rife",
        "--no-audio",
        "--no-resume",
    ]);
    assert!(ok, "the reference run failed:\n{log}");
    let reference = frame_hashes(&dir.join("ref.mkv"));
    // Without this, two missing files would both hash to nothing, compare equal, and
    // the test would pass while proving nothing. That failure mode has already cost
    // this project once.
    assert!(
        !reference.is_empty(),
        "the reference run produced no frames to compare against"
    );
    // How many chunks an uninterrupted run makes, so the kill can be aimed at the
    // middle of the job rather than wherever a fixed count happens to land. The
    // completion bar says "late in processing", and killing after the first of four
    // chunks is not that.
    let reference_store =
        sr_core::state::Store::open(&dir.join("ref.db")).expect("reference store");
    let total_chunks = only_job(&reference_store)
        .map(|job| committed_chunk_count(&reference_store, &job.id))
        .unwrap_or(0);
    assert!(
        total_chunks >= 3,
        "the fixture must produce several chunks for an interruption to mean anything, got \
         {total_chunks}"
    );
    let target = (total_chunks / 2).max(1);

    // ---- the run that gets killed ----------------------------------------
    let store_path = dir.join("killed.db");
    let store = sr_core::state::Store::open(&store_path).expect("open the store");
    let mut command = convert("killed");
    command
        .arg("--keep-intermediates")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = command.spawn().expect("spawn the run to kill");

    // Kill partway through: at least `target` chunks committed, and at least one
    // still to do. Requiring the latter is what guarantees the kill landed with work
    // remaining -- a kill after the last chunk would have the test assert that
    // nothing was recomputed when there was nothing left to recompute.
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut killed_job = None;
    loop {
        if let Some(job) = only_job(&store) {
            let committed = committed_chunk_count(&store, &job.id);
            if committed >= target && committed < total_chunks {
                let _ = child.kill();
                let _ = child.wait();
                killed_job = Some(job);
                break;
            }
        }
        if let Ok(Some(_)) = child.try_wait() {
            break;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }

    let Some(job) = killed_job else {
        let _ = std::fs::remove_dir_all(&dir);
        panic!(
            "the run finished before any chunk was committed, so this test cannot say \
             anything about resume. The fixture needs to be longer or the machine slower."
        );
    };
    let committed_before = committed_chunk_count(&store, &job.id);
    assert!(
        !dir.join("killed.mkv").exists(),
        "a killed run must not have published an output"
    );

    // ---- resume ----------------------------------------------------------
    // No `--no-resume`, and the same state database and job identity.
    let (ok, log) = run(&[
        "--state-db",
        store_path.to_str().unwrap(),
        "convert",
        input.to_str().unwrap(),
        "-o",
        dir.join("killed.mkv").to_str().unwrap(),
        "--profile",
        "deterministic",
        "--interpolate",
        "rife",
        "--no-audio",
    ]);
    assert!(ok, "the resumed run failed:\n{log}");

    let after = committed_chunk_count(&store, &job.id);
    assert!(
        after >= committed_before && committed_before >= 1,
        "the resume discarded committed work: {committed_before} chunks before, {after} after"
    );

    // The count above would also hold if the resume had recomputed every chunk from
    // scratch, so the claim the completion bar actually makes -- resume *without
    // recomputing committed chunks* -- has to be read off the run's own account of
    // itself. The runner reports how many it reused; that number must be non-zero.
    let reused = log
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(3)
        .find(|window| window[1] == "reused" && window[2] == "from")
        .and_then(|window| window[0].parse::<usize>().ok())
        .unwrap_or_else(|| panic!("the resumed run did not report chunk reuse:\n{log}"));
    assert!(
        reused > 0,
        "the resume committed {committed_before} chunk(s) before it was killed and reused none \
         of them:\n{log}"
    );
    eprintln!("resumed with {reused} chunk(s) reused of {after} total");

    let resumed = frame_hashes(&dir.join("killed.mkv"));
    assert_eq!(
        resumed.len(),
        reference.len(),
        "the resumed output has {} frames against the reference's {}",
        resumed.len(),
        reference.len()
    );
    assert_eq!(
        resumed, reference,
        "the resumed run produced different frames from the uninterrupted one"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
