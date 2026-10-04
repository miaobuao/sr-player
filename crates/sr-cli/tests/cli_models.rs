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
