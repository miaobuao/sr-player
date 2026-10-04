//! The engine driving a real GPU backend.
//!
//! `native_execution.rs` proves the native executor runs *a* plugin; this proves
//! it runs a backend that actually computes on the GPU, through the published ABI,
//! loaded the way a user would load it (`SR_INFER_PLUGIN`).
//!
//! The distinction matters because the two halves were developed separately: the
//! engine can be right about the protocol while the backend is wrong about the
//! contract, and the failure mode is quiet — the frame count is correct, the
//! project's own checks pass, and the file is full of blank frames. That happened
//! once (the backend filled only the synthesised slots and left the pass-through
//! frames zeroed), which is why this test looks at the *pixels* of the committed
//! chunk rather than only at its length.
//!
//! It skips itself when FFmpeg or Vulkan is unavailable.

use sr_core::events::{Event, Stage, StageStatus};
use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::infer::EngineRegistry;
use sr_core::media::probe;
use sr_core::pipeline::native::ChunkEncoding;
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::{InterpolationMethod, RestorationProfile};
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::EventBus;
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

fn gpu_plugin() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?.parent()?;
    let name = if cfg!(windows) {
        "sr_infer_gpu.dll"
    } else if cfg!(target_os = "macos") {
        "libsr_infer_gpu.dylib"
    } else {
        "libsr_infer_gpu.so"
    };
    [dir.join(name), dir.join("deps").join(name)]
        .into_iter()
        .find(|path| path.exists())
}

/// Two shots at 24 fps, moving content throughout so a duplicated frame and a
/// synthesised one cannot be confused.
fn build_fixture(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let input = dir.join("gpu-input.mkv");
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
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "16",
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
    let mut argv = args(&["-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(path.display().to_string());
    argv.extend(args(&["-map", "0:v:0", "-f", "framemd5", "-"]));
    let text = capture(&ff.ffmpeg, &argv).expect("framemd5");
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| line.rsplit(',').next().map(|h| h.trim().to_string()))
        .collect()
}

fn call_log_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|text| text.lines().map(str::to_string).collect())
        .unwrap_or_default()
}

#[test]
fn the_engine_runs_the_vulkan_backend_over_a_real_file() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let Some(plugin) = gpu_plugin() else {
        eprintln!("SKIPPED: the GPU backend was not built");
        return;
    };
    if EngineRegistry::probe(Arc::clone(&ff)).engines().is_empty() {
        eprintln!("SKIPPED: no engines");
        return;
    }
    std::env::set_var("SR_INFER_PLUGIN", &plugin);

    let dir = tempfile::tempdir().expect("temp dir");
    let call_log = dir.path().join("gpu-calls.jsonl");
    std::env::set_var("SR_INFER_CALL_LOG", &call_log);

    let input = build_fixture(&ff, dir.path());
    let output = dir.path().join("gpu-output.mkv");
    let source = probe(&ff, &input).expect("probe the fixture");
    let source_frames = source
        .primary_video()
        .and_then(|video| video.base.nb_frames)
        .unwrap_or(48);
    assert_eq!(source_frames, 48, "two one-second shots at 24 fps");

    let engines = Arc::new(EngineRegistry::probe(Arc::clone(&ff)));
    let model = engines
        .model_engine()
        .expect("the GPU backend must be discovered as a model engine");
    let caps = model.capabilities();
    assert_eq!(model.id(), "inference-plugin");
    assert!(
        caps.interpolate,
        "the backend must advertise interpolation"
    );
    assert!(
        !caps.restore,
        "this backend does not restore, and the engine must not be told it does"
    );
    assert!(
        !caps.devices.is_empty(),
        "the backend must report at least one Vulkan device"
    );
    eprintln!("backend: {} | {}", caps.model, caps.devices.join("; "));

    let store = Arc::new(Store::open(&dir.path().join("jobs.sqlite3")).expect("state store"));
    let bus = EventBus::new();
    let events = bus.subscribe();
    let runner = PipelineRunner::new(
        Arc::clone(&ff),
        engines,
        bus.clone(),
        Arc::clone(&store),
        Arc::new(AtomicBool::new(false)),
        dir.path().join("scratch"),
    );

    let mut profile = RestorationProfile::deterministic();
    // The backend has no restoration, so ask only for what it can do: the plan
    // must resolve to the native executor for interpolation and report the
    // absence of restoration rather than inventing it.
    profile.interpolation.method = InterpolationMethod::Plugin;
    profile.interpolation.multiplier = 2;
    profile.restoration.enabled = false;
    profile.output.prefer_hardware = false;
    profile.output.quality = 28;

    let request = PlanRequest {
        job_id: "gpu".into(),
        input: input.clone(),
        output: output.clone(),
        profile,
    };
    let outcome = runner
        .run(
            request,
            &RunnerOptions {
                resume: true,
                // Keep the lossless chunks: they are what the pixels are checked
                // against, without an encoder's decisions in between.
                keep_intermediates: true,
                dry_run: false,
                max_degrade_retries: 2,
                chunk_encoding: ChunkEncoding::LosslessIntermediate,
            },
        )
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
        "the GPU pipeline failed: {}\n{}",
        outcome.message,
        logs.iter()
            .filter(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("gpu") || lower.contains("chunk") || lower.contains("fail")
            })
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        stages.iter().any(|(stage, status)| *stage == Stage::Qc
            && *status == StageStatus::Done),
        "quality control must pass, stages seen: {stages:?}"
    );

    // ---- the backend was actually called, once per shot at least ----------
    let calls = call_log_lines(&call_log);
    assert!(
        !calls.is_empty(),
        "the GPU backend logged no calls:\n{}",
        logs.join("\n")
    );
    let chunks: std::collections::HashSet<&str> = calls
        .iter()
        .filter_map(|line| {
            let start = line.find("\"chunk\":")? + 8;
            Some(line[start..].split(',').next()?.trim())
        })
        .collect();
    assert!(
        chunks.len() >= 2,
        "each shot is its own chunk, saw chunk ids {chunks:?}"
    );

    // ---- no call ever straddled the cut at 1.0 s --------------------------
    for line in &calls {
        let field = |key: &str| -> Option<f64> {
            let needle = format!("\"{key}\":");
            let start = line.find(&needle)? + needle.len();
            line[start..].split(',').next()?.trim().parse().ok()
        };
        let (first, last) = (field("first_pts"), field("last_pts"));
        if let (Some(first), Some(last)) = (first, last) {
            assert!(
                !(first < 1.0 - 1e-6 && last > 1.0 + 1e-6),
                "a GPU call covered the cut: {first:.4}s..{last:.4}s in {line}"
            );
        }
    }

    // ---- the frames are real, not blank and not duplicated ----------------
    let chunk_dir = dir.path().join("scratch").join("job-gpu").join("chunks");
    let mut chunk_files: Vec<PathBuf> = std::fs::read_dir(&chunk_dir)
        .expect("chunk directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|ext| ext == "mkv").unwrap_or(false))
        .collect();
    chunk_files.sort();
    let chunk = chunk_files.first().expect("at least one chunk file").clone();

    // Decoded to grayscale, so the measurements are about pixels rather than
    // about which of the two encoders wrote the file.
    let mut argv = args(&["-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(chunk.display().to_string());
    argv.extend(args(&[
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]));
    let raw = capture(&ff.ffmpeg, &argv).expect("decode the chunk");
    let bytes = raw.as_bytes();
    let frame_bytes = 320 * 240;
    assert!(
        bytes.len() >= frame_bytes,
        "the chunk decoded to {} bytes, less than a frame",
        bytes.len()
    );
    let frame_count = bytes.len() / frame_bytes;
    let mut means = Vec::with_capacity(frame_count);
    let mut fingerprints = Vec::with_capacity(frame_count);
    for index in 0..frame_count {
        let frame = &bytes[index * frame_bytes..(index + 1) * frame_bytes];
        let sum: u64 = frame.iter().map(|byte| *byte as u64).sum();
        means.push(sum as f64 / frame_bytes as f64);
        // FNV-1a over the frame: exact, and cheap enough for a test.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in frame {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        fingerprints.push(hash);
    }

    let blank: Vec<usize> = means
        .iter()
        .enumerate()
        .filter(|(_, mean)| **mean < 4.0)
        .map(|(index, _)| index)
        .collect();
    assert!(
        blank.is_empty(),
        "frames {blank:?} are blank: the backend left the pass-through slots unwritten"
    );
    let distinct: std::collections::HashSet<&u64> = fingerprints.iter().collect();
    assert!(
        distinct.len() > 24,
        "the chunk holds {} distinct frames for the 24 the shot contains: nothing was synthesised",
        distinct.len()
    );
    assert!(
        distinct.len() * 10 >= fingerprints.len() * 9,
        "only {} of {} frames are distinct: the interpolation is mostly repetition",
        distinct.len(),
        fingerprints.len()
    );

    // ---- and the published file has the planned frame count ---------------
    let result = probe(&ff, &output).expect("probe the output");
    let fps = result
        .primary_video()
        .and_then(|video| video.fps())
        .map(|f| f.to_f64())
        .unwrap_or(0.0);
    assert!(
        (fps - 48.0).abs() < 0.1,
        "the output must run at twice the source rate, got {fps:.3} fps"
    );
    assert!(
        result.audio.is_empty() || result.audio.len() == source.audio.len(),
        "audio must survive"
    );
}
