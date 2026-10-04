//! Format and field-order regression cases.
//!
//! Continuing the corpus started in `cadence_corpus.rs`. Three shapes that a
//! restoration job meets constantly and that the pipeline handles with different
//! code paths, none of which had been run on real files:
//!
//! * **anamorphic** — 720x480 with a non-square pixel aspect. A model must see
//!   square pixels, the target raster has to be derived from the *display* shape,
//!   and the output must not come out squeezed;
//! * **true interlaced** — 60 fields per second of real motion, which must be
//!   deinterlaced and must **not** be decimated. This is the counterpart of the
//!   telecine case: the two are easy to confuse and the consequences are
//!   opposite;
//! * **PAL 50i** — the same shape at 50 fields, where the temptation is to
//!   "normalise" 25 fps to 24 and silently change the runtime, the pitch, the
//!   subtitles and the chapter positions.
//!
//! Skips itself when FFmpeg is unavailable.

use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::infer::EngineRegistry;
use sr_core::media::classify::{classify, ClassifyOptions, TemporalMode};
use sr_core::media::probe;
use sr_core::media::scene::{detect_scenes, SceneDecode, SceneOptions};
use sr_core::pipeline::plan::{build_plan, cadence_for, PlanRequest};
use sr_core::pipeline::profile::{InterpolationMethod, RestorationProfile};
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::{EventBus, Reporter};
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

fn build(ff: &Ffmpeg, output: &Path, arguments: Vec<String>) {
    let mut argv = args(&["-y", "-hide_banner", "-loglevel", "error"]);
    argv.extend(arguments);
    argv.push(output.display().to_string());
    capture(&ff.ffmpeg, &argv).unwrap_or_else(|err| panic!("build {}: {err}", output.display()));
}

/// 720x480 at 24 fps with a 32:27 pixel aspect — an NTSC 16:9 DVD frame.
fn build_anamorphic(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let path = dir.join("anamorphic.mkv");
    build(
        ff,
        &path,
        args(&[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=720x480:rate=24:duration=1",
            "-vf",
            "setsar=32/27",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
        ]),
    );
    path
}

/// Real interlaced video: 60 distinct moments per second woven into 30 frames.
///
/// `tinterlace=interleave_top` takes two consecutive frames and puts one in each
/// field, so the result has genuine field-to-field motion rather than being a
/// progressive frame flagged as interlaced. That distinction is the whole point:
/// a flagged-progressive fixture would let a pipeline that ignores the flags look
/// correct.
fn build_interlaced(ff: &Ffmpeg, dir: &Path, rate: &str, out_rate: &str, name: &str) -> PathBuf {
    let path = dir.join(name);
    build(
        ff,
        &path,
        args(&[
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size=320x240:rate={rate}:duration=1"),
            "-vf",
            "tinterlace=interleave_top",
            "-flags",
            "+ilme+ildct",
            "-r",
            out_rate,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
        ]),
    );
    path
}

fn frames(ff: &Ffmpeg, path: &Path) -> u64 {
    let mut argv = args(&["-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(path.display().to_string());
    argv.extend(args(&["-map", "0:v:0", "-f", "framemd5", "-"]));
    capture(&ff.ffmpeg, &argv)
        .expect("framemd5")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .count() as u64
}

struct Fixture {
    plan: sr_core::pipeline::plan::ConversionPlan,
    input: PathBuf,
}

/// Classifies, finds the shots through the cadence chain, and builds the real
/// plan — the same three stages the runner runs, without touching pixels.
fn plan_for(ff: &Arc<Ffmpeg>, input: &Path, dir: &Path) -> Fixture {
    let manifest = probe(ff, input).expect("probe");
    let bus = EventBus::new();
    let reporter = Reporter::new(bus);
    let cancel = AtomicBool::new(false);

    let temporal = classify(ff, &manifest, &reporter, &cancel, &ClassifyOptions::default())
        .expect("classify");
    let cadence = cadence_for(&manifest, &temporal).expect("cadence");
    let scenes = detect_scenes(
        ff,
        &manifest,
        &reporter,
        &cancel,
        &SceneOptions::default(),
        SceneDecode {
            pre_chain: cadence.chain_str(),
            fps: cadence.fps,
            reason: "corpus",
        },
    )
    .expect("scene analysis");

    let mut profile = RestorationProfile::deterministic();
    profile.interpolation.method = InterpolationMethod::Off;
    profile.interpolation.multiplier = 1;
    profile.restoration.max_upscale = 1.0;
    profile.audio.enabled = false;
    profile.output.prefer_hardware = false;

    let request = PlanRequest {
        job_id: "corpus".into(),
        input: input.to_path_buf(),
        output: dir.join("out.mkv"),
        profile,
    };
    let engines = EngineRegistry::probe(Arc::clone(ff));
    let plan = build_plan(ff, &engines, &manifest, &temporal, &scenes, None, &request)
        .expect("build a plan");
    Fixture {
        plan,
        input: input.to_path_buf(),
    }
}

fn run_pipeline(ff: &Arc<Ffmpeg>, input: &Path, output: &Path, scratch: &Path) -> (bool, String) {
    let runner = PipelineRunner::new(
        Arc::clone(ff),
        Arc::new(EngineRegistry::probe(Arc::clone(ff))),
        EventBus::new(),
        Arc::new(Store::open(&scratch.join("jobs.sqlite3")).expect("state store")),
        Arc::new(AtomicBool::new(false)),
        scratch.to_path_buf(),
    );
    let mut profile = RestorationProfile::deterministic();
    profile.interpolation.method = InterpolationMethod::Off;
    profile.interpolation.multiplier = 1;
    profile.restoration.max_upscale = 1.0;
    profile.audio.enabled = false;
    profile.output.prefer_hardware = false;
    let outcome = runner
        .run(
            PlanRequest {
                job_id: "formats".into(),
                input: input.to_path_buf(),
                output: output.to_path_buf(),
                profile,
            },
            &RunnerOptions {
                resume: true,
                ..Default::default()
            },
        )
        .expect("the runner always reports an outcome");
    (outcome.ok, outcome.message)
}

#[test]
fn an_anamorphic_source_is_resolved_to_square_pixels() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_anamorphic(&ff, dir.path());
    let manifest = probe(&ff, &input).expect("probe");
    let video = manifest.primary_video().expect("video");
    assert_eq!(video.size(), Some((720, 480)));
    let (square_w, square_h) = video.square_pixel_size().expect("square size");
    eprintln!(
        "raw 720x480 sar {} -> square {square_w}x{square_h}",
        video.sar()
    );
    assert!(
        square_w > 800 && square_w < 900,
        "720x480 at 32:27 is about 854 wide in square pixels, got {square_w}"
    );
    assert_eq!(square_h, 480, "only the width changes");

    let fixture = plan_for(&ff, &input, dir.path());
    let plan = &fixture.plan;
    // The target comes from the *display* shape, so it must be wide, not 720.
    assert!(
        plan.video.target_width > 800,
        "the target raster must follow the display shape, got {}x{}",
        plan.video.target_width,
        plan.video.target_height
    );
    // A model must see square pixels, so the chain that feeds it has to correct
    // the raster first. That correction lives in `build_decode_args` — the
    // analysis chain in `plan.video.decode_chain` is a different thing, and
    // asserting on it was the first version of this test's mistake.
    let decode_args = sr_core::pipeline::native::build_decode_args(&input, &plan.video);
    let chain = decode_args
        .iter()
        .position(|arg| arg == "-vf")
        .and_then(|at| decode_args.get(at + 1))
        .cloned()
        .unwrap_or_default();
    eprintln!("model decode chain: `{chain}`");
    assert!(
        chain.contains(&format!("scale={square_w}:{square_h}")),
        "the model must be fed square pixels, got `{chain}`"
    );
    assert!(
        chain.contains("format=rgb24"),
        "the model takes rgb24, got `{chain}`"
    );

    // And the published file must be square-pixel with the same display shape.
    let output = dir.path().join("square.mkv");
    let (ok, message) = run_pipeline(&ff, &input, &output, &dir.path().join("scratch"));
    assert!(ok, "the job failed: {message}");
    let result = probe(&ff, &output).expect("probe the output");
    let out_video = result.primary_video().expect("video");
    let sar = out_video.sar();
    assert!(
        sar.to_f64() > 0.99 && sar.to_f64() < 1.01,
        "the output must have square pixels, got sar {sar}"
    );
    let dar = out_video
        .dar()
        .map(|ratio| ratio.to_f64())
        .unwrap_or_else(|| {
            let (width, height) = out_video.size().unwrap_or((0, 0));
            width as f64 / height as f64
        });
    assert!(
        (dar - 16.0 / 9.0).abs() < 0.05,
        "the output must keep the 16:9 display shape, got {dar:.3}"
    );
}

#[test]
fn true_interlacing_is_deinterlaced_and_never_decimated() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_interlaced(&ff, dir.path(), "60000/1001", "30000/1001", "interlaced.mkv");
    let manifest = probe(&ff, &input).expect("probe");
    let source_frames = frames(&ff, &input);
    eprintln!(
        "interlaced fixture: {source_frames} frames at {}",
        manifest
            .primary_video()
            .and_then(|v| v.fps())
            .map(|f| f.to_f64())
            .unwrap_or(0.0)
    );
    assert!(source_frames >= 29, "a second at 29.97 is about 30 frames");

    let bus = EventBus::new();
    let reporter = Reporter::new(bus);
    let cancel = AtomicBool::new(false);
    let temporal = classify(&ff, &manifest, &reporter, &cancel, &ClassifyOptions::default())
        .expect("classify");
    eprintln!("interlaced: {}", temporal.summary());
    assert_eq!(
        temporal.mode,
        TemporalMode::Interlaced,
        "real interlacing must be recognised as interlaced: {}",
        temporal.summary()
    );
    assert!(temporal.mode.needs_deinterlace());
    assert!(
        !temporal.mode.needs_ivtc(),
        "interlaced video has no pulldown to undo, and running `decimate` on it \
         would destroy half the temporal information"
    );

    let cadence = cadence_for(&manifest, &temporal).expect("cadence");
    let chain = cadence.chain_str().unwrap_or_default();
    assert!(chain.contains("bwdif"), "chain was `{chain}`");
    assert!(
        !chain.contains("decimate"),
        "deinterlacing must not drop frames: `{chain}`"
    );
    assert!(
        !cadence.renumbers_frames,
        "the frame count must survive deinterlacing"
    );

    // The frame count through the chain is the count the rest of the pipeline
    // reasons with, so losing frames here would silently shorten the film.
    let scenes = detect_scenes(
        &ff,
        &manifest,
        &reporter,
        &cancel,
        &SceneOptions::default(),
        SceneDecode {
            pre_chain: cadence.chain_str(),
            fps: cadence.fps,
            reason: "corpus",
        },
    )
    .expect("scene analysis");
    assert_eq!(
        scenes.frames_analyzed, source_frames,
        "deinterlacing must keep every frame"
    );

    let output = dir.path().join("deinterlaced.mkv");
    let (ok, message) = run_pipeline(&ff, &input, &output, &dir.path().join("scratch"));
    assert!(ok, "the job failed: {message}");
    assert_eq!(
        frames(&ff, &output),
        source_frames,
        "the output must have the same number of frames as the input"
    );
    let result = probe(&ff, &output).expect("probe the output");
    let out_fps = result
        .primary_video()
        .and_then(|v| v.fps())
        .map(|f| f.to_f64())
        .unwrap_or(0.0);
    assert!(
        (out_fps - 29.97).abs() < 0.05,
        "deinterlacing must not change the frame rate, got {out_fps:.3}"
    );
}

#[test]
fn pal_fifty_is_deinterlaced_and_not_speed_changed() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_interlaced(&ff, dir.path(), "50", "25", "pal-50i.mkv");
    let manifest = probe(&ff, &input).expect("probe");
    let source_frames = frames(&ff, &input);

    let bus = EventBus::new();
    let reporter = Reporter::new(bus);
    let cancel = AtomicBool::new(false);
    let temporal = classify(&ff, &manifest, &reporter, &cancel, &ClassifyOptions::default())
        .expect("classify");
    eprintln!("pal 50i: {}", temporal.summary());
    assert_eq!(
        temporal.mode,
        TemporalMode::Interlaced,
        "50 fields per second of motion is interlaced video: {}",
        temporal.summary()
    );

    let cadence = cadence_for(&manifest, &temporal).expect("cadence");
    assert_eq!(
        cadence.fps.to_f64(),
        25.0,
        "deinterlacing 50i gives 25p; it must not become 24"
    );
    assert!(
        !cadence.renumbers_frames,
        "nothing may be dropped: a PAL film played back at 24 fps is 4% short"
    );
    let chain = cadence.chain_str().unwrap_or_default();
    assert!(
        chain.contains("bwdif") && !chain.contains("decimate"),
        "chain was `{chain}`"
    );

    let output = dir.path().join("pal-out.mkv");
    let (ok, message) = run_pipeline(&ff, &input, &output, &dir.path().join("scratch"));
    assert!(ok, "the job failed: {message}");
    assert_eq!(
        frames(&ff, &output),
        source_frames,
        "a PAL source must come out with the frames it went in with"
    );
    let result = probe(&ff, &output).expect("probe the output");
    let out_fps = result
        .primary_video()
        .and_then(|v| v.fps())
        .map(|f| f.to_f64())
        .unwrap_or(0.0);
    assert!(
        (out_fps - 25.0).abs() < 0.05,
        "25 fps must stay 25 fps, got {out_fps:.3}"
    );
}
