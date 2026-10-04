//! The cadence regression corpus, starting with the chain that had never been run.
//!
//! `fieldmatch → decimate → 23.976 → 47.952` is the pipeline's most intricate
//! path: a cadence classifier decides from `idet` statistics that a source is
//! telecined, the decode chain removes the duplicated fields, the frame numbering
//! changes by 20% underneath every downstream stage, and interpolation then
//! doubles what is left. Each step was implemented and none of it had been
//! exercised end to end — `classify` was tested against synthetic `idet` counts,
//! `decimate` was never run, and the interpolation tests used progressive
//! fixtures where nothing is renumbered.
//!
//! The fixture is a real 3:2 pulldown: 48 progressive frames at 24 fps put
//! through FFmpeg's `telecine=pattern=23`, which gives 60 frames at 29.97 with a
//! duplicated field in every fifth frame. `idet` reads it as 40% repeated fields
//! and 97% single-frame TFF. The test asserts the classifier agrees, that inverse
//! telecine puts the frame count back to 48, and that the full chain reaches
//! 47.952 fps at the right frame count.
//!
//! Skips itself when FFmpeg is unavailable.

mod support;

use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::infer::EngineRegistry;
use sr_core::media::classify::{classify, ClassifyOptions, TemporalMode};
use sr_core::media::probe;
use sr_core::media::scene::{detect_scenes, SceneDecode, SceneOptions};
use sr_core::pipeline::plan::{cadence_for, PlanRequest};
use sr_core::pipeline::profile::{InterpolationMethod, RestorationProfile};
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::{EventBus, Reporter};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// 48 progressive frames at 24 fps.
const SOURCE_FRAMES: u64 = 48;

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
    [dir.join(name), dir.join("deps").join(name)]
        .into_iter()
        .find(|path| path.exists())
}

/// Which `TemporalMode` the classifier reports, and what it based that on.
struct Cadence {
    mode: TemporalMode,
    summary: String,
}

fn classify_fixture(ff: &Ffmpeg, manifest: &sr_core::media::manifest::MediaManifest) -> Cadence {
    let bus = EventBus::new();
    let reporter = Reporter::new(bus);
    let cancel = AtomicBool::new(false);
    let report = classify(ff, manifest, &reporter, &cancel, &ClassifyOptions::default())
        .expect("classify the fixture");
    Cadence {
        mode: report.mode,
        summary: report.summary(),
    }
}

/// A 24p source, and the same content as a 3:2 pulldown.
fn build_fixture(ff: &Ffmpeg, dir: &Path) -> (PathBuf, PathBuf) {
    let progressive = dir.join("progressive-24p.mkv");
    capture(
        &ff.ffmpeg,
        &args(&[
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=24:duration=2",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
        ])
        .into_iter()
        .chain(std::iter::once(progressive.display().to_string()))
        .collect::<Vec<_>>(),
    )
    .expect("build the 24p source");

    let telecined = dir.join("telecined.mkv");
    let mut argv = args(&["-y", "-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(progressive.display().to_string());
    argv.extend(args(&[
        "-vf",
        "telecine=pattern=23",
        "-r",
        "30000/1001",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "matroska",
    ]));
    argv.push(telecined.display().to_string());
    capture(&ff.ffmpeg, &argv).expect("apply 3:2 pulldown");
    (progressive, telecined)
}

/// Frames in a file, counted by decoding it.
fn count_frames(ff: &Ffmpeg, path: &Path) -> u64 {
    let mut argv = args(&["-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(path.display().to_string());
    argv.extend(args(&["-map", "0:v:0", "-f", "framemd5", "-"]));
    let text = capture(&ff.ffmpeg, &argv).expect("framemd5");
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .count() as u64
}

/// The fixture is what the test claims it is, before anything is asserted about
/// the engine.
#[test]
fn the_pulldown_fixture_is_a_real_three_two_cadence() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let (progressive, telecined) = build_fixture(&ff, dir.path());

    assert_eq!(count_frames(&ff, &progressive), SOURCE_FRAMES);
    // 3:2 pulldown adds one frame per four, so 48 becomes 60.
    assert_eq!(
        count_frames(&ff, &telecined),
        SOURCE_FRAMES * 5 / 4,
        "the pulldown must add a frame in every four"
    );
    let manifest = probe(&ff, &telecined).expect("probe");
    let fps = manifest.primary_video().and_then(|v| v.fps()).expect("fps");
    assert!(
        (fps.to_f64() - 29.97).abs() < 0.01,
        "the pulldown runs at 29.97, got {fps}"
    );
}

#[test]
fn a_three_two_pulldown_is_classified_as_telecine_not_interlaced() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let (_, telecined) = build_fixture(&ff, dir.path());
    let manifest = probe(&ff, &telecined).expect("probe");

    let cadence = classify_fixture(&ff, &manifest);
    eprintln!("telecined: {}", cadence.summary);
    assert_eq!(
        cadence.mode,
        TemporalMode::Telecine,
        "a 3:2 pulldown must be recognised as telecine, not as interlaced video \
         (deinterlacing it would throw away half the temporal information): {}",
        cadence.summary
    );
    assert!(cadence.mode.needs_ivtc());
    assert!(!cadence.mode.needs_deinterlace());

    // The progressive original must at least never be *telecined*: the telecine
    // rule triggers `fieldmatch,decimate`, and running it on a progressive source
    // would throw away one frame in five for nothing.
    //
    // Asserting `== Progressive` would be nicer and is not available: `idet`'s
    // multi-frame detector needs real motion to reach a progressive verdict, and
    // every synthetic source tried here — `testsrc2`, `gradients`, `mandelbrot`,
    // `smptebars`, with and without added temporal noise — comes back
    // "Undetermined", which the classifier reads as `Mixed`. That is a limitation
    // of a corpus built from synthetic video, and it is the reason the fixtures
    // for this project are built from samples rather than trusted.
    let progressive = probe(&ff, dir.path().join("progressive-24p.mkv").as_path());
    if let Ok(manifest) = progressive {
        let cadence = classify_fixture(&ff, &manifest);
        eprintln!("progressive: {}", cadence.summary);
        assert_ne!(
            cadence.mode,
            TemporalMode::Telecine,
            "a progressive source must never be treated as telecined: {}",
            cadence.summary
        );
        assert!(
            !cadence.mode.needs_ivtc(),
            "nothing may decimate a progressive source: {}",
            cadence.summary
        );
    }
}

#[test]
fn inverse_telecine_puts_the_frame_count_back() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let (_, telecined) = build_fixture(&ff, dir.path());
    let manifest = probe(&ff, &telecined).expect("probe");
    let bus = EventBus::new();
    let reporter = Reporter::new(bus);
    let cancel = AtomicBool::new(false);

    let temporal = classify(
        &ff,
        &manifest,
        &reporter,
        &cancel,
        &ClassifyOptions::default(),
    )
    .expect("classify");
    let cadence = cadence_for(&manifest, &temporal).expect("cadence");
    assert!(
        cadence.renumbers_frames,
        "telecine must be marked as renumbering the frames"
    );
    assert!(
        (cadence.fps.to_f64() - 23.976).abs() < 0.01,
        "inverse telecine lands on 23.976, got {}",
        cadence.fps
    );
    let chain = cadence.chain_str().unwrap_or_default();
    assert!(chain.contains("fieldmatch"), "chain was `{chain}`");
    assert!(chain.contains("decimate"), "chain was `{chain}`");

    // The shot analysis runs on the *decoded* stream, so its frame count is the
    // post-IVTC one: 60 frames in the file, 48 out of the chain.
    let scenes = detect_scenes(
        &ff,
        &manifest,
        &reporter,
        &cancel,
        &SceneOptions::default(),
        SceneDecode {
            pre_chain: cadence.chain_str(),
            fps: cadence.fps,
            reason: "inverse telecine",
        },
    )
    .expect("scene analysis through the cadence chain");
    eprintln!(
        "analysed {} frames at {:.3} fps from a {}-frame file",
        scenes.frames_analyzed,
        scenes.fps.to_f64(),
        SOURCE_FRAMES * 5 / 4
    );
    assert_eq!(
        scenes.frames_analyzed, SOURCE_FRAMES,
        "the cadence chain must give back the 48 frames the pulldown was made from"
    );
    assert!(
        (scenes.fps.to_f64() - 23.976).abs() < 0.01,
        "the shot list must be in post-IVTC time, got {:.3} fps",
        scenes.fps.to_f64()
    );
}

#[test]
fn the_whole_chain_runs_from_pulldown_to_double_rate() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let Some(plugin) = example_plugin() else {
        eprintln!("SKIPPED: the example inference plugin was not built");
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let (_, telecined) = build_fixture(&ff, dir.path());
    let output = dir.path().join("restored.mkv");
    std::env::set_var("SR_INFER_PLUGIN", &plugin);

    let store = Arc::new(Store::open(&dir.path().join("jobs.sqlite3")).expect("state store"));
    let bus = EventBus::new();
    let events = bus.subscribe();
    let runner = PipelineRunner::new(
        Arc::clone(&ff),
        Arc::new(EngineRegistry::probe(Arc::clone(&ff))),
        bus.clone(),
        Arc::clone(&store),
        Arc::new(AtomicBool::new(false)),
        dir.path().join("scratch"),
    );

    let mut profile = RestorationProfile::deterministic();
    // Interpolation from the model, everything else deterministic: the point of
    // this test is the cadence chain, not the backend.
    profile.interpolation.method = InterpolationMethod::Plugin;
    profile.interpolation.multiplier = 2;
    profile.restoration.enabled = false;
    // Keep the raster at the source size so the test measures cadence rather than
    // the cost of an upscale.
    profile.restoration.max_upscale = 1.0;
    profile.output.prefer_hardware = false;
    profile.audio.enabled = false;

    let outcome = runner
        .run(
            PlanRequest {
                job_id: "cadence".into(),
                input: telecined.clone(),
                output: output.clone(),
                profile,
            },
            &RunnerOptions {
                resume: true,
                keep_intermediates: false,
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
    assert!(
        outcome.ok,
        "the telecine chain failed: {}\n{}",
        outcome.message,
        logs.iter()
            .filter(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("cadence") || lower.contains("chunk") || lower.contains("fail")
            })
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );

    // The cadence decision must be visible in the log, not just in the plan.
    assert!(
        logs.iter().any(|line| line.contains("inverse telecine")),
        "the log must say the pulldown was removed:\n{}",
        logs.iter()
            .filter(|line| line.contains("cadence"))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );

    let result = probe(&ff, &output).expect("probe the output");
    let fps = result
        .primary_video()
        .and_then(|video| video.fps())
        .map(|f| f.to_f64())
        .unwrap_or(0.0);
    assert!(
        (fps - 47.952).abs() < 0.05,
        "the output must run at twice the post-IVTC rate, got {fps:.3} fps"
    );
    // 48 frames after inverse telecine, doubled: 2*(48-1)+1 = 95.
    let frames = count_frames(&ff, &output);
    assert_eq!(
        frames,
        2 * (SOURCE_FRAMES - 1) + 1,
        "the whole chain must emit 95 frames: 60 in the file, 48 after IVTC, 95 after doubling"
    );
}
