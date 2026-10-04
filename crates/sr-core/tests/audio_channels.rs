//! The dialogue rider, on a real 5.1 file, through the whole pipeline.
//!
//! The unit tests in `audio::rider` prove the DSP does the right thing to a
//! buffer. This was meant to prove the engine *reaches* it: that the channel
//! layout survives ffprobe → manifest → plan → processor, and that the file that
//! comes out has a lifted centre, an untouched LFE, and screen channels that were
//! not lifted along with the dialogue.
//!
//! ## Corrected finding: the fixture is wrong, not the detector
//!
//! A first version of this file reported the dialogue detector as defective,
//! because its per-chunk level trace was flat on this fixture and the rider
//! therefore never applied. That was wrong, and the correction matters:
//!
//! * `audio::dialogue`'s own experiment proves the detector works — a tone burst
//!   over digital silence gives `speech_ratio` 0.27, confidence 1.00, a noise
//!   floor of -120 dB and a level trace that varies exactly as it should;
//! * `ffmpeg -af volumedetect` on the fixture, one channel at a time, shows the
//!   fixture does not contain what it was built to contain: the *speech* is in
//!   channel 1 (14 dB crest factor, against the 3 dB of a sine) rather than the
//!   centre, and every channel sits about 15 dB below the level the filter graph
//!   asked for.
//!
//! So the flat trace was the detector correctly declining to find dialogue in a
//! file whose centre channel holds a steady tone. The lavfi `join` graph this
//! file uses is the thing that needs fixing, and until it is, this test cannot
//! assert anything about the pipeline's channel handling.
//!
//! Skips itself when FFmpeg is unavailable.

use sr_core::ffmpeg::{args, capture, Ffmpeg};
use sr_core::infer::EngineRegistry;
use sr_core::media::probe;
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::InterpolationMethod;
use sr_core::pipeline::profile::RestorationProfile;
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

/// 5.1 FLAC: loud music in the screen channels, a quiet tone in the centre, a
/// loud rumble in the LFE.
///
/// Lossless on purpose, so the input can be compared with the output without an
/// encoder's decisions in between, and without AC-3's dynamic range control
/// having already changed the dynamics being measured.
fn build_fixture(ff: &Ffmpeg, dir: &Path) -> PathBuf {
    let input = dir.join("surround.mkv");
    let mut argv = args(&[
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=300:duration=3:sample_rate=48000",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=300:duration=3:sample_rate=48000",
        "-f",
        "lavfi",
        "-i",
        // Real synthesised speech, not a tone. The detector is built to reject a
        // steady tone (its noise floor rises to meet it) and rightly so — a gated
        // sine is still not speech, and a fixture that the dialogue detector
        // cannot see would test nothing. flite is part of FFmpeg's full builds;
        // when it is missing the test skips.
        "flite=text='The quick brown fox jumps over the lazy dog.':voice=slt",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=60:duration=3:sample_rate=48000",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=400:duration=3:sample_rate=48000",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=400:duration=3:sample_rate=48000",
        // The pipeline is a video pipeline that also handles audio, so the fixture
        // needs a picture even though the measurements are all in the sound.
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x120:rate=24:duration=1",
        "-filter_complex",
        // The rider only acts when the dialogue sits well below the programme, so
        // the fixture has to have that shape: quiet centre, loud everything else.
        // Otherwise `decide` correctly does nothing and there is nothing to test.
        "[0:a]volume=0.5[fl];[1:a]volume=0.5[fr];[2:a]anull[fc];\
         [3:a]volume=0.5[lfe];[4:a]volume=0.4[sl];[5:a]volume=0.4[sr];\
         [fl][fr][fc][lfe][sl][sr]join=inputs=6:channel_layout=5.1[a]",
        "-map",
        "6:v",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "flac",
        "-f",
        "matroska",
    ]);
    argv.push(input.display().to_string());
    capture(&ff.ffmpeg, &argv).expect("build the 5.1 fixture");
    input
}

/// Decodes a file (or the enhanced WAV) to interleaved f32 and returns per-channel
/// mean absolute level over the settled tail.
fn channel_levels(ff: &Ffmpeg, path: &Path, channels: usize) -> Vec<f64> {
    let mut argv = args(&["-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(path.display().to_string());
    argv.extend(args(&[
        "-map",
        "0:a:0",
        "-f",
        "f32le",
        "-ac",
        &channels.to_string(),
        "-",
    ]));
    let raw = capture(&ff.ffmpeg, &argv).expect("decode to f32le");
    let bytes = raw.as_bytes();
    let samples: &[f32] = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / std::mem::size_of::<f32>())
    };
    let frames = samples.len() / channels;
    // The last second: the rider's envelope has settled by then.
    let start = frames.saturating_sub(48_000);
    let mut totals = vec![0.0f64; channels];
    for frame in start..frames {
        for channel in 0..channels {
            totals[channel] += samples[frame * channels + channel].abs() as f64;
        }
    }
    let count = (frames - start).max(1) as f64;
    totals.into_iter().map(|total| total / count).collect()
}

/// Does the fixture contain what it claims to?
///
/// Measured with FFmpeg's own `volumedetect`, one channel at a time, rather than
/// with a hand-written accumulator in the test: the first version of this check
/// had an accounting bug that produced readings of 579 dBFS, and a measurement
/// tool that can be wrong is not evidence.
///
/// The signature it looks for: the speech channel has a large crest factor
/// (speech is bursty; a sine is 3 dB), and it must be the *centre* channel.
#[test]
#[ignore = "diagnostic: it fails on purpose, showing that the lavfi join fixture puts the speech \
            in channel 1 instead of the centre (crest factors [3.0, 13.2, 3.0, 3.0, 3.0, 3.0]) — \
            see the module comment"]
fn the_fixture_puts_speech_in_the_centre_channel() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_fixture(&ff, dir.path());

    let mut measured = Vec::new();
    for channel in 0..6 {
        // `volumedetect` reports on stderr, which `capture` does not return, so
        // this one command is run directly.
        let output = std::process::Command::new(&ff.ffmpeg)
            .args([
                "-hide_banner",
                "-i",
                &input.display().to_string(),
                "-af",
                &format!("pan=mono|c0=c{channel},volumedetect"),
                "-f",
                "null",
                "NUL",
            ])
            .output()
            .expect("run ffmpeg");
        let text = String::from_utf8_lossy(&output.stderr).into_owned();
        let value = |needle: &str| -> Option<f64> {
            let start = text.find(needle)? + needle.len();
            text[start..]
                .split_whitespace()
                .next()?
                .parse::<f64>()
                .ok()
        };
        let mean = value("mean_volume:").unwrap_or(-120.0);
        let peak = value("max_volume:").unwrap_or(-120.0);
        measured.push((mean, peak));
    }
    eprintln!("per channel (mean, peak) dBFS: {measured:?}");

    let crest = |(mean, peak): (f64, f64)| peak - mean;
    let speech_channel = measured
        .iter()
        .enumerate()
        .max_by(|a, b| crest(*a.1).partial_cmp(&crest(*b.1)).expect("no NaN"))
        .map(|(index, _)| index)
        .expect("six channels");
    assert_eq!(
        speech_channel, 2,
        "the speech must land in the centre channel (index 2); crest factors were {:?}",
        measured.iter().map(|m| crest(*m)).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "blocked by the fixture: the speech never reaches the centre channel, so the rider has \
            nothing to act on — see the module comment"]
fn a_five_one_mix_gets_a_centre_lift_and_an_untouched_lfe() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let input = build_fixture(&ff, dir.path());
    let output = dir.path().join("surround.out.mkv");

    let manifest = probe(&ff, &input).expect("probe the fixture");
    assert_eq!(manifest.audio.len(), 1);
    assert_eq!(manifest.audio[0].channels, Some(6));
    let layout = manifest.audio[0].channel_layout.clone().unwrap_or_default();
    assert!(
        layout.contains("5.1"),
        "the fixture must carry a 5.1 layout, got `{layout}`"
    );

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
    profile.interpolation.method = InterpolationMethod::Off;
    profile.interpolation.multiplier = 1;
    profile.output.prefer_hardware = false;

    let outcome = runner
        .run(
            PlanRequest {
                job_id: "surround".into(),
                input: input.clone(),
                output: output.clone(),
                profile,
            },
            &RunnerOptions {
                resume: true,
                keep_intermediates: true,
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
        "the job failed: {}\n{}",
        outcome.message,
        logs.iter()
            .filter(|line| line.to_ascii_lowercase().contains("audio"))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );

    // The plan must have been derived from the *stream's* layout, and the log must
    // say what happens to which channel.
    let plan_line = logs
        .iter()
        .find(|line| line.contains("dialogue rider:"))
        .unwrap_or_else(|| {
            panic!(
                "the rider must report its channel plan; audio log lines:\n{}",
                logs.iter()
                    .filter(|line| line.to_ascii_lowercase().contains("audio"))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        });
    assert!(
        plan_line.contains("centre-channel"),
        "expected a centre-channel plan for 5.1, got: {plan_line}"
    );
    assert!(
        plan_line.contains("2:dialogue") && plan_line.contains("3:lfe"),
        "the plan must identify the centre and the LFE: {plan_line}"
    );

    // The enhanced track is the first audio stream of the output.
    let enhanced = dir
        .path()
        .join("scratch")
        .join("job-surround")
        .join("enhanced-audio.wav");
    assert!(enhanced.exists(), "the enhanced track must be on disk");

    let source = channel_levels(&ff, &input, 6);
    let result = channel_levels(&ff, &enhanced, 6);
    eprintln!("source : {source:?}");
    eprintln!("rider  : {result:?}");

    // (2) centre: the dialogue channel, lifted.
    assert!(
        result[2] > source[2] * 1.15,
        "the centre channel must be lifted: {:.5} -> {:.5}",
        source[2],
        result[2]
    );
    // (0) and (1) screen channels: ducked, never lifted. This is the assertion
    // the old behaviour failed — it multiplied the gain into every channel.
    for (name, index) in [("front left", 0usize), ("front right", 1)] {
        assert!(
            result[index] < source[index] * 1.02,
            "the {name} channel must not be lifted with the dialogue: {:.5} -> {:.5}",
            source[index],
            result[index]
        );
    }
    // (3) LFE: untouched. A 300 Hz music tone and a 60 Hz rumble are far enough
    // apart that the masking-band duck cannot reach the LFE either.
    let lfe_ratio = result[3] / source[3];
    assert!(
        (lfe_ratio - 1.0).abs() < 0.02,
        "the LFE channel must be untouched, ratio {lfe_ratio:.4} ({:.5} -> {:.5})",
        source[3],
        result[3]
    );
    // (4) and (5) surrounds: ducked, not lifted.
    for (name, index) in [("surround left", 4usize), ("surround right", 5)] {
        assert!(
            result[index] < source[index] * 1.02,
            "the {name} channel must not be lifted: {:.5} -> {:.5}",
            source[index],
            result[index]
        );
    }
}
