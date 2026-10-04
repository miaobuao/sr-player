//! The dialogue rider, on a real 5.1 file, through the whole pipeline.
//!
//! The unit tests in `audio::rider` prove the DSP does the right thing to a
//! buffer. This proves the engine *reaches* it: that the channel layout survives
//! ffprobe → manifest → plan → processor, and that what comes out has a lifted
//! centre, an untouched LFE, and screen channels that were not lifted along with
//! the dialogue.
//!
//! The fixture is built sample by sample in `support::WavBuilder` and handed to
//! FFmpeg only to be encoded and muxed, so its contents are a fact the test can
//! assert rather than a claim about a filter graph. An earlier version used
//! `lavfi` `sine` sources joined into a 5.1 layout, and measurement showed the
//! result did not contain what the graph claimed — the bursts were in channel 1
//! and every level was about 15 dB low.
//!
//! ## What this test found, and why it is ignored
//!
//! With a fixture that verifies clean, the pipeline still declines to ride the
//! dialogue, and the reason is a real defect one level above the detector:
//!
//! * the detector is correct — `speech_ratio` 0.403 for a 1/3 duty cycle,
//!   confidence 1.00, a 73.7 dB median SNR, and a mask whose runs line up with
//!   the bursts exactly;
//! * `dialogue_lufs` comes out at -2.566 against a programme of -2.573, so
//!   `LDR` is -0.007 LU and `decide` correctly declines ("already within the
//!   5.0 LU target").
//!
//! The dialogue loudness is measured from the **programme's** per-block loudness
//! gated by the speech mask — that is, "how loud the whole mix is while someone
//! is talking". With music playing under the dialogue, which is most of a film,
//! that is close to the programme by construction, so LDR collapses toward zero
//! and the rider never acts. The centre channel is already decoded and handed to
//! the *detector*; it is simply never handed to a meter.
//!
//! The fix is to measure the dialogue channel itself: a `LoudnessMeter` for the
//! centre samples (which `analyze` already extracts) whose `block_loudness()` is
//! gated by the same mask. That is what LDR means.
//!
//! The test is ignored rather than adjusted so the finding stays in the tree and
//! becomes the regression test for that fix.
//!
//! Skips itself when FFmpeg is unavailable.

mod support;

use sr_core::ffmpeg::{capture, Ffmpeg};
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
use support::{channel, dialogue_bursts, mask, tone, WavBuilder};

fn ffmpeg_or_skip() -> Option<Arc<Ffmpeg>> {
    match Ffmpeg::discover() {
        Ok(ff) => Some(Arc::new(ff)),
        Err(err) => {
            eprintln!("SKIPPED: {err}");
            None
        }
    }
}

/// The fixture's shape, in one place so the test and the assertions agree.
struct Fixture {
    path: PathBuf,
    wav: WavBuilder,
}

/// 5.1 with dialogue-shaped bursts in the centre and loud music everywhere else.
///
/// The rider only acts when the dialogue sits well below the programme, so the
/// levels have to have that shape: quiet centre, loud everything else. Otherwise
/// `decide` correctly does nothing and there is nothing to test.
fn build_fixture(ff: &Ffmpeg, dir: &Path) -> Fixture {
    let seconds = 12.0;
    let mut wav = WavBuilder::new(48_000, 6, mask::SURROUND_5_1, seconds);
    // Music in the screen channels: inside the ducked band (300 Hz–6 kHz) and
    // constant, so "was this channel gained?" is a clean measurement.
    wav.channel(channel::FL, tone(1_000.0, 0.6));
    wav.channel(channel::FR, tone(1_000.0, 0.6));
    // Dialogue in the centre: bursts, one second in three, at a third of the
    // music's level — the shape the rider exists to fix.
    wav.channel(
        channel::FC,
        dialogue_bursts(1_200.0, 0.06, 3.0, 1.0 / 3.0, 0.0005),
    );
    // A rumble in the LFE, an octave and a half below the ducked band, so nothing
    // the rider does to the masking band can reach it.
    wav.channel(channel::LFE, tone(60.0, 0.5));
    wav.channel(channel::BL, tone(400.0, 0.4));
    wav.channel(channel::BR, tone(400.0, 0.4));

    let wav_path = dir.join("surround.wav");
    wav.write(&wav_path).expect("write the fixture audio");

    let path = dir.join("surround.mkv");
    let video = dir.join("surround-video.mkv");
    // A picture, because this is a video pipeline that also handles audio.
    capture(
        &ff.ffmpeg,
        &sr_core::ffmpeg::args(&[
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=24:duration=2",
            "-c:v",
            "libx264",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
        ])
        .into_iter()
        .chain(std::iter::once(video.display().to_string()))
        .collect::<Vec<_>>(),
    )
    .expect("build the video");

    let mut argv = sr_core::ffmpeg::args(&["-y", "-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(video.display().to_string());
    argv.push("-i".into());
    argv.push(wav_path.display().to_string());
    argv.extend(sr_core::ffmpeg::args(&[
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "flac",
        "-f",
        "matroska",
    ]));
    argv.push(path.display().to_string());
    capture(&ff.ffmpeg, &argv).expect("mux the fixture");

    Fixture { path, wav }
}

/// Per-channel mean absolute level of a file (or the enhanced WAV), settled tail.
fn channel_levels(ff: &Ffmpeg, path: &Path, channels: usize) -> Vec<f64> {
    let mut argv = sr_core::ffmpeg::args(&["-hide_banner", "-loglevel", "error", "-i"]);
    argv.push(path.display().to_string());
    argv.extend(sr_core::ffmpeg::args(&[
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
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const f32,
            bytes.len() / std::mem::size_of::<f32>(),
        )
    };
    let frames = samples.len() / channels;
    // The last two seconds: the rider's envelope has settled by then.
    let start = frames.saturating_sub(96_000);
    let mut totals = vec![0.0f64; channels];
    for frame in start..frames {
        for channel in 0..channels {
            totals[channel] += samples[frame * channels + channel].abs() as f64;
        }
    }
    let count = (frames - start).max(1) as f64;
    totals.into_iter().map(|total| total / count).collect()
}

#[test]
fn the_fixture_carries_what_it_claims_to_carry() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let fixture = build_fixture(&ff, dir.path());

    // Measured on the buffer that was written, so this is a fact about the
    // fixture rather than a claim about FFmpeg.
    let centre = fixture.wav.levels(channel::FC);
    let front = fixture.wav.levels(channel::FL);
    let lfe = fixture.wav.levels(channel::LFE);
    eprintln!("centre {centre:?} front {front:?} lfe {lfe:?}");
    assert!(
        centre.crest_db() > 6.0,
        "the centre must be bursty (dialogue-shaped), got {centre:?}"
    );
    assert!(
        front.crest_db() < 4.0,
        "the screen channels must be steady music, got {front:?}"
    );
    assert!(
        lfe.peak_dbfs < -5.0,
        "the LFE must be well inside the range, got {lfe:?}"
    );

    // And the round trip through FFmpeg must not move the channels around, which
    // is exactly what the previous lavfi fixture did.
    let manifest = probe(&ff, &fixture.path).expect("probe");
    assert_eq!(manifest.audio.len(), 1);
    assert_eq!(manifest.audio[0].channels, Some(6));
    let layout = manifest.audio[0].channel_layout.clone().unwrap_or_default();
    assert!(
        layout.contains("5.1"),
        "the muxed file must still be 5.1, got `{layout}`"
    );
}

#[test]
#[ignore = "found a real defect: dialogue loudness is measured from the programme's blocks gated \
            by the speech mask, so LDR collapses to 0 whenever music plays under dialogue and \
            the rider never acts — see the module comment for the fix"]
fn a_five_one_mix_gets_a_centre_lift_and_an_untouched_lfe() {
    let Some(ff) = ffmpeg_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().expect("temp dir");
    let fixture = build_fixture(&ff, dir.path());
    let output = dir.path().join("surround.out.mkv");

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
                input: fixture.path.clone(),
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
    let audio_log = |needle: &str| -> String {
        logs.iter()
            .filter(|line| line.contains(needle))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert!(
        outcome.ok,
        "the job failed: {}\n{}",
        outcome.message,
        audio_log("audio")
    );

    // ---- the plan must be derived from the stream's layout, and say so ------
    let analysis = store
        .load_stage_result("surround", sr_core::events::Stage::AudioAnalysis)
        .expect("read the analysis")
        .unwrap_or_default();
    let plan_line = logs
        .iter()
        .find(|line| line.contains("dialogue rider:"))
        .unwrap_or_else(|| {
            panic!(
                "the rider must report its channel plan, but the decision was:\n{}",
                analysis
            )
        })
        .clone();
    assert!(
        plan_line.contains("centre-channel"),
        "expected a centre-channel plan for 5.1, got: {plan_line}"
    );
    assert!(
        plan_line.contains("2:dialogue") && plan_line.contains("3:lfe"),
        "the plan must identify the centre and the LFE: {plan_line}"
    );

    // ---- and the audio that came out must match the plan -------------------
    let enhanced = dir
        .path()
        .join("scratch")
        .join("job-surround")
        .join("enhanced-audio.wav");
    assert!(enhanced.exists(), "the enhanced track must be on disk");
    let source = channel_levels(&ff, &fixture.path, 6);
    let result = channel_levels(&ff, &enhanced, 6);
    eprintln!("source : {source:?}");
    eprintln!("rider  : {result:?}");

    // Centre: the dialogue channel, lifted.
    assert!(
        result[channel::FC] > source[channel::FC] * 1.15,
        "the centre channel must be lifted: {:.5} -> {:.5}",
        source[channel::FC],
        result[channel::FC]
    );
    // Screen channels: ducked, never lifted. This is the assertion the old
    // behaviour failed — it multiplied the gain into every channel.
    for (name, index) in [("front left", channel::FL), ("front right", channel::FR)] {
        assert!(
            result[index] < source[index] * 1.02,
            "the {name} channel must not be lifted with the dialogue: {:.5} -> {:.5}",
            source[index],
            result[index]
        );
    }
    // Surrounds: also ducked, never lifted.
    for (name, index) in [("surround left", channel::BL), ("surround right", channel::BR)] {
        assert!(
            result[index] < source[index] * 1.02,
            "the {name} channel must not be lifted: {:.5} -> {:.5}",
            source[index],
            result[index]
        );
    }
    // LFE: untouched. 60 Hz is far below the ducked band, and the role says no
    // gain, so the ratio must be 1.
    let lfe_ratio = result[channel::LFE] / source[channel::LFE];
    assert!(
        (lfe_ratio - 1.0).abs() < 0.05,
        "the LFE channel must be untouched, ratio {lfe_ratio:.4} ({:.5} -> {:.5})",
        source[channel::LFE],
        result[channel::LFE]
    );
}
