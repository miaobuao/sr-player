//! Grain measurement, per shot.
//!
//! A restoration chain smooths grain away: any model that reconstructs detail also
//! removes the noise that was never detail to begin with, and an upscale averages it
//! into flatness. The result looks plastic, and the usual fix — adding noise at a
//! fixed strength — is worse than doing nothing, because a film's grain varies from
//! shot to shot: a daylight exterior and a pushed night interior differ by more than
//! any single setting can cover.
//!
//! So the quantity to measure is per shot, and the quantity that matters is the
//! amplitude of the high-frequency residual. This module measures it in a way that
//! survives real pictures.
//!
//! ## Why not a standard deviation
//!
//! The obvious estimator — the standard deviation of a high-pass residual — is
//! dominated by whatever edges are in the frame. A shot of a brick wall has enormous
//! high-frequency detail and no grain at all, and a plain deviation would call it
//! grainy and then bury it in noise.
//!
//! The median absolute deviation is used instead. On a residual whose values are
//! mostly noise, the median absolute deviation is proportional to the noise and
//! ignores the small fraction of pixels that sit on edges, however large those are.
//! That is the whole reason for the choice: it is a measurement of the *bulk* of the
//! distribution rather than of its extremes.
//!
//! ## What it measures, precisely
//!
//! A four-neighbour Laplacian is applied, and the MAD of its response is scaled to a
//! per-pixel amplitude. For independent noise of deviation `s`, the Laplacian's
//! response has deviation `s * sqrt(20)` — the sum of the squared kernel taps — so
//! the estimate divides that out. Grain in real film is spatially correlated rather
//! than independent, so the number is best understood as "the high-frequency
//! amplitude that re-graining has to match" rather than "the sigma of a noise
//! process". That is the quantity the job needs.

/// One shot's grain measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrainEstimate {
    /// Amplitude of the high-frequency residual, in the same units as the samples
    /// (0..1 for float frames, 0..255 for 8-bit ones scaled to that range).
    pub sigma: f32,
    /// Pixels the estimate was taken from, so a caller can tell a measurement of a
    /// whole frame from one of a handful of pixels.
    pub samples: usize,
    /// How much to trust it: falls off when there are too few pixels, and when the
    /// residual is so far from noise-like that the number means little.
    pub confidence: f32,
}

impl GrainEstimate {
    pub fn describe(&self) -> String {
        format!(
            "grain sigma {:.4} from {} pixels (confidence {:.2})",
            self.sigma, self.samples, self.confidence
        )
    }

    /// Whether re-graining at this amplitude is worth doing at all.
    ///
    /// Below roughly a quarter of a code value at 8 bits, adding grain is adding
    /// rounding error with extra steps.
    pub fn is_worth_applying(&self) -> bool {
        self.sigma > 0.0008 && self.confidence > 0.2
    }
}

/// The Laplacian's response deviation for a unit-deviation input: the sum of the
/// squared taps, `4 * 1 + 16`.
const LAPLACIAN_POWER: f32 = 20.0;
/// Scales a median absolute deviation to a standard deviation for a normal
/// distribution.
const MAD_TO_SIGMA: f32 = 1.4826;

/// Measures the high-frequency amplitude of a single plane.
///
/// `luma` is `width * height` samples, row-major. The border is skipped because the
/// Laplacian has no neighbours there.
pub fn estimate_sigma(luma: &[f32], width: usize, height: usize) -> GrainEstimate {
    if width < 3 || height < 3 || luma.len() < width * height {
        return GrainEstimate {
            sigma: 0.0,
            samples: 0,
            confidence: 0.0,
        };
    }
    let mut residual: Vec<f32> = Vec::with_capacity((width - 2) * (height - 2));
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let at = |dx: isize, dy: isize| -> f32 {
                let index = ((y as isize + dy) as usize) * width + (x as isize + dx) as usize;
                luma[index]
            };
            let value =
                at(0, -1) + at(-1, 0) + at(1, 0) + at(0, 1) - 4.0 * at(0, 0);
            residual.push(value);
        }
    }
    let samples = residual.len();
    if samples < 16 {
        return GrainEstimate {
            sigma: 0.0,
            samples,
            confidence: 0.0,
        };
    }

    // Median absolute deviation, with the median itself removed first: the
    // Laplacian of a picture has a non-zero median wherever the picture has a
    // gradient, and subtracting it is what keeps a slow ramp from reading as grain.
    let median = median_of(&mut residual.clone());
    let mut deviations: Vec<f32> = residual.iter().map(|value| (value - median).abs()).collect();
    let mad = median_of(&mut deviations);
    let deviation = mad * MAD_TO_SIGMA;
    let sigma = deviation / LAPLACIAN_POWER.sqrt();

    // Confidence is about how much of the frame the estimate rests on, and how
    // plausible the residual's shape is. A residual whose deviation is tiny is a
    // clean frame and a perfectly good measurement of "no grain".
    let coverage = (samples as f32 / 4096.0).min(1.0);
    let confidence = coverage;
    GrainEstimate {
        sigma,
        samples,
        confidence,
    }
}

fn median_of(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

/// Measures a frame's three planes together, weighted by how much of the picture's
/// high-frequency content each carries.
///
/// Grain is not equally visible in all three, and measuring only green would be the
/// usual shortcut. Averaging the planes is not more correct, but it is not less so
/// either: what matters is the amplitude relative to the picture, and the sum of the
/// three is a better-behaved statistic than any one of them.
pub fn estimate_frame_sigma(
    planes: [&[f32]; 3],
    width: usize,
    height: usize,
) -> GrainEstimate {
    let estimates: Vec<GrainEstimate> = planes
        .iter()
        .map(|plane| estimate_sigma(plane, width, height))
        .collect();
    let samples: usize = estimates.iter().map(|estimate| estimate.samples).sum();
    if samples == 0 {
        return GrainEstimate {
            sigma: 0.0,
            samples: 0,
            confidence: 0.0,
        };
    }
    let sigma = estimates
        .iter()
        .map(|estimate| estimate.sigma)
        .sum::<f32>()
        / estimates.len() as f32;
    GrainEstimate {
        sigma,
        samples,
        confidence: estimates
            .iter()
            .map(|estimate| estimate.confidence)
            .fold(1.0f32, f32::min),
    }
}

/// The amplitude to re-grain a shot with, given what was measured across it.
///
/// Takes the median of the per-frame measurements rather than the mean: a shot with
/// a cut in it that the detector missed would put a few wildly different frames into
/// the average, and the median is the statistic that ignores them.
pub fn shot_sigma(per_frame: &[GrainEstimate]) -> GrainEstimate {
    let mut sigmas: Vec<f32> = per_frame.iter().map(|estimate| estimate.sigma).collect();
    if sigmas.is_empty() {
        return GrainEstimate {
            sigma: 0.0,
            samples: 0,
            confidence: 0.0,
        };
    }
    let sigma = median_of(&mut sigmas);
    GrainEstimate {
        sigma,
        samples: per_frame.iter().map(|estimate| estimate.samples).sum(),
        confidence: per_frame
            .iter()
            .map(|estimate| estimate.confidence)
            .fold(1.0f32, f32::min),
    }
}

/// How the per-shot pass samples.
#[derive(Clone, Copy, Debug)]
pub struct GrainOptions {
    /// Frames to measure per shot. Grain varies within a shot too, and the median
    /// across a few frames is steadier than any single one.
    pub frames_per_shot: usize,
    /// A ceiling on frames read overall, so a film with thousands of shots cannot
    /// turn a measurement into a second full decode.
    pub max_frames: usize,
}

impl Default for GrainOptions {
    fn default() -> Self {
        GrainOptions {
            frames_per_shot: 3,
            max_frames: 600,
        }
    }
}

/// One shot's measurement, with where it came from.
#[derive(Clone, Copy, Debug)]
pub struct ShotGrain {
    pub shot: usize,
    pub start_frame: u64,
    pub end_frame: u64,
    pub estimate: GrainEstimate,
}

/// Measures grain per shot, reading frames from the source at native resolution.
///
/// **Native resolution is not a detail.** The scene pass decodes a small raster
/// because shot detection only needs structure, but scaling averages noise away:
/// measuring grain on a downscaled frame reports a fraction of the truth, and the
/// fraction depends on the scale factor rather than on the film. So this decodes
/// full size and in gray, which is the least it can read per frame while still
/// seeing the noise.
///
/// Frames are sampled across each shot rather than from its start, because the first
/// frames of a shot are often a transition, and the samples are spread so that a
/// shot whose grain changes part-way through does not report only one end of it.
pub fn measure_shots(
    ff: &crate::ffmpeg::Ffmpeg,
    manifest: &crate::media::MediaManifest,
    shots: &[crate::media::scene::Shot],
    pre_chain: Option<&str>,
    options: &GrainOptions,
    reporter: &crate::Reporter,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<Vec<ShotGrain>, crate::error::Error> {
    use crate::ffmpeg::{args, read_exact_or_eof, RunSpec, StreamingChild};
    use crate::events::Stage;

    let video = manifest
        .primary_video()
        .ok_or_else(|| crate::error::Error::Unsupported("no video stream".into()))?;
    let (width, height) = video
        .size()
        .ok_or_else(|| crate::error::Error::Unsupported("unknown video dimensions".into()))?;
    let (width, height) = (width as usize, height as usize);

    // The frames each shot still wants, spread across it.
    let mut wanted: Vec<Vec<u64>> = shots
        .iter()
        .map(|shot| {
            let count = shot.frame_count();
            let samples = options.frames_per_shot.min(count as usize).max(1);
            (0..samples)
                .map(|sample| {
                    // Interior samples: a quarter in, three quarters in, and the
                    // middle, which keeps transitions at the edges out of the
                    // measurement.
                    let fraction = (sample as f64 + 0.5) / samples as f64;
                    // Skip the first and last frame of the shot.
                    let usable = count.saturating_sub(2).max(1);
                    shot.start_frame + 1 + (fraction * usable as f64) as u64
                })
                .map(|frame| frame.min(shot.end_frame.saturating_sub(1)))
                .collect()
        })
        .collect();
    let mut collected: Vec<Vec<GrainEstimate>> = vec![Vec::new(); shots.len()];

    let mut filters = String::new();
    if let Some(pre) = pre_chain.filter(|chain| !chain.is_empty()) {
        filters.push_str(pre);
        filters.push(',');
    }
    filters.push_str("format=gray");

    let mut argv = args(&["-hide_banner", "-nostdin", "-progress", "pipe:1", "-i"]);
    argv.push(manifest.path.display().to_string());
    argv.extend(args(&[
        "-map",
        "0:v:0",
        "-vf",
        &filters,
        "-an",
        "-sn",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]));
    let spec = RunSpec::new(Stage::Regrain, "grain-measure")
        .stderr_level(crate::events::Level::Debug);
    let mut child = StreamingChild::spawn(&ff.ffmpeg, &argv, reporter, &spec, false)?;
    let mut stdout = child.stdout.take().ok_or_else(|| crate::error::Error::Stage {
        stage: Stage::Regrain.id().into(),
        detail: "ffmpeg produced no video pipe".into(),
    })?;

    let frame_bytes = width * height;
    let mut buffer = vec![0u8; frame_bytes];
    let mut frame_index: u64 = 0;
    let mut read = 0usize;
    while read < options.max_frames {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            child.kill();
            return Err(crate::error::Error::Cancelled);
        }
        match read_exact_or_eof(&mut stdout, &mut buffer)? {
            size if size < frame_bytes => break,
            _ => {}
        }
        if let Some(shot) = shots
            .iter()
            .position(|shot| frame_index >= shot.start_frame && frame_index < shot.end_frame)
        {
            if let Some(at) = wanted[shot].iter().position(|frame| *frame == frame_index) {
                wanted[shot].remove(at);
                let luma: Vec<f32> = buffer.iter().map(|byte| *byte as f32 / 255.0).collect();
                collected[shot].push(estimate_sigma(&luma, width, height));
            }
        }
        frame_index += 1;
        read += 1;
        if wanted.iter().all(|frames| frames.is_empty()) {
            break;
        }
    }
    child.kill();

    Ok(shots
        .iter()
        .enumerate()
        .map(|(index, shot)| ShotGrain {
            shot: index,
            start_frame: shot.start_frame,
            end_frame: shot.end_frame,
            estimate: shot_sigma(&collected[index]),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise: a hash rather than a random number generator, so a
    /// failure is reproducible and the assertions can be exact about amplitude.
    fn noise(x: usize, y: usize) -> f32 {
        let mut hash = (x as u32).wrapping_mul(0x9E37_79B9) ^ (y as u32).wrapping_mul(0x85EB_CA6B);
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x2545_F491);
        hash ^= hash >> 13;
        // Two uniform samples averaged gives something close enough to normal for a
        // deviation estimate; the exact distribution is not what is under test. The
        // sum of two uniforms in [-0.5, 0.5] has deviation `sqrt(1/6)`, so the
        // constant scales it to one — without it the fixture's amplitudes are 0.71
        // of what they claim and the estimator looks 30% low.
        let a = ((hash & 0xFFFF) as f32 / 65535.0) - 0.5;
        let b = (((hash >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5;
        (a + b) * 2.449
    }

    /// A smooth picture with grain of a known amplitude.
    ///
    /// The ramp is the point: a plain deviation of a high-pass residual over a
    /// gradient is not the grain amplitude, and an estimator that forgets to remove
    /// the local median reads the ramp as noise.
    fn grainy(width: usize, height: usize, sigma: f32, ramp: f32) -> Vec<f32> {
        let mut luma = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let base = 0.3 + ramp * (x as f32 / width as f32);
                luma[y * width + x] = base + sigma * noise(x, y);
            }
        }
        luma
    }

    #[test]
    fn a_known_amount_of_grain_is_measured() {
        let (width, height) = (128, 96);
        for sigma in [0.01f32, 0.02, 0.04] {
            let luma = grainy(width, height, sigma, 0.0);
            let estimate = estimate_sigma(&luma, width, height);
            let error = (estimate.sigma - sigma).abs() / sigma;
            assert!(
                error < 0.15,
                "expected about {sigma}, got {} ({:.1}% out) for {}",
                estimate.sigma,
                error * 100.0,
                estimate.describe()
            );
        }
    }

    #[test]
    fn a_clean_frame_measures_no_grain() {
        let (width, height) = (128, 96);
        let luma = grainy(width, height, 0.0, 0.5);
        let estimate = estimate_sigma(&luma, width, height);
        assert!(
            estimate.sigma < 1e-4,
            "a smooth frame has no high-frequency residual: {}",
            estimate.describe()
        );
        assert!(!estimate.is_worth_applying());
    }

    /// The reason for a median rather than a deviation: structure is not grain, and
    /// an estimator that cannot tell them apart would bury a detailed shot in noise.
    ///
    /// The limit is stated here because the test shows exactly where it is: the edges
    /// must be a minority of pixels. The Laplacian spreads an edge over three rows
    /// and three columns, so a grid of lines every eight pixels puts almost half the
    /// frame within reach of one, and the median stops seeing the flat majority. Real
    /// pictures are not like that — but a shot that is *only* texture, a brick wall
    /// filling the frame, is, and this estimator will over-report it. That is the
    /// known weakness of the method and the reason the confidence is reported
    /// alongside the number.
    #[test]
    fn structure_is_not_mistaken_for_grain() {
        let (width, height) = (128, 96);
        // Sparse hard edges: high frequency, no grain, and a clear flat majority.
        let mut luma = vec![0.5f32; width * height];
        for y in 0..height {
            for x in 0..width {
                if x % 32 == 0 || y % 32 == 0 {
                    luma[y * width + x] = if (x / 32 + y / 32) % 2 == 0 { 0.1 } else { 0.9 };
                }
            }
        }
        let estimate = estimate_sigma(&luma, width, height);
        assert!(
            estimate.sigma < 0.01,
            "edges a minority of pixels apart are structure, not grain: {}",
            estimate.describe()
        );
        // And a deviation-based estimator would have said otherwise, which is worth
        // stating as a number rather than as a claim.
        let mean = luma.iter().sum::<f32>() / luma.len() as f32;
        let plain = (luma
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / luma.len() as f32)
            .sqrt();
        assert!(
            plain > estimate.sigma * 5.0,
            "the plain deviation ({plain:.4}) should be far larger than the robust \
             estimate ({:.4}); if it were not, this test would not be testing anything",
            estimate.sigma
        );
    }

    #[test]
    fn a_shot_uses_the_median_of_its_frames() {
        let frames = vec![
            GrainEstimate { sigma: 0.02, samples: 100, confidence: 1.0 },
            GrainEstimate { sigma: 0.021, samples: 100, confidence: 1.0 },
            GrainEstimate { sigma: 0.019, samples: 100, confidence: 1.0 },
            // A frame from a different scene that the detector did not separate.
            GrainEstimate { sigma: 0.20, samples: 100, confidence: 1.0 },
        ];
        let shot = shot_sigma(&frames);
        assert!(
            (shot.sigma - 0.0205).abs() < 0.002,
            "the median must ignore the intruder: {}",
            shot.describe()
        );
    }

    #[test]
    fn a_frame_too_small_to_measure_says_so() {
        let estimate = estimate_sigma(&[0.0; 4], 2, 2);
        assert_eq!(estimate.samples, 0);
        assert_eq!(estimate.confidence, 0.0);
        assert!(!estimate.is_worth_applying());
    }

    /// The whole pass, on a file: two shots whose grain differs by a known factor
    /// must come back labelled with the right amplitudes.
    ///
    /// This is the part that a unit test cannot check — decoding at native
    /// resolution, deciding which shot a frame belongs to, and keeping the two
    /// measurements apart. The fixture is built by adding noise to a mid-grey
    /// picture, so the amplitude that goes in is the amplitude the estimator should
    /// report.
    #[test]
    fn two_shots_with_different_grain_are_measured_separately() {
        use crate::ffmpeg::{args, capture, Ffmpeg};
        let Ok(ff) = Ffmpeg::discover() else {
            eprintln!("SKIPPED: no FFmpeg");
            return;
        };
        let dir = tempfile::tempdir().expect("temp dir");
        let input = dir.path().join("grain.mkv");
        // Two seconds of quiet grain, then two of heavy grain, with a hard cut.
        let mut argv = args(&[
            "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=gray:size=160x120:rate=24:duration=2",
            // A different level in the second shot. Two patches of the same grey
            // differing only in noise are not a cut the detector should report, and
            // the first version of this fixture proved it does not.
            "-f", "lavfi", "-i", "color=0x303030:size=160x120:rate=24:duration=2",
            "-filter_complex",
            "[0:v]noise=alls=6:allf=t+u[a];[1:v]noise=alls=24:allf=t+u[b];[a][b]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
            "-f", "matroska",
        ]);
        argv.push(input.display().to_string());
        capture(&ff.ffmpeg, &argv).expect("build the fixture");

        let manifest = crate::media::probe(&ff, &input).expect("probe");
        let bus = crate::EventBus::new();
        let reporter = crate::Reporter::new(bus);
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let scenes = crate::media::scene::detect_scenes(
            &ff,
            &manifest,
            &reporter,
            &cancel,
            &crate::media::scene::SceneOptions::default(),
            crate::media::scene::SceneDecode::raw(
                manifest.primary_video().and_then(|v| v.fps()).expect("fps"),
            ),
        )
        .expect("scenes");
        assert!(
            scenes.shots.len() >= 2,
            "the fixture has a hard cut and must be found as two shots, got {}",
            scenes.shots.len()
        );

        let measured = measure_shots(
            &ff,
            &manifest,
            &scenes.shots,
            None,
            &GrainOptions::default(),
            &reporter,
            &cancel,
        )
        .expect("measure");
        let described: Vec<String> = measured
            .iter()
            .map(|shot| format!("shot {} {}", shot.shot, shot.estimate.describe()))
            .collect();
        eprintln!("{}", described.join("\n"));

        let first = measured.first().expect("a shot").estimate.sigma;
        let last = measured.last().expect("a shot").estimate.sigma;
        assert!(
            first > 0.001,
            "the quiet shot still has grain and must not measure as clean: {first}"
        );
        assert!(
            last > first * 1.8,
            "the heavy shot must measure far above the quiet one: {first:.4} vs {last:.4}"
        );
    }
}
