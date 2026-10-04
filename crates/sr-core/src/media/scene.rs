//! Shot (scene-cut) detection — pure Rust, no model, no Python.
//!
//! Why this must be native and must exist before any AI stage: interpolation and
//! restoration models both fail catastrophically *across a cut*. A single
//! synthesised frame between "a man's face" and "a car" is a visible artefact,
//! and a diffusion model asked to restore a frame that contains two shots will
//! hallucinate something that is in neither.
//!
//! The detector works on a small grayscale raster (FFmpeg does the scaling) and
//! combines three independent signals:
//!
//! * 32-bin luma histogram distance — catches content changes
//! * edge-map difference — catches structural changes with identical histograms
//! * mean absolute luma difference — catches exposure changes
//!
//! The threshold is *adaptive*: it is derived from the median and MAD of the
//! recent score history, so a dark, grainy film and a bright, clean one both get
//! a sensible threshold without a human tuning anything. Flash frames (camera
//! flashes, lightning, explosions) produce a single-frame spike and are rejected
//! by both the MAD-based threshold and the minimum shot length guard.

use crate::error::{Error, Result};
use crate::events::{Reporter, Stage, StageProgress};
use crate::ffmpeg::{args, read_exact_or_eof, Ffmpeg, RunSpec, StreamingChild};
use crate::media::manifest::MediaManifest;
use crate::time::{Rational, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Analysis raster width. 160 px is enough for a cut to be obvious and cheap
/// enough that analysing a two hour film takes seconds of CPU.
pub const ANALYSIS_WIDTH: usize = 160;

pub const HISTOGRAM_BINS: usize = 32;

#[derive(Clone, Debug)]
pub struct GrayFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl GrayFrame {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        GrayFrame {
            width,
            height,
            data,
        }
    }

    pub fn pixels(&self) -> usize {
        self.width * self.height
    }

    /// Normalised 32-bin luma histogram.
    pub fn histogram(&self) -> [f32; HISTOGRAM_BINS] {
        let mut hist = [0f32; HISTOGRAM_BINS];
        if self.data.is_empty() {
            return hist;
        }
        for &px in &self.data {
            let bin = (px as usize * HISTOGRAM_BINS) / 256;
            hist[bin.min(HISTOGRAM_BINS - 1)] += 1.0;
        }
        let n = self.data.len() as f32;
        for v in hist.iter_mut() {
            *v /= n;
        }
        hist
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|&p| p as f32).sum::<f32>() / self.data.len() as f32
    }

    /// Simple gradient-magnitude edge map, 0 or 1 per pixel.
    pub fn edge_map(&self, threshold: u8) -> Vec<u8> {
        let (w, h) = (self.width, self.height);
        let mut out = vec![0u8; w * h];
        if w < 3 || h < 3 {
            return out;
        }
        let at = |x: usize, y: usize| self.data[y * w + x] as i32;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let gx = (at(x + 1, y) - at(x - 1, y)).abs();
                let gy = (at(x, y + 1) - at(x, y - 1)).abs();
                if (gx + gy) as u8 >= threshold {
                    out[y * w + x] = 1;
                }
            }
        }
        out
    }

    /// Mean absolute difference against another frame of the same size.
    pub fn mean_abs_diff(&self, other: &GrayFrame) -> f32 {
        if self.data.len() != other.data.len() || self.data.is_empty() {
            return 0.0;
        }
        let sum: u64 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
            .sum();
        sum as f32 / self.data.len() as f32
    }
}

/// Half the L1 distance between two normalised histograms: 0 identical, 1 disjoint.
pub fn histogram_distance(a: &[f32; HISTOGRAM_BINS], b: &[f32; HISTOGRAM_BINS]) -> f32 {
    let mut sum = 0.0;
    for i in 0..HISTOGRAM_BINS {
        sum += (a[i] - b[i]).abs();
    }
    (sum / 2.0).clamp(0.0, 1.0)
}

/// Fraction of pixels whose edge classification changed.
pub fn edge_distance(a: &[u8], b: &[u8]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let changed = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    changed as f32 / a.len() as f32
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrameDiff {
    pub histogram: f32,
    pub edge: f32,
    /// Luma MAD in 0..255 units.
    pub mad: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneOptions {
    /// Minimum frames between two cuts; also the flash-frame guard.
    pub min_shot_frames: u64,
    /// Absolute histogram distance that always counts as a cut.
    pub hard_histogram_threshold: f32,
    /// Absolute MAD (0..255) that a cut must also reach.
    pub mad_threshold: f32,
    pub adaptive_window: usize,
    /// Cut threshold = median + k * MAD of the recent score history.
    pub adaptive_k: f32,
    /// Floor for the adaptive threshold, so flat content cannot set it to zero.
    pub min_threshold: f32,
    pub edge_gradient_threshold: u8,
    /// Sequential elevated scores that mark a gradual transition (dissolve).
    pub dissolve_run: usize,
    /// Ratio of the cut threshold that counts as "elevated".
    pub dissolve_level: f32,
    /// Read real PTS with `showinfo` instead of deriving from the frame rate.
    pub accurate_timestamps: bool,
}

impl Default for SceneOptions {
    fn default() -> Self {
        SceneOptions {
            min_shot_frames: 6,
            hard_histogram_threshold: 0.52,
            mad_threshold: 12.0,
            adaptive_window: 24,
            adaptive_k: 6.0,
            min_threshold: 0.18,
            edge_gradient_threshold: 24,
            dissolve_run: 4,
            dissolve_level: 0.35,
            accurate_timestamps: false,
        }
    }
}

/// What the detector concluded about one frame.
#[derive(Clone, Debug)]
pub struct FrameVerdict {
    pub frame_index: u64,
    pub is_cut: bool,
    pub diff: FrameDiff,
    pub threshold: f32,
}

/// Streaming cut detector. Feed frames in presentation order.
///
/// A candidate cut is held for one frame before it is recorded: a real cut stays
/// changed on the next frame, while a camera flash, a lightning strike or an
/// explosion returns to the previous content. Without that check every flash
/// would disable interpolation around it.
pub struct SceneDetector {
    opts: SceneOptions,
    prev: Option<GrayFrame>,
    prev_edges: Option<Vec<u8>>,
    recent_scores: VecDeque<f32>,
    frame_index: u64,
    frames_since_cut: u64,
    elevated_run: usize,
    pending: Option<PendingCut>,
    cuts: Vec<(u64, f32, bool)>,
}

struct PendingCut {
    index: u64,
    score: f32,
    dissolve: bool,
    /// The frame immediately before the candidate, used to prove the change
    /// persisted.
    reference: GrayFrame,
}

/// Difference between two frames, using the same three signals as the detector.
fn compare(a: &GrayFrame, b: &GrayFrame, opts: &SceneOptions) -> FrameDiff {
    FrameDiff {
        histogram: histogram_distance(&a.histogram(), &b.histogram()),
        edge: edge_distance(
            &a.edge_map(opts.edge_gradient_threshold),
            &b.edge_map(opts.edge_gradient_threshold),
        ),
        mad: a.mean_abs_diff(b),
    }
}

impl SceneDetector {
    pub fn new(opts: SceneOptions) -> Self {
        let window = opts.adaptive_window.max(4);
        SceneDetector {
            opts,
            prev: None,
            prev_edges: None,
            recent_scores: VecDeque::with_capacity(window + 1),
            frame_index: 0,
            frames_since_cut: u64::MAX,
            elevated_run: 0,
            pending: None,
            cuts: Vec::new(),
        }
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn cuts(&self) -> &[(u64, f32, bool)] {
        &self.cuts
    }

    /// Combined score: a cut needs a content change *and* a pixel change, so a
    /// global fade or a uniform exposure shift does not register as one.
    fn score(diff: &FrameDiff) -> f32 {
        (diff.histogram * 0.5 + diff.edge.min(0.5) * 0.4 + (diff.mad / 255.0) * 0.6)
            .clamp(0.0, 1.0)
    }

    fn adaptive_threshold(&self) -> f32 {
        if self.recent_scores.len() < 4 {
            return self.opts.hard_histogram_threshold;
        }
        let mut sorted: Vec<f32> = self.recent_scores.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let deviations: Vec<f32> = sorted.iter().map(|s| (s - median).abs()).collect();
        let mut dev_sorted = deviations;
        dev_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = dev_sorted[dev_sorted.len() / 2];
        (median + self.opts.adaptive_k * mad).max(self.opts.min_threshold)
    }

    /// `is_cut` is reported on the frame *after* the boundary, once the change
    /// has been confirmed to persist. Boundary indices are the true ones.
    pub fn push(&mut self, frame: &GrayFrame) -> FrameVerdict {
        let index = self.frame_index;
        self.frame_index += 1;

        // (1) Resolve a candidate held from the previous frame.
        let mut confirmed = false;
        // The frame the previous one should be compared against, when a candidate
        // turned out to be a flash.
        let mut pre_flash: Option<GrayFrame> = None;
        if let Some(pending) = self.pending.take() {
            let persists = compare(&pending.reference, frame, &self.opts);
            if persists.mad >= self.opts.mad_threshold
                || persists.histogram >= self.opts.hard_histogram_threshold * 0.6
            {
                self.cuts.push((pending.index, pending.score, pending.dissolve));
                self.frames_since_cut = 0;
                self.recent_scores.clear();
                self.elevated_run = 0;
                confirmed = true;
            } else {
                // It was a flash: nothing is recorded and the score history
                // continues undisturbed.
                //
                // The flash frame must not become the reference for the *next*
                // comparison either, which is what let flashes through before. The
                // transition out of a flash is as large as the transition into it,
                // so the pair (flash, next) looks exactly like a cut from a bright
                // scene to a dark one — and it is confirmed by its own persistence
                // check, because the frame after it resembles the flash no more than
                // the frame before it did. Holding the pre-flash frame as the
                // reference makes the burst invisible, which is what it is.
                pre_flash = Some(pending.reference);
            }
        }

        // (2) Compare with the previous frame — or with the frame before the flash,
        // when this one followed a flash.
        //
        // Skipping the flash frame here matters as much as skipping it as a
        // reference: comparing against it arms a *new* candidate on the way out of
        // the burst, whose persistence check then compares the frame after against
        // the flash and confirms it. Rejecting the leading edge while arming the
        // trailing one is how a flash became a cut.
        let (against, against_edges) = match &pre_flash {
            Some(pre_flash) => (
                Some(pre_flash),
                Some(pre_flash.edge_map(self.opts.edge_gradient_threshold)),
            ),
            None => (self.prev.as_ref(), self.prev_edges.clone()),
        };
        let (diff, threshold) = match (against, &against_edges) {
            (Some(prev), Some(prev_edges)) => {
                let hist = histogram_distance(&prev.histogram(), &frame.histogram());
                let edges = frame.edge_map(self.opts.edge_gradient_threshold);
                let edge = edge_distance(prev_edges, &edges);
                let mad = prev.mean_abs_diff(frame);
                (
                    FrameDiff {
                        histogram: hist,
                        edge,
                        mad,
                    },
                    self.adaptive_threshold(),
                )
            }
            _ => (
                FrameDiff::default(),
                self.opts.hard_histogram_threshold,
            ),
        };

        let score = Self::score(&diff);
        let changed_a_lot = diff.histogram >= self.opts.hard_histogram_threshold
            && diff.mad >= self.opts.mad_threshold;
        let adaptive_cut = score >= threshold && diff.mad >= self.opts.mad_threshold;
        let long_enough = self.frames_since_cut >= self.opts.min_shot_frames;
        let is_candidate = self.prev.is_some()
            && long_enough
            && self.pending.is_none()
            && (changed_a_lot || adaptive_cut);

        if is_candidate {
            let reference = self.prev.clone().expect("candidate requires a previous frame");
            self.pending = Some(PendingCut {
                index,
                score,
                dissolve: self.elevated_run >= self.opts.dissolve_run,
                reference,
            });
        } else {
            self.frames_since_cut = self.frames_since_cut.saturating_add(1);
            if score >= threshold * self.opts.dissolve_level {
                self.elevated_run += 1;
            } else {
                self.elevated_run = 0;
            }
            if self.recent_scores.len() == self.opts.adaptive_window {
                self.recent_scores.pop_front();
            }
            self.recent_scores.push_back(score);
        }

        // (3) Remember this frame, unless it was a flash — in which case the frame
        // before it stays the reference, so the burst leaves no trace.
        let (reference, edges) = match pre_flash {
            Some(pre_flash) => {
                let edges = pre_flash.edge_map(self.opts.edge_gradient_threshold);
                (pre_flash, edges)
            }
            None => (
                GrayFrame::new(frame.width, frame.height, frame.data.clone()),
                self.prev_edges
                    .clone()
                    .unwrap_or_else(|| frame.edge_map(self.opts.edge_gradient_threshold)),
            ),
        };
        self.prev = Some(reference);
        self.prev_edges = Some(edges);

        FrameVerdict {
            frame_index: index,
            is_cut: confirmed,
            diff,
            threshold,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShotKind {
    /// First shot in the file.
    Start,
    /// Hard cut into this shot.
    Cut,
    /// Gradual transition into this shot (dissolve/fade).
    Dissolve,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shot {
    pub index: usize,
    pub start_frame: u64,
    pub end_frame: u64,
    pub start: Timestamp,
    pub end: Timestamp,
    pub kind: ShotKind,
    /// Detector score at the boundary that opened this shot.
    pub cut_score: Option<f32>,
    pub duration_seconds: f64,
}

impl Shot {
    pub fn frame_count(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame).max(1)
    }

    pub fn label(&self) -> String {
        format!(
            "shot {:04}  {} → {}  ({:.2}s)",
            self.index,
            self.start.format_hms(),
            self.end.format_hms(),
            self.duration_seconds
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneReport {
    pub shots: Vec<Shot>,
    pub frames_analyzed: u64,
    pub analysis_width: usize,
    pub analysis_height: usize,
    pub fps: Rational,
    pub timebase: Rational,
    pub accurate_timestamps: bool,
    pub notes: Vec<String>,
}

impl SceneReport {
    pub fn cut_count(&self) -> usize {
        self.shots.len().saturating_sub(1)
    }

    pub fn median_shot_seconds(&self) -> f64 {
        if self.shots.is_empty() {
            return 0.0;
        }
        let mut durations: Vec<f64> = self.shots.iter().map(|s| s.duration_seconds).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        durations[durations.len() / 2]
    }

    pub fn summary(&self) -> String {
        format!(
            "{} shots ({} cuts, median {:.2}s) from {} frames",
            self.shots.len(),
            self.cut_count(),
            self.median_shot_seconds(),
            self.frames_analyzed
        )
    }

    /// True when `frame` sits on a shot boundary (interpolation must not cross).
    pub fn is_boundary_frame(&self, frame: u64) -> bool {
        self.shots
            .iter()
            .any(|s| s.start_frame == frame || s.end_frame == frame)
    }

    pub fn shot_at_frame(&self, frame: u64) -> Option<&Shot> {
        self.shots
            .iter()
            .find(|s| frame >= s.start_frame && frame < s.end_frame)
    }
}

/// Builds the shot list from cut positions and a frame → timestamp mapping.
pub fn build_shots(
    cuts: &[(u64, f32, bool)],
    total_frames: u64,
    frame_time: impl Fn(u64) -> Timestamp,
) -> Vec<Shot> {
    let mut shots = Vec::with_capacity(cuts.len() + 1);
    let mut start_frame = 0u64;
    let mut kind = ShotKind::Start;
    let mut score = None;

    for (idx, cut_score, dissolve) in cuts {
        if *idx <= start_frame {
            continue;
        }
        shots.push(Shot {
            index: shots.len(),
            start_frame,
            end_frame: *idx,
            start: frame_time(start_frame),
            end: frame_time(*idx),
            kind,
            cut_score: score,
            duration_seconds: frame_time(*idx).seconds_f64() - frame_time(start_frame).seconds_f64(),
        });
        start_frame = *idx;
        kind = if *dissolve {
            ShotKind::Dissolve
        } else {
            ShotKind::Cut
        };
        score = Some(*cut_score);
    }

    let end_frame = total_frames.max(start_frame + 1);
    shots.push(Shot {
        index: shots.len(),
        start_frame,
        end_frame,
        start: frame_time(start_frame),
        end: frame_time(end_frame),
        kind,
        cut_score: score,
        duration_seconds: frame_time(end_frame).seconds_f64() - frame_time(start_frame).seconds_f64(),
    });
    shots
}

/// Which stream the shot boundaries describe.
///
/// A shot list is only meaningful relative to a frame numbering. Analysing the
/// *decoded* stream — after the same IVTC/deinterlace chain the encoder uses —
/// is what makes [`Shot::start_frame`] mean "the Nth frame the model will see".
/// On a 3:2 telecine source the raw and the decoded numbering differ by 20%, and
/// a shot list that is off by 20% protects nothing.
#[derive(Clone, Debug)]
pub struct SceneDecode<'a> {
    /// Filters applied before analysis. Must be the temporal chain the encode
    /// path uses, or the frame indices will not line up with the model's input.
    pub pre_chain: Option<&'a str>,
    /// Frame rate after the pre-chain: inverse telecine changes it.
    pub fps: Rational,
    /// Why this chain was chosen, for the log.
    pub reason: &'a str,
}

impl<'a> SceneDecode<'a> {
    /// Analyses the file as stored. Correct when no cadence change is applied.
    pub fn raw(fps: Rational) -> Self {
        SceneDecode {
            pre_chain: None,
            fps,
            reason: "as stored",
        }
    }
}

/// Decodes a small grayscale raster and runs the detector over it.
pub fn detect_scenes(
    ff: &Ffmpeg,
    manifest: &MediaManifest,
    reporter: &Reporter,
    cancel: &AtomicBool,
    opts: &SceneOptions,
    decode: SceneDecode<'_>,
) -> Result<SceneReport> {
    let video = manifest
        .primary_video()
        .ok_or_else(|| Error::Unsupported("no video stream to analyse".into()))?;
    let fps = decode.fps;
    let tb = fps.inverse().unwrap_or(Rational::ONE);

    let (src_w, src_h) = video
        .square_pixel_size()
        .ok_or_else(|| Error::Unsupported("unknown video dimensions".into()))?;

    // Analysis raster: fixed width, aspect preserved, even height, 8-bit gray.
    let analysis_w = ANALYSIS_WIDTH;
    let analysis_h = (((src_h as f64 / src_w as f64) * analysis_w as f64).round() as usize).max(2);
    let analysis_h = if analysis_h % 2 == 1 { analysis_h + 1 } else { analysis_h };

    let mut filters = String::new();
    if let Some(pre) = decode.pre_chain.filter(|p| !p.is_empty()) {
        // The cadence chain runs first so that shot indices count *decoded*
        // frames.
        filters.push_str(pre);
        filters.push(',');
    }
    filters.push_str(&format!(
        "scale={analysis_w}:{analysis_h}:flags=fast_bilinear,format=gray"
    ));
    if opts.accurate_timestamps {
        filters.push_str(",showinfo");
    }

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

    let pts_times: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink_pts = Arc::clone(&pts_times);
    let expected_frames = video.base.nb_frames.or_else(|| video.expected_frames());

    let spec = RunSpec::new(Stage::Scenes, "scene-detect")
        .quiet("showinfo")
        .quiet("Parsed_showinfo")
        .stderr_level(crate::events::Level::Debug)
        .with_sink(move |line: &str| {
            if let Some(rest) = line.split("pts_time:").nth(1) {
                if let Some(token) = rest.split_whitespace().next() {
                    if let Ok(v) = token.parse::<f64>() {
                        sink_pts.lock().expect("pts sink").push(v);
                    }
                }
            }
        });

    reporter.info(
        Some(Stage::Scenes),
        format!(
            "scanning {} frames at {analysis_w}x{analysis_h} gray ({}), cadence: {}{}",
            expected_frames
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".into()),
            decode.reason,
            match decode.pre_chain.filter(|p| !p.is_empty()) {
                Some(chain) => chain,
                None => "unchanged",
            },
            if opts.accurate_timestamps {
                " with real PTS"
            } else {
                ""
            }
        ),
    );

    let mut child = StreamingChild::spawn(&ff.ffmpeg, &argv, reporter, &spec, false)?;
    let mut stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            child.abort();
            return Err(Error::Stage {
                stage: Stage::Scenes.id().into(),
                detail: "ffmpeg produced no stdout pipe".into(),
            });
        }
    };

    let frame_bytes = analysis_w * analysis_h;
    let mut buffer = vec![0u8; frame_bytes];
    let mut detector = SceneDetector::new(opts.clone());
    let mut frames: u64 = 0;
    let mut last_report = std::time::Instant::now();

    let read_result: Result<()> = loop {
        if let Err(err) = child.check_cancel(cancel) {
            child.kill();
            break Err(err);
        }
        match read_exact_or_eof(&mut stdout, &mut buffer) {
            Ok(0) => break Ok(()),
            Ok(n) if n < frame_bytes => {
                // A trailing partial frame means the stream ended mid-frame.
                break Ok(());
            }
            Ok(_) => {
                let frame = GrayFrame::new(analysis_w, analysis_h, buffer.clone());
                detector.push(&frame);
                frames += 1;
                if last_report.elapsed() >= Duration::from_millis(400) {
                    last_report = std::time::Instant::now();
                    reporter.progress(StageProgress {
                        stage: Stage::Scenes,
                        fraction: expected_frames
                            .filter(|n| *n > 0)
                            .map(|n| frames as f32 / n as f32),
                        detail: format!("{frames} frames · {} cuts", detector.cuts().len()),
                        frames: Some(frames),
                        fps: None,
                        speed: None,
                        out_time: None,
                        eta: None,
                    });
                }
            }
            Err(err) => {
                child.kill();
                break Err(err);
            }
        }
    };

    let outcome = child.wait();
    read_result?;
    outcome?;

    let pts = pts_times.lock().expect("pts sink").clone();
    let mut notes = Vec::new();
    let accurate = opts.accurate_timestamps && pts.len() >= frames as usize && frames > 0;
    if opts.accurate_timestamps && !accurate {
        notes.push(format!(
            "requested real timestamps but got {} PTS for {} frames: falling back to index/fps",
            pts.len(),
            frames
        ));
    }

    // Frame index -> timestamp. With real PTS we keep exact microseconds; the
    // frame-rate path is exact only because it stays rational.
    let us_tb = Rational::new(1, 1_000_000).unwrap_or(Rational::ONE);
    let frame_time = |frame: u64| -> Timestamp {
        if accurate {
            let value = pts.get(frame as usize).copied().unwrap_or_else(|| {
                pts.last().copied().unwrap_or(0.0)
                    + (frame as f64 - pts.len() as f64 + 1.0) / fps.to_f64()
            });
            Timestamp::new((value * 1_000_000.0).round() as i64, us_tb)
        } else {
            Timestamp::new(frame as i64, tb)
        }
    };

    let shots = build_shots(detector.cuts(), frames, frame_time);

    if frames == 0 {
        notes.push("no frames were decoded for analysis".to_string());
    }
    if shots.len() == 1 && frames > 1 {
        notes.push("no cuts detected: the runtime looks like one continuous shot".to_string());
    }
    let dissolve_count = shots
        .iter()
        .filter(|s| s.kind == ShotKind::Dissolve)
        .count();
    if dissolve_count > 0 {
        notes.push(format!(
            "{dissolve_count} gradual transitions detected; interpolation stays disabled across them"
        ));
    }

    let report = SceneReport {
        shots,
        frames_analyzed: frames,
        analysis_width: analysis_w,
        analysis_height: analysis_h,
        fps,
        timebase: if accurate { us_tb } else { tb },
        accurate_timestamps: accurate,
        notes,
    };
    reporter.info(Some(Stage::Scenes), report.summary());
    Ok(report)
}

/// Frame index for a timestamp, used by the encode stage to place boundaries.
pub fn frame_at_time(t: Timestamp, fps: Rational) -> u64 {
    let seconds = t.seconds_f64();
    (seconds * fps.to_f64()).round().max(0.0) as u64
}

/// Convenience for the pipeline: does this report say "never interpolate across
/// a boundary here"?
pub fn boundary_time_set(report: &SceneReport) -> Vec<Timestamp> {
    report.shots.iter().skip(1).map(|s| s.start).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(value: u8, w: usize, h: usize) -> GrayFrame {
        GrayFrame::new(w, h, vec![value; w * h])
    }

    /// A frame with a vertical split: left half dark, right half bright.
    fn split(left: u8, right: u8, w: usize, h: usize) -> GrayFrame {
        let mut data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if x < w / 2 { left } else { right };
            }
        }
        GrayFrame::new(w, h, data)
    }

    /// A frame with a horizontal split: top dark, bottom bright. Used to check
    /// that the edge signal reacts to *topology*, not just to pixel counts.
    fn split_h(top: u8, bottom: u8, w: usize, h: usize) -> GrayFrame {
        let mut data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if y < h / 2 { top } else { bottom };
            }
        }
        GrayFrame::new(w, h, data)
    }

    #[test]
    fn histogram_distance_is_a_metric() {
        let a = flat(10, 16, 16).histogram();
        let b = flat(10, 16, 16).histogram();
        let c = flat(200, 16, 16).histogram();
        assert!(histogram_distance(&a, &b) < 1e-6);
        assert!(histogram_distance(&a, &c) > 0.9);
    }

    #[test]
    fn edge_map_reacts_to_a_change_of_topology() {
        // A vertical split and a horizontal split have identical histograms but
        // different edge structure: only the edge signal can separate them.
        let vertical = split(0, 255, 32, 32);
        let horizontal = split_h(0, 255, 32, 32);
        let ev = vertical.edge_map(24);
        let eh = horizontal.edge_map(24);
        assert!(ev.iter().any(|&v| v == 1), "split must produce edges");
        assert!(eh.iter().any(|&v| v == 1));
        assert!(
            edge_distance(&ev, &eh) > 0.02,
            "different edge layouts must differ"
        );
        assert_eq!(edge_distance(&ev, &ev), 0.0);
        assert_eq!(histogram_distance(&vertical.histogram(), &horizontal.histogram()), 0.0);
    }

    #[test]
    fn static_frames_never_produce_a_cut() {
        let mut d = SceneDetector::new(SceneOptions::default());
        for _ in 0..60 {
            let v = d.push(&flat(128, 32, 32));
            assert!(!v.is_cut);
        }
        assert!(d.cuts().is_empty());
    }

    #[test]
    fn a_hard_content_change_produces_exactly_one_cut() {
        let mut d = SceneDetector::new(SceneOptions::default());
        for _ in 0..20 {
            d.push(&flat(20, 32, 32));
        }
        let mut cuts = 0;
        for _ in 0..20 {
            if d.push(&flat(230, 32, 32)).is_cut {
                cuts += 1;
            }
        }
        assert_eq!(cuts, 1, "one scene change must yield one cut");
        assert_eq!(d.cuts().len(), 1);
    }

    #[test]
    fn a_single_frame_flash_is_not_a_cut() {
        // Camera flashes and lightning are one-frame spikes; treating them as
        // cuts would disable interpolation around every explosion.
        let mut d = SceneDetector::new(SceneOptions::default());
        for _ in 0..20 {
            d.push(&flat(60, 32, 32));
        }
        let flash = d.push(&flat(250, 32, 32));
        let back = d.push(&flat(60, 32, 32));
        assert!(!flash.is_cut);
        assert!(!back.is_cut, "the frame after a flash must not become a cut");
        assert!(d.cuts().is_empty());
    }

    #[test]
    fn a_slow_fade_does_not_register_as_a_cut() {
        let mut d = SceneDetector::new(SceneOptions::default());
        let mut value = 10u8;
        let mut cuts = 0;
        for _ in 0..120 {
            if d.push(&flat(value, 32, 32)).is_cut {
                cuts += 1;
            }
            value = value.saturating_add(2);
        }
        assert_eq!(cuts, 0, "a fade is not a cut");
    }

    #[test]
    fn cuts_closer_than_the_minimum_shot_length_are_suppressed() {
        let mut opts = SceneOptions::default();
        opts.min_shot_frames = 10;
        let mut d = SceneDetector::new(opts);
        d.push(&flat(10, 32, 32));
        let mut cuts = 0;
        // alternate content every frame: only the first can be accepted
        for i in 0..8 {
            let v = if i % 2 == 0 { 240 } else { 10 };
            if d.push(&flat(v, 32, 32)).is_cut {
                cuts += 1;
            }
        }
        assert!(cuts <= 1, "rapid alternation must not spray cuts, got {cuts}");
    }

    #[test]
    fn structural_change_with_similar_histogram_is_detected() {
        // Same amount of dark and bright pixels, different layout. The histogram
        // is identical, so detection has to come from the pixel difference.
        let mut d = SceneDetector::new(SceneOptions::default());
        for _ in 0..20 {
            d.push(&split(0, 255, 32, 32));
        }
        let mut cuts = 0;
        for _ in 0..20 {
            if d.push(&split_h(0, 255, 32, 32)).is_cut {
                cuts += 1;
            }
        }
        assert_eq!(cuts, 1, "layout change must be detected");
    }

    #[test]
    fn an_inverted_frame_with_the_same_histogram_is_a_cut() {
        // Bright/dark halves swapped: identical histogram and identical edge map,
        // so only the luma difference can see it.
        let mut d = SceneDetector::new(SceneOptions::default());
        for _ in 0..20 {
            d.push(&split(0, 255, 32, 32));
        }
        let mut cuts = 0;
        for _ in 0..20 {
            if d.push(&split(255, 0, 32, 32)).is_cut {
                cuts += 1;
            }
        }
        assert_eq!(cuts, 1, "an inversion is a real cut");
    }

    #[test]
    fn shots_are_contiguous_and_cover_the_runtime() {
        let tb = Rational::new(1, 24).unwrap();
        let cuts = vec![(100u64, 0.9f32, false), (250, 0.8, true)];
        let shots = build_shots(&cuts, 400, |f| Timestamp::new(f as i64, tb));
        assert_eq!(shots.len(), 3);
        assert_eq!(shots[0].kind, ShotKind::Start);
        assert_eq!(shots[1].kind, ShotKind::Cut);
        assert_eq!(shots[2].kind, ShotKind::Dissolve);
        for pair in shots.windows(2) {
            assert_eq!(pair[0].end_frame, pair[1].start_frame, "shots must abut");
        }
        assert_eq!(shots[0].start_frame, 0);
        assert_eq!(shots.last().unwrap().end_frame, 400);
        assert!((shots[0].duration_seconds - 100.0 / 24.0).abs() < 1e-6);
    }

    #[test]
    fn a_single_shot_report_has_no_cuts() {
        let tb = Rational::new(1, 25).unwrap();
        let shots = build_shots(&[], 250, |f| Timestamp::new(f as i64, tb));
        assert_eq!(shots.len(), 1);
        assert_eq!(shots[0].duration_seconds, 10.0);
    }

    #[test]
    fn boundary_lookup_matches_the_shot_list() {
        let tb = Rational::new(1, 24).unwrap();
        let shots = build_shots(&[(48u64, 0.9f32, false)], 96, |f| {
            Timestamp::new(f as i64, tb)
        });
        let report = SceneReport {
            shots,
            frames_analyzed: 96,
            analysis_width: 160,
            analysis_height: 90,
            fps: Rational::from_i64(24),
            timebase: tb,
            accurate_timestamps: false,
            notes: vec![],
        };
        assert!(report.is_boundary_frame(48));
        assert!(!report.is_boundary_frame(47));
        assert_eq!(report.shot_at_frame(10).unwrap().index, 0);
        assert_eq!(report.shot_at_frame(50).unwrap().index, 1);
        assert_eq!(boundary_time_set(&report).len(), 1);
    }

    #[test]
    fn frame_at_time_round_trips() {
        let fps = Rational::new(24000, 1001).unwrap();
        let tb = fps.inverse().unwrap();
        let t = Timestamp::new(1000, tb);
        assert_eq!(frame_at_time(t, fps), 1000);
    }
}
