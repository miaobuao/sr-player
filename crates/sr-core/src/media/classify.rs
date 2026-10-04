//! Temporal classification: is this progressive, telecined, or genuinely
//! interlaced?
//!
//! `29.970 fps` on a DVD can mean at least five different things, and picking
//! wrong is unrecoverable: run `decimate` on genuinely interlaced video and you
//! destroy half the temporal information; run nothing on 3:2 pulldown and the
//! restoration model sees combed frames for the whole film.
//!
//! So we do not trust the container. We sample the picture at several points
//! across the runtime with FFmpeg's `idet`, aggregate the field statistics, and
//! classify from the frames themselves. The container's `field_order` flag is
//! only used to raise a disagreement note — it lies often enough to matter.

use crate::error::Result;
use crate::events::{Reporter, Stage, StageProgress};
use crate::ffmpeg::{args, run_tool, Ffmpeg, RunSpec};
use crate::media::manifest::MediaManifest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalMode {
    /// Clean progressive frames at the declared rate.
    Progressive,
    /// Film frames carried in interlaced fields (3:2 pulldown and friends).
    /// Requires field matching + decimation, not deinterlacing.
    Telecine,
    /// Real interlaced capture (video-origin material). Requires deinterlacing;
    /// decimating this would throw away half of the motion.
    Interlaced,
    /// Both behaviours present in different parts of the runtime.
    Mixed,
    /// Not enough signal to decide.
    Unknown,
}

impl TemporalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            TemporalMode::Progressive => "progressive",
            TemporalMode::Telecine => "telecine",
            TemporalMode::Interlaced => "interlaced",
            TemporalMode::Mixed => "mixed",
            TemporalMode::Unknown => "unknown",
        }
    }

    /// Field matching then decimation.
    pub fn needs_ivtc(self) -> bool {
        matches!(self, TemporalMode::Telecine)
    }

    /// Motion-adaptive deinterlacing.
    pub fn needs_deinterlace(self) -> bool {
        matches!(self, TemporalMode::Interlaced | TemporalMode::Mixed)
    }

    /// Conservative answer for the pipeline: when we do not know, we do the
    /// least destructive thing available (selective deinterlace, never drop).
    pub fn is_confident(self) -> bool {
        !matches!(self, TemporalMode::Unknown)
    }
}

/// Aggregated `idet` counters. Field names mirror FFmpeg's own vocabulary.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IdetCounts {
    pub tff: u64,
    pub bff: u64,
    pub progressive: u64,
    pub undetermined: u64,
    pub single_tff: u64,
    pub single_bff: u64,
    pub single_progressive: u64,
    pub single_undetermined: u64,
    pub repeated_neither: u64,
    pub repeated_top: u64,
    pub repeated_bottom: u64,
    pub repeated_repeat: u64,
}

impl IdetCounts {
    pub fn add(&mut self, other: &IdetCounts) {
        self.tff += other.tff;
        self.bff += other.bff;
        self.progressive += other.progressive;
        self.undetermined += other.undetermined;
        self.single_tff += other.single_tff;
        self.single_bff += other.single_bff;
        self.single_progressive += other.single_progressive;
        self.single_undetermined += other.single_undetermined;
        self.repeated_neither += other.repeated_neither;
        self.repeated_top += other.repeated_top;
        self.repeated_bottom += other.repeated_bottom;
        self.repeated_repeat += other.repeated_repeat;
    }

    pub fn frames(&self) -> u64 {
        self.repeated_neither + self.repeated_top + self.repeated_bottom + self.repeated_repeat
    }

    pub fn multi_total(&self) -> u64 {
        self.tff + self.bff + self.progressive + self.undetermined
    }

    pub fn progressive_ratio(&self) -> f64 {
        ratio(self.progressive, self.multi_total())
    }

    pub fn interlaced_ratio(&self) -> f64 {
        ratio(self.tff + self.bff, self.multi_total())
    }

    /// Fraction of frames carrying a duplicated field — the telecine signature.
    pub fn repeated_field_ratio(&self) -> f64 {
        ratio(
            self.repeated_top + self.repeated_bottom + self.repeated_repeat,
            self.frames(),
        )
    }
}

fn ratio(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 / total as f64
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdetSample {
    pub at_seconds: f64,
    pub window_seconds: f64,
    pub counts: IdetCounts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalReport {
    pub mode: TemporalMode,
    pub confidence: f32,
    pub container_field_order: Option<String>,
    pub container_says_interlaced: bool,
    pub totals: IdetCounts,
    pub samples: Vec<IdetSample>,
    pub sampled_seconds: f64,
    pub notes: Vec<String>,
}

impl TemporalReport {
    pub fn progressive_ratio(&self) -> f64 {
        self.totals.progressive_ratio()
    }

    pub fn interlaced_ratio(&self) -> f64 {
        self.totals.interlaced_ratio()
    }

    pub fn repeated_field_ratio(&self) -> f64 {
        self.totals.repeated_field_ratio()
    }

    /// Frames the per-frame detector saw combing in, over every frame it looked
    /// at. This is the signal that can veto the aggregate.
    pub fn single_frame_combing_ratio(&self) -> f64 {
        let seen = self.totals.single_tff
            + self.totals.single_bff
            + self.totals.single_progressive
            + self.totals.single_undetermined;
        if seen == 0 {
            0.0
        } else {
            (self.totals.single_tff + self.totals.single_bff) as f64 / seen as f64
        }
    }

    /// What the plan builder and the UI both want in one line.
    ///
    /// The label comes first and the raw detector ratios after it in brackets,
    /// because the two can legitimately disagree — the per-frame veto in
    /// [`decide`] exists precisely for that case — and a line reading
    /// "progressive (interlaced 100%)" without that framing looks like a bug.
    pub fn summary(&self) -> String {
        format!(
            "{} ({:.0}% confident; idet aggregate: progressive {:.0}%, interlaced {:.0}%, \
             repeated fields {:.0}%; per-frame combing {:.0}%)",
            self.mode.as_str(),
            self.confidence * 100.0,
            self.progressive_ratio() * 100.0,
            self.interlaced_ratio() * 100.0,
            self.repeated_field_ratio() * 100.0,
            self.single_frame_combing_ratio() * 100.0
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassifyOptions {
    /// Where in the runtime to sample, as fractions of the duration.
    pub fractions: Vec<f64>,
    pub window_seconds: f64,
    pub enabled: bool,
}

impl Default for ClassifyOptions {
    fn default() -> Self {
        ClassifyOptions {
            // Head and tail are often logos/credits with different cadence, so
            // they are sampled but not trusted alone.
            fractions: vec![0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99],
            window_seconds: 6.0,
            enabled: true,
        }
    }
}

/// Start times of the analysis windows, clamped into the runtime.
pub fn sample_points(duration_secs: f64, fractions: &[f64], window: f64) -> Vec<f64> {
    if duration_secs <= 0.0 {
        return vec![0.0];
    }
    if duration_secs <= window {
        return vec![0.0];
    }
    let last_start = (duration_secs - window).max(0.0);
    let mut points: Vec<f64> = fractions
        .iter()
        .map(|f| (duration_secs * f).clamp(0.0, last_start))
        .collect();
    points.push(0.0);
    // dedup with a tolerance of one second, then sort
    points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    points.dedup_by(|a, b| (*a - *b).abs() < 1.0);
    points
}

/// Parses one line of `idet` output.
///
/// `[Parsed_idet_0 @ 0x..] Repeated Fields: Neither: 100 Top: 0 Repeat: 0 Bottom: 0`
pub fn parse_idet_line(line: &str) -> Option<(&'static str, BTreeMap<String, u64>)> {
    let (kind, rest) = if let Some(rest) = line.split_once("Repeated Fields:").map(|(_, r)| r) {
        ("repeated", rest)
    } else if let Some(rest) = line
        .split_once("Single frame detection:")
        .map(|(_, r)| r)
    {
        ("single", rest)
    } else if let Some(rest) = line
        .split_once("Multi frame detection:")
        .map(|(_, r)| r)
    {
        ("multi", rest)
    } else {
        return None;
    };
    let counts = parse_labeled_counts(rest);
    if counts.is_empty() {
        None
    } else {
        Some((kind, counts))
    }
}

/// FFmpeg prints `Label: value Label: value` pairs, so each label ends one
/// colon-segment and its number starts the next one.
fn parse_labeled_counts(rest: &str) -> BTreeMap<String, u64> {
    let parts: Vec<&str> = rest.split(':').collect();
    let mut out = BTreeMap::new();
    for i in 1..parts.len() {
        let label = parts[i - 1]
            .split_whitespace()
            .last()
            .unwrap_or("")
            .to_ascii_lowercase();
        let value = parts[i]
            .split_whitespace()
            .next()
            .and_then(|v| v.parse::<u64>().ok());
        if let (false, Some(value)) = (label.is_empty(), value) {
            out.insert(label, value);
        }
    }
    out
}

fn apply_counts(target: &mut IdetCounts, kind: &str, counts: &BTreeMap<String, u64>) {
    let get = |k: &str| counts.get(k).copied().unwrap_or(0);
    match kind {
        "multi" => {
            target.tff += get("tff");
            target.bff += get("bff");
            target.progressive += get("progressive");
            target.undetermined += get("undetermined");
        }
        "single" => {
            target.single_tff += get("tff");
            target.single_bff += get("bff");
            target.single_progressive += get("progressive");
            target.single_undetermined += get("undetermined");
        }
        "repeated" => {
            target.repeated_neither += get("neither");
            target.repeated_top += get("top");
            target.repeated_bottom += get("bottom");
            target.repeated_repeat += get("repeat");
        }
        _ => {}
    }
}

/// How much of the single-frame detector's verdict has to agree before the frame
/// analysis overrules an explicit "progressive" in the container.
///
/// `idet`'s multi-frame detector is stateful: once a frame is called TFF the next
/// one is judged against it, so a single pathological pattern can colour the whole
/// file. `ffmpeg -f lavfi -i testsrc2 ... -c:v libx264` does exactly that — 24
/// progressive frames, reported as `Multi frame detection: TFF: 24` — while every
/// other synthetic source comes back undetermined. The per-frame detector is not
/// fooled the same way, so it gets a vote before a progressive file is
/// deinterlaced.
const SINGLE_FRAME_VETO: f64 = 0.50;

/// Classifies from aggregated statistics. Pure, so it is unit-tested directly.
pub fn decide(
    totals: &IdetCounts,
    declared_interlaced: bool,
    field_order: Option<&str>,
) -> (TemporalMode, f32, Vec<String>) {
    let mut notes = Vec::new();
    let frames = totals.frames();
    if frames == 0 || totals.multi_total() == 0 {
        notes.push("idet produced no usable frame statistics".to_string());
        return (TemporalMode::Unknown, 0.0, notes);
    }

    let progressive = totals.progressive_ratio();
    let mut interlaced = totals.interlaced_ratio();
    let repeated = totals.repeated_field_ratio();

    // A container that says "progressive" is evidence, and so is a per-frame
    // detector that mostly cannot see combing. Neither outranks the frames
    // themselves, but together they outrank one stateful aggregate.
    //
    // The denominator is every frame the per-frame detector looked at, not only
    // the ones it managed to classify: "undetermined" is the detector saying it
    // saw nothing, and it must not be counted as agreement with the aggregate.
    let single_seen =
        totals.single_tff + totals.single_bff + totals.single_progressive + totals.single_undetermined;
    let single_interlaced = if single_seen == 0 {
        1.0
    } else {
        (totals.single_tff + totals.single_bff) as f64 / single_seen as f64
    };
    let mut vetoed = false;
    if !declared_interlaced && interlaced >= 0.60 && single_interlaced < SINGLE_FRAME_VETO {
        notes.push(format!(
            "container field_order={} and the per-frame detector ({}% of {} frames showed \
             combing) disagree with the aggregate ({}% interlaced): treating this as progressive \
             rather than deinterlacing on one signal",
            field_order.unwrap_or("unknown"),
            (single_interlaced * 100.0).round(),
            single_seen,
            (interlaced * 100.0).round()
        ));
        interlaced = 0.0;
        vetoed = true;
    }

    // Telecine: duplicated fields are the fingerprint. 3:2 pulldown duplicates
    // one field in every 5-frame group, so even a clean transfer shows ~20%.
    let telecine = !vetoed && repeated >= 0.10 && interlaced >= 0.05;
    let mode = if telecine {
        TemporalMode::Telecine
    } else if progressive >= 0.90 {
        TemporalMode::Progressive
    } else if interlaced >= 0.60 {
        TemporalMode::Interlaced
    } else if progressive >= 0.60 && interlaced < 0.20 {
        TemporalMode::Progressive
    } else if vetoed {
        TemporalMode::Progressive
    } else {
        TemporalMode::Mixed
    };

    // Confidence: distance from the decision boundary, floored so a clear
    // answer never reads as "0% sure".
    let confidence = match mode {
        TemporalMode::Progressive => (0.5 + (progressive - 0.90) * 5.0).clamp(0.5, 1.0),
        TemporalMode::Telecine => (0.5 + (repeated - 0.10) * 5.0).clamp(0.5, 1.0),
        TemporalMode::Interlaced => (0.5 + (interlaced - 0.60) * 2.0).clamp(0.5, 1.0),
        TemporalMode::Mixed => (0.5 + (0.5 - (progressive - interlaced).abs()) * 2.0).clamp(0.4, 0.9),
        TemporalMode::Unknown => 0.0,
    };

    if declared_interlaced && mode == TemporalMode::Progressive {
        notes.push(
            "container claims interlaced but the frames are progressive: trusting the frames"
                .to_string(),
        );
    }
    if !declared_interlaced && !vetoed && matches!(mode, TemporalMode::Interlaced | TemporalMode::Telecine) {
        notes.push(format!(
            "container field_order={} disagrees with frame analysis ({}): trusting the frames",
            field_order.unwrap_or("unknown"),
            mode.as_str()
        ));
    }
    if totals.tff > 0 && totals.bff > 0 && mode == TemporalMode::Interlaced {
        notes.push("both field orders detected; field order may be inconsistent".to_string());
    }
    if mode == TemporalMode::Telecine {
        notes.push(
            "inverse telecine planned: field match, then decimate (never a plain deinterlace)"
                .to_string(),
        );
    }
    if mode == TemporalMode::Mixed {
        notes.push(
            "mixed cadence: selective deinterlace only, no frame dropping (lossy and irreversible)"
                .to_string(),
        );
    }
    let undetermined_ratio = ratio(totals.undetermined, totals.multi_total());
    if undetermined_ratio > 0.5 {
        notes.push(format!(
            "{:.0}% of sampled frames were undetermined (flat or dark content)",
            undetermined_ratio * 100.0
        ));
    }

    (mode, confidence as f32, notes)
}

/// Samples the source with `idet` and classifies the result.
pub fn classify(
    ff: &Ffmpeg,
    manifest: &MediaManifest,
    reporter: &Reporter,
    cancel: &AtomicBool,
    opts: &ClassifyOptions,
) -> Result<TemporalReport> {
    let video = match manifest.primary_video() {
        Some(v) => v,
        None => {
            return Ok(TemporalReport {
                mode: TemporalMode::Unknown,
                confidence: 0.0,
                container_field_order: None,
                container_says_interlaced: false,
                totals: IdetCounts::default(),
                samples: Vec::new(),
                sampled_seconds: 0.0,
                notes: vec!["no video stream to classify".to_string()],
            })
        }
    };
    let field_order = video.field_order.clone();
    let declared_interlaced = video.declared_interlaced();

    if !opts.enabled {
        return Ok(TemporalReport {
            mode: TemporalMode::Unknown,
            confidence: 0.0,
            container_field_order: field_order,
            container_says_interlaced: declared_interlaced,
            totals: IdetCounts::default(),
            samples: Vec::new(),
            sampled_seconds: 0.0,
            notes: vec!["cadence classification disabled by request".to_string()],
        });
    }

    if !ff.has_filter("idet") {
        return Ok(TemporalReport {
            mode: TemporalMode::Unknown,
            confidence: 0.0,
            container_field_order: field_order.clone(),
            container_says_interlaced: declared_interlaced,
            totals: IdetCounts::default(),
            samples: Vec::new(),
            sampled_seconds: 0.0,
            notes: vec![format!(
                "this FFmpeg build has no `idet` filter ({}): falling back to the container flag",
                ff.version
            )],
        });
    }

    let duration = manifest.duration_seconds();
    let points = sample_points(duration, &opts.fractions, opts.window_seconds);
    let mut samples = Vec::new();
    let mut totals = IdetCounts::default();

    for (i, at) in points.iter().enumerate() {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(crate::error::Error::Cancelled);
        }
        reporter.progress(StageProgress {
            stage: Stage::Temporal,
            fraction: Some(i as f32 / points.len() as f32),
            detail: format!("idet @ {at:.1}s ({}/{})", i + 1, points.len()),
            frames: None,
            fps: None,
            speed: None,
            out_time: None,
            eta: None,
        });

        let counts = Arc::new(Mutex::new(IdetCounts::default()));
        let sink_counts = Arc::clone(&counts);
        let window = opts.window_seconds.min(duration.max(1.0));
        let mut argv = args(&["-hide_banner", "-nostdin", "-progress", "pipe:1"]);
        argv.push("-ss".into());
        argv.push(format!("{at:.6}"));
        argv.push("-t".into());
        argv.push(format!("{window:.6}"));
        argv.push("-i".into());
        argv.push(manifest.path.display().to_string());
        argv.extend(args(&["-map", "0:v:0", "-vf", "idet", "-an", "-sn", "-f", "null"]));
        argv.push(if cfg!(windows) { "NUL".into() } else { "-".into() });

        let spec = RunSpec::new(Stage::Temporal, "idet")
            .with_duration(Duration::from_secs_f64(window))
            .quiet("idet")
            .stderr_level(crate::events::Level::Debug)
            .with_sink(move |line: &str| {
                if let Some((kind, parsed)) = parse_idet_line(line) {
                    let mut guard = sink_counts.lock().expect("idet counts");
                    apply_counts(&mut guard, kind, &parsed);
                }
            });

        match run_tool(&ff.ffmpeg, &argv, reporter, cancel, &spec) {
            Ok(_) => {
                let counts = counts.lock().expect("idet counts").clone();
                totals.add(&counts);
                samples.push(IdetSample {
                    at_seconds: *at,
                    window_seconds: window,
                    counts,
                });
            }
            Err(crate::error::Error::Cancelled) => return Err(crate::error::Error::Cancelled),
            Err(err) => {
                // A single unreadable sample should not sink the classification.
                reporter.warn(
                    Some(Stage::Temporal),
                    format!("idet sample at {at:.1}s failed: {err}"),
                );
            }
        }
    }

    let (mode, confidence, notes) = decide(&totals, declared_interlaced, field_order.as_deref());
    let report = TemporalReport {
        mode,
        confidence,
        container_field_order: field_order,
        container_says_interlaced: declared_interlaced,
        totals,
        sampled_seconds: samples.iter().map(|s| s.window_seconds).sum(),
        samples,
        notes,
    };

    reporter.info(
        Some(Stage::Temporal),
        format!(
            "cadence: {} (analysed {:.0}s across {} windows)",
            report.summary(),
            report.sampled_seconds,
            report.samples.len()
        ),
    );
    for note in &report.notes {
        reporter.warn(Some(Stage::Temporal), note.clone());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MULTI: &str = "[Parsed_idet_0 @ 0x55f0] Multi frame detection: TFF:    0 BFF:    0 Progressive:  298 Undetermined:    2";
    const SINGLE: &str = "[Parsed_idet_0 @ 0x55f0] Single frame detection: TFF:    0 BFF:    0 Progressive:  300 Undetermined:    0";
    const REPEATED: &str = "[Parsed_idet_0 @ 0x55f0] Repeated Fields: Neither:  300 Top:    0 Repeat:    0 Bottom:   0";

    #[test]
    fn parses_each_idet_line_shape() {
        let (kind, counts) = parse_idet_line(MULTI).unwrap();
        assert_eq!(kind, "multi");
        assert_eq!(counts["progressive"], 298);
        assert_eq!(counts["undetermined"], 2);
        assert_eq!(counts["tff"], 0);

        let (kind, counts) = parse_idet_line(SINGLE).unwrap();
        assert_eq!(kind, "single");
        assert_eq!(counts["progressive"], 300);

        let (kind, counts) = parse_idet_line(REPEATED).unwrap();
        assert_eq!(kind, "repeated");
        assert_eq!(counts["neither"], 300);
        assert_eq!(counts["bottom"], 0);
    }

    #[test]
    fn ignores_unrelated_lines() {
        assert!(parse_idet_line("frame=  120 fps=0.0 q=-0.0 size=N/A").is_none());
        assert!(parse_idet_line("").is_none());
    }

    fn counts(multi: (u64, u64, u64, u64), repeated: (u64, u64, u64, u64), _frames: u64) -> IdetCounts {
        IdetCounts {
            tff: multi.0,
            bff: multi.1,
            progressive: multi.2,
            undetermined: multi.3,
            repeated_neither: repeated.0,
            repeated_top: repeated.1,
            repeated_bottom: repeated.2,
            repeated_repeat: repeated.3,
            ..Default::default()
        }
    }

    /// The real numbers `ffmpeg -f lavfi -i testsrc2 -c:v libx264` produces:
    /// the aggregate says interlaced for every frame, the per-frame detector
    /// cannot see combing in most of them, and the container says progressive.
    fn with_single(
        mut base: IdetCounts,
        single: (u64, u64, u64, u64),
    ) -> IdetCounts {
        base.single_tff = single.0;
        base.single_bff = single.1;
        base.single_progressive = single.2;
        base.single_undetermined = single.3;
        base
    }

    #[test]
    fn a_progressive_container_and_weak_per_frame_evidence_beat_the_aggregate() {
        let c = with_single(
            counts((72, 0, 0, 0), (72, 0, 0, 0), 72),
            (23, 0, 4, 45),
        );
        let (mode, _, notes) = decide(&c, false, Some("progressive"));
        assert_eq!(
            mode,
            TemporalMode::Progressive,
            "a stateful aggregate must not deinterlace a file whose container says \
             progressive and whose frames mostly show no combing: {notes:?}"
        );
        assert!(!mode.needs_deinterlace());
        assert!(
            notes.iter().any(|n| n.contains("per-frame detector")),
            "the decision must be explained: {notes:?}"
        );
    }

    #[test]
    fn real_interlacing_still_wins_over_a_progressive_container_flag() {
        // Interlaced video: the per-frame detector sees combing in most frames, so
        // there is no veto and the frames win, as they should.
        let c = with_single(
            counts((280, 0, 0, 20), (300, 0, 0, 0), 300),
            (200, 0, 10, 90),
        );
        let (mode, _, notes) = decide(&c, false, Some("progressive"));
        assert_eq!(mode, TemporalMode::Interlaced, "{notes:?}");
        assert!(mode.needs_deinterlace());
        assert!(notes.iter().any(|n| n.contains("trusting the frames")));
    }

    #[test]
    fn clean_progressive_is_classified_progressive() {
        let c = counts((0, 0, 298, 2), (300, 0, 0, 0), 300);
        let (mode, conf, notes) = decide(&c, false, Some("progressive"));
        assert_eq!(mode, TemporalMode::Progressive);
        assert!(conf >= 0.5);
        assert!(!mode.needs_ivtc());
        assert!(!mode.needs_deinterlace());
        assert!(notes.is_empty(), "unexpected notes: {notes:?}");
    }

    #[test]
    fn three_two_pulldown_is_classified_telecine_not_interlaced() {
        // 3:2 pulldown: ~40% clean progressive frames, the rest carrying mixed
        // fields, and duplicated fields throughout.
        let c = counts(
            (90, 90, 118, 2),
            (0, 100, 0, 0), // 100 of 300 frames repeat a field
            300,
        );
        let (mode, _, notes) = decide(&c, true, Some("tt"));
        assert_eq!(mode, TemporalMode::Telecine);
        assert!(mode.needs_ivtc());
        assert!(!mode.needs_deinterlace() || mode.needs_ivtc());
        assert!(notes.iter().any(|n| n.contains("inverse telecine")));
    }

    #[test]
    fn real_interlace_is_classified_interlaced() {
        let c = counts((150, 145, 5, 0), (300, 0, 0, 0), 300);
        let (mode, _, _) = decide(&c, true, Some("tt"));
        assert_eq!(mode, TemporalMode::Interlaced);
        assert!(mode.needs_deinterlace());
        assert!(!mode.needs_ivtc(), "interlaced video must not be decimated");
    }

    #[test]
    fn a_tie_lands_on_mixed_and_never_drops_frames() {
        // 160 clean progressive frames against 140 carrying mixed fields: no
        // side wins clearly, so the answer must be Mixed.
        let c = counts((70, 70, 160, 0), (300, 0, 0, 0), 300);
        let (mode, _, notes) = decide(&c, false, Some("progressive"));
        assert_eq!(mode, TemporalMode::Mixed);
        assert!(mode.needs_deinterlace());
        assert!(!mode.needs_ivtc());
        assert!(notes.iter().any(|n| n.contains("mixed cadence")));
    }

    #[test]
    fn container_flag_disagreement_is_reported_but_not_trusted() {
        let c = counts((0, 0, 300, 0), (300, 0, 0, 0), 300);
        let (mode, _, notes) = decide(&c, true, Some("tt"));
        assert_eq!(mode, TemporalMode::Progressive);
        assert!(notes.iter().any(|n| n.contains("trusting the frames")));
    }

    #[test]
    fn empty_statistics_are_honestly_unknown() {
        let (mode, conf, notes) = decide(&IdetCounts::default(), false, None);
        assert_eq!(mode, TemporalMode::Unknown);
        assert_eq!(conf, 0.0);
        assert!(!mode.is_confident());
        assert!(!notes.is_empty());
    }

    #[test]
    fn sample_points_cover_the_runtime_and_stay_in_bounds() {
        let points = sample_points(7200.0, &[0.01, 0.25, 0.5, 0.75, 0.99], 6.0);
        assert!(points.first().unwrap() >= &0.0);
        assert!(points.iter().all(|p| *p <= 7200.0 - 6.0 + 1e-9));
        assert!(points.windows(2).all(|w| w[0] < w[1]), "must be sorted+unique");
        assert!(points.len() >= 5);
    }

    #[test]
    fn short_clips_get_a_single_window() {
        assert_eq!(sample_points(3.0, &[0.1, 0.5, 0.9], 6.0), vec![0.0]);
        assert_eq!(sample_points(0.0, &[0.5], 6.0), vec![0.0]);
    }
}
