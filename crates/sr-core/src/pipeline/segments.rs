//! Turning a shot list into the exact sequence of model calls and output frames.
//!
//! This module is pure logic — no FFmpeg, no files, no clock — because it is
//! where the two promises the product makes are kept or broken:
//!
//! 1. **Nothing is synthesised across a cut.** The interpolator only ever
//!    receives frames that the plan says belong together, so it cannot bridge a
//!    boundary even if it wanted to. That is stronger than asking a model (or an
//!    FFmpeg filter) to detect cuts itself: the illegal frame pair never exists.
//! 2. **The output frame count is exact.** `m` times interpolation of `n` input
//!    frames produces exactly `m*(n-1)+1` frames, whether the work is split into
//!    one segment or a thousand, and whether or not a failure forced a resume.
//!
//! ## How a segment maps to frames
//!
//! A segment covers input frames `[first..=last]` and emits output slots
//! `m*first ..= m*last`. Consecutive segments overlap by exactly one input frame,
//! and every segment after the first drops its first output slot, which the
//! previous segment already emitted. The counts therefore telescope:
//!
//! ```text
//! emitted = Σ (m*(last-first)+1) − (segments − 1) = m*(n-1) + 1
//! ```
//!
//! Three kinds of segment exist:
//!
//! * [`SegmentKind::Synthesise`] — every pair inside is legal; the model runs.
//! * [`SegmentKind::Hold`] — a dissolve guard or a shot boundary: the
//!   intermediate frames are copies of the frame before them. The model is not
//!   called at all, because a morph inside a dissolve is exactly the artefact
//!   this is here to avoid.
//! * [`SegmentKind::Pass`] — a single frame; only used for a one-frame input.

use crate::media::scene::{Shot, ShotKind};
use serde::{Deserialize, Serialize};

/// Options that shape the split.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentOptions {
    /// Frames of input the model may see in one call (its temporal window).
    pub max_input_frames: u32,
    /// Frames of input per chunk, before a new checkpoint is written. Chunks are
    /// cut at shot boundaries regardless of this number.
    pub target_chunk_frames: u64,
    /// How many frames either side of a gradual transition stay uninterpolated.
    pub dissolve_guard_frames: u64,
    /// True when the model synthesises frames at all (`m > 1` and an
    /// interpolation engine is attached).
    pub interpolate: bool,
}

impl Default for SegmentOptions {
    fn default() -> Self {
        SegmentOptions {
            max_input_frames: 2,
            target_chunk_frames: 1000,
            dissolve_guard_frames: 12,
            interpolate: true,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentKind {
    /// The model synthesises the frames between the endpoints.
    Synthesise,
    /// No synthesis: intermediate frames repeat the frame before them.
    Hold,
    /// A single frame, emitted as-is.
    Pass,
}

/// One unit of work: what to read, what to call, what to emit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub kind: SegmentKind,
    pub first: u64,
    /// Inclusive.
    pub last: u64,
    /// Index into the shot list this segment belongs to.
    pub shot: usize,
    /// Output slots this segment emits on its own: `m*(last-first)+1`.
    pub outputs: u64,
    /// The previous segment already emitted the slot at `m*first`.
    pub drop_first: bool,
}

impl Segment {
    pub fn input_frames(&self) -> u64 {
        self.last - self.first + 1
    }

    /// Frames this segment contributes to the final output.
    pub fn emitted(&self) -> u64 {
        if self.drop_first {
            self.outputs.saturating_sub(1)
        } else {
            self.outputs
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "{:?} [{}, {}] {} in → {} out",
            self.kind,
            self.first,
            self.last,
            self.input_frames(),
            self.emitted()
        )
    }
}

/// A checkpointable unit: one encode, one row in the `chunks` table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub index: u32,
    pub shot: usize,
    pub first_segment: usize,
    pub last_segment: usize,
    pub first_input: u64,
    pub last_input: u64,
    /// Frames this chunk writes, after dropping the overlapping slots.
    pub outputs: u64,
    /// True when the chunk starts with a slot the previous chunk already wrote.
    pub drop_first: bool,
    /// Shot boundaries and dissolve guards inside this chunk: the log shows them
    /// so a human can see the protection working.
    pub protected_boundaries: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunPlan {
    pub segments: Vec<Segment>,
    pub chunks: Vec<Chunk>,
    /// `m*(n-1)+1` for `n` input frames.
    pub total_output_frames: u64,
    pub input_frames: u64,
    pub multiplier: u32,
    pub notes: Vec<String>,
}

impl RunPlan {
    /// The number of frames the plan says it will emit. Checked against the
    /// segment arithmetic so a bug cannot silently produce a short file.
    pub fn emitted_frames(&self) -> u64 {
        let all: u64 = self.segments.iter().map(Segment::emitted).sum();
        let by_chunk: u64 = self.chunks.iter().map(|c| c.outputs).sum();
        debug_assert_eq!(all, by_chunk, "segment and chunk accounting disagree");
        all
    }
}

/// Which shot a frame belongs to.
///
/// `partition_point` rather than `binary_search`: a frame outside every shot (an
/// off-by-one at the tail, or a malformed list from a resumed analysis) is
/// attributed to the last shot that starts at or before it, which is monotone and
/// cannot panic in the middle of a long job.
fn shot_of(shots: &[Shot], frame: u64) -> usize {
    if shots.is_empty() {
        return 0;
    }
    let index = shots.partition_point(|s| s.start_frame <= frame);
    index.saturating_sub(1).min(shots.len() - 1)
}

/// Builds the segment and chunk plan.
pub fn plan_run(
    input_frames: u64,
    multiplier: u32,
    shots: &[Shot],
    options: &SegmentOptions,
) -> RunPlan {
    let m = multiplier.max(1) as u64;
    let n = input_frames;
    let mut notes = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();

    if n == 0 {
        return RunPlan {
            segments,
            chunks: Vec::new(),
            total_output_frames: 0,
            input_frames: 0,
            multiplier: m as u32,
            notes: vec!["the input has no frames to process".into()],
        };
    }

    // A pair may be synthesised only when both frames belong to the same shot and
    // the pair is clear of a gradual transition.
    let boundaries: Vec<u64> = shots
        .iter()
        .filter(|s| s.kind == ShotKind::Dissolve)
        .map(|s| s.start_frame)
        .collect();
    let guard = options.dissolve_guard_frames;
    let guarded = |pair: u64| -> bool {
        boundaries.iter().any(|d| {
            let low = d.saturating_sub(guard);
            let high = d.saturating_add(guard);
            pair >= low && pair < high
        })
    };
    let allowed = |pair: u64| -> bool {
        if !options.interpolate || m <= 1 || pair + 1 >= n {
            return false;
        }
        if shot_of(shots, pair) != shot_of(shots, pair + 1) {
            return false;
        }
        !guarded(pair)
    };

    if n == 1 {
        segments.push(Segment {
            kind: SegmentKind::Pass,
            first: 0,
            last: 0,
            shot: shot_of(shots, 0),
            outputs: 1,
            drop_first: false,
        });
        return finish(segments, n, m, options, notes);
    }

    let window = options.max_input_frames.max(2) as u64;
    let mut start: u64 = 0;
    while start < n - 1 {
        let shot = shot_of(shots, start);
        if !allowed(start) {
            // Hold: cover every consecutive pair that may not be synthesised.
            let mut end = start + 1;
            while end + 1 < n
                && !allowed(end)
                && shot_of(shots, end + 1) == shot
            {
                end += 1;
            }
            segments.push(Segment {
                kind: SegmentKind::Hold,
                first: start,
                last: end,
                shot,
                outputs: m * (end - start) + 1,
                drop_first: !segments.is_empty(),
            });
            start = end;
        } else {
            let mut end = start + 1;
            while end + 1 < n
                && allowed(end)
                && shot_of(shots, end + 1) == shot
                && (end + 1 - start + 1) < window
            {
                end += 1;
            }
            segments.push(Segment {
                kind: SegmentKind::Synthesise,
                first: start,
                last: end,
                shot,
                outputs: m * (end - start) + 1,
                drop_first: !segments.is_empty(),
            });
            start = end;
        }
    }

    let protected = segments
        .iter()
        .filter(|s| s.kind != SegmentKind::Synthesise)
        .count();
    if protected > 0 {
        notes.push(format!(
            "{protected} region(s) are never handed to the model: shot boundaries and the \
             {guard}-frame guard either side of a gradual transition"
        ));
    }
    let synthesised: u64 = segments
        .iter()
        .filter(|s| s.kind == SegmentKind::Synthesise)
        .map(|s| s.emitted())
        .sum();
    if synthesised > 0 {
        notes.push(format!(
            "{synthesised} frame(s) will be synthesised by the model, in {} call(s) of at most \
             {window} input frames",
            segments
                .iter()
                .filter(|s| s.kind == SegmentKind::Synthesise)
                .count()
        ));
    }

    finish(segments, n, m, options, notes)
}

fn finish(
    segments: Vec<Segment>,
    input_frames: u64,
    m: u64,
    options: &SegmentOptions,
    mut notes: Vec<String>,
) -> RunPlan {
    let total_output_frames = if input_frames == 0 {
        0
    } else {
        m * (input_frames - 1) + 1
    };
    let chunks = group_chunks(&segments, options, m);
    let plan = RunPlan {
        segments,
        chunks,
        total_output_frames,
        input_frames,
        multiplier: m as u32,
        notes: Vec::new(),
    };
    let emitted = plan.segments.iter().map(Segment::emitted).sum::<u64>();
    if emitted != total_output_frames {
        notes.push(format!(
            "internal accounting: {emitted} frames planned, {total_output_frames} expected"
        ));
    }
    RunPlan { notes, ..plan }
}

/// Groups segments into checkpointable chunks.
///
/// A chunk never spans a shot boundary, so a failure costs one shot's work at
/// most, and it stops short of `target_chunk_frames` input frames so the
/// intermediate file size stays bounded.
fn group_chunks(segments: &[Segment], options: &SegmentOptions, m: u64) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut index = 0usize;
    while index < segments.len() {
        let shot = segments[index].shot;
        let first = index;
        let mut last = index;
        let mut protected = 0u32;
        if segments[index].kind != SegmentKind::Synthesise {
            protected += 1;
        }
        while last + 1 < segments.len() {
            let next = &segments[last + 1];
            if next.shot != shot {
                break;
            }
            let span = next.last - segments[first].first;
            if span > options.target_chunk_frames {
                break;
            }
            last += 1;
            if next.kind != SegmentKind::Synthesise {
                protected += 1;
            }
        }
        let outputs: u64 = segments[first..=last]
            .iter()
            .map(Segment::emitted)
            .sum();
        chunks.push(Chunk {
            index: chunks.len() as u32,
            shot,
            first_segment: first,
            last_segment: last,
            first_input: segments[first].first,
            last_input: segments[last].last,
            outputs,
            drop_first: segments[first].drop_first,
            protected_boundaries: protected,
        });
        index = last + 1;
    }
    let _ = m;
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::scene::ShotKind;
    use crate::time::{Rational, Timestamp};

    fn shots(boundaries: &[u64], total: u64) -> Vec<Shot> {
        let tb = Rational::new(1, 24).expect("timebase");
        let mut shots = Vec::new();
        let mut start = 0u64;
        let mut index = 0usize;
        // Degenerate boundaries (before the start, or past the end) are dropped
        // rather than turned into empty shots: a real detector never emits them,
        // and the planner must not depend on that.
        let mut clean: Vec<u64> = boundaries
            .iter()
            .copied()
            .filter(|b| *b > 0 && *b < total)
            .collect();
        clean.sort_unstable();
        clean.dedup();
        for boundary in clean.into_iter().chain(std::iter::once(total)) {
            shots.push(Shot {
                index,
                start_frame: start,
                end_frame: boundary,
                start: Timestamp::new(start as i64, tb),
                end: Timestamp::new(boundary as i64, tb),
                kind: if index == 0 {
                    ShotKind::Start
                } else {
                    ShotKind::Cut
                },
                cut_score: None,
                duration_seconds: (boundary - start) as f64 / 24.0,
            });
            start = boundary;
            index += 1;
        }
        shots
    }

    fn options(window: u32) -> SegmentOptions {
        SegmentOptions {
            max_input_frames: window,
            target_chunk_frames: 1000,
            dissolve_guard_frames: 0,
            interpolate: true,
        }
    }

    fn total_emitted(plan: &RunPlan) -> u64 {
        plan.segments.iter().map(Segment::emitted).sum()
    }

    #[test]
    fn a_single_shot_is_exactly_doubled() {
        let plan = plan_run(100, 2, &shots(&[], 100), &options(2));
        assert_eq!(plan.total_output_frames, 199);
        assert_eq!(total_emitted(&plan), 199);
        assert!(plan.segments.iter().all(|s| s.kind == SegmentKind::Synthesise));
    }

    #[test]
    fn no_synthesis_ever_spans_a_cut() {
        let list = shots(&[70, 140], 200);
        let plan = plan_run(200, 2, &list, &options(5));
        for segment in plan
            .segments
            .iter()
            .filter(|s| s.kind == SegmentKind::Synthesise)
        {
            assert_eq!(
                shot_of(&list, segment.first),
                shot_of(&list, segment.last),
                "synthesis segment {} spans a cut",
                segment.describe()
            );
            // The cut pairs themselves must never be inside a synthesis run.
            for pair in segment.first..segment.last {
                assert!(
                    shot_of(&list, pair) == shot_of(&list, pair + 1),
                    "synthesis covered the cut pair ({pair}, {})",
                    pair + 1
                );
            }
        }
        // A held region may straddle the boundary, because it synthesises
        // nothing: that is the whole point of holding it.
        let holds: Vec<&Segment> = plan
            .segments
            .iter()
            .filter(|s| s.kind == SegmentKind::Hold)
            .collect();
        assert_eq!(holds.len(), 2, "one held region per cut, got {holds:?}");
        assert_eq!(total_emitted(&plan), 399);
    }

    #[test]
    fn the_frame_count_is_exact_for_awkward_shapes() {
        for n in [1u64, 2, 3, 7, 8, 9, 23, 24, 25, 240, 999] {
            for boundaries in [vec![], vec![n / 3 + 1], vec![2, n / 2, n - 1]] {
                let list = shots(&boundaries, n);
                for window in [2u32, 3, 5] {
                    let plan = plan_run(n, 2, &list, &options(window));
                    assert_eq!(
                        total_emitted(&plan),
                        plan.total_output_frames,
                        "n={n} boundaries={boundaries:?} window={window}"
                    );
                    assert_eq!(plan.total_output_frames, 2 * (n - 1) + 1);
                    let by_chunk: u64 = plan.chunks.iter().map(|c| c.outputs).sum();
                    assert_eq!(by_chunk, plan.total_output_frames);
                }
            }
        }
    }

    #[test]
    fn a_three_x_multiplier_is_also_exact() {
        let plan = plan_run(120, 3, &shots(&[40, 80], 120), &options(4));
        assert_eq!(plan.total_output_frames, 3 * 119 + 1);
        assert_eq!(total_emitted(&plan), plan.total_output_frames);
    }

    #[test]
    fn without_interpolation_every_frame_passes_through_untouched() {
        let mut opts = options(2);
        opts.interpolate = false;
        let plan = plan_run(50, 2, &shots(&[20], 50), &opts);
        // The rate still doubles, but every segment is a Hold: no model call.
        assert!(plan.segments.iter().all(|s| s.kind == SegmentKind::Hold));
        assert_eq!(total_emitted(&plan), 99);
    }

    #[test]
    fn a_dissolve_guard_is_held_while_the_cut_is_still_synthesised_up_to_it() {
        let mut list = shots(&[50], 120);
        list[1].kind = ShotKind::Dissolve;
        let mut opts = options(2);
        opts.dissolve_guard_frames = 5;
        let plan = plan_run(120, 2, &list, &opts);
        let holds: Vec<&Segment> = plan
            .segments
            .iter()
            .filter(|s| s.kind == SegmentKind::Hold)
            .collect();
        assert!(!holds.is_empty(), "a dissolve must produce a held region");
        // Pairs 44..54 (5 either side of frame 50) may not be synthesised.
        for segment in plan.segments.iter().filter(|s| s.kind == SegmentKind::Synthesise) {
            for pair in segment.first..segment.last {
                assert!(
                    pair < 45 || pair >= 55,
                    "pair ({pair}, {}) is inside the dissolve guard",
                    pair + 1
                );
            }
        }
        assert_eq!(total_emitted(&plan), 239);
    }

    #[test]
    fn chunks_never_mix_two_shots_of_synthesis_work() {
        let list = shots(&[300, 900, 1500], 2000);
        let mut opts = options(2);
        opts.target_chunk_frames = 250;
        let plan = plan_run(2000, 2, &list, &opts);
        assert!(plan.chunks.len() > 4, "the plan should be split for resume");
        for chunk in &plan.chunks {
            // A chunk's synthesis work belongs to exactly one shot. A held region
            // may straddle a boundary, because it synthesises nothing.
            let synthesised_shots: Vec<usize> = plan.segments
                [chunk.first_segment..=chunk.last_segment]
                .iter()
                .filter(|s| s.kind == SegmentKind::Synthesise)
                .map(|s| s.shot)
                .collect();
            if let Some(first) = synthesised_shots.first() {
                assert!(
                    synthesised_shots.iter().all(|s| s == first),
                    "chunk {} synthesises across shots {:?}",
                    chunk.index,
                    synthesised_shots
                );
                assert_eq!(
                    *first,
                    shot_of(&list, chunk.first_input),
                    "chunk {} is labelled with the wrong shot",
                    chunk.index
                );
            }
            assert!(
                chunk.last_input - chunk.first_input < 400,
                "chunk {} covers {} input frames, more than intended",
                chunk.index,
                chunk.last_input - chunk.first_input
            );
        }
        let by_chunk: u64 = plan.chunks.iter().map(|c| c.outputs).sum();
        assert_eq!(by_chunk, plan.total_output_frames);
    }

    #[test]
    fn a_one_frame_input_produces_one_frame() {
        let plan = plan_run(1, 2, &shots(&[], 1), &options(2));
        assert_eq!(plan.total_output_frames, 1);
        assert_eq!(total_emitted(&plan), 1);
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].kind, SegmentKind::Pass);
    }

    #[test]
    fn an_empty_input_produces_nothing_and_says_so() {
        let plan = plan_run(0, 2, &[], &options(2));
        assert_eq!(plan.total_output_frames, 0);
        assert!(plan.segments.is_empty());
        assert!(!plan.notes.is_empty());
    }

    #[test]
    fn a_temporal_window_never_produces_an_oversized_call() {
        for window in [2u32, 3, 5] {
            let plan = plan_run(500, 2, &shots(&[200], 500), &options(window));
            for segment in plan.segments.iter().filter(|s| s.kind == SegmentKind::Synthesise) {
                assert!(
                    segment.input_frames() <= window as u64,
                    "segment {} exceeds the {window}-frame window",
                    segment.describe()
                );
            }
        }
    }
}
