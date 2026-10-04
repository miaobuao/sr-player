//! The restoration pipeline: profile, policy, plan, segments and runner.
//!
//! The flow is deliberately linear and inspectable:
//!
//! ```text
//! probe → classify cadence → detect shots → analyse audio → build plan
//!       → remaster audio → execute video → quality control
//! ```
//!
//! "execute video" has two shapes, and the plan says which one it will be:
//!
//! * `FfmpegSinglePass` — one FFmpeg pass with a filter chain. No model runs, so
//!   nothing is restored or synthesised.
//! * `Chunked` — the engine decodes frames, encodes each chunk with its own
//!   per-shot filter chain, and checkpoints each one, so a job that dies half way
//!   through a feature resumes at the chunk that failed instead of starting again.
//!   This is what per-shot re-grain requires, and it is the shape the model stage
//!   will need as well.
//!
//! Every stage checkpoints its result, and every stage reports what it did to the
//! event bus, so the UI never has to guess.

pub mod grain;
pub mod native;
pub mod plan;
pub mod resize;
pub mod policy;
pub mod profile;
pub mod runner;
pub mod segments;

pub use native::{Frame, NativeExecutor, NativeOutcome};
pub use plan::{
    build_filter_chain, build_plan, cadence_for, choose_target, AudioPlan, AudioSummary,
    CadencePlan, ConversionPlan, PlanRequest, VideoExecutor, VideoPlan,
};
pub use policy::{plan_vram, VramBudget, WorkingSet};
pub use profile::{
    AnalysisSettings, AudioSettings, GpuPolicy, InterpolationMethod, InterpolationSettings,
    LoudnessTarget, OutputSettings, RestorationProfile, RestorationSettings, TILE_LADDER,
};
pub use runner::{ai_runtime_line, PipelineRunner, QcCheck, QcReport, RunnerOptions};
pub use segments::{plan_run, Chunk, RunPlan, Segment, SegmentKind, SegmentOptions};
