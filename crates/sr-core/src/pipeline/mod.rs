//! The restoration pipeline: profile, policy, plan and runner.
//!
//! The flow is deliberately linear and inspectable:
//!
//! ```text
//! probe → classify cadence → detect shots → analyse audio → build plan
//!       → remaster audio → encode + mux (one FFmpeg pass) → quality control
//! ```
//!
//! Every stage checkpoints its result, so a job that dies half way through a
//! feature resumes instead of starting again, and every stage reports what it
//! did to the event bus, so the UI never has to guess.

pub mod plan;
pub mod policy;
pub mod profile;
pub mod runner;

pub use plan::{
    build_filter_chain, build_plan, choose_target, AudioPlan, AudioSummary, ConversionPlan,
    PlanRequest, VideoPlan,
};
pub use policy::{plan_vram, OffloadMode, VramBudget, WorkingSet};
pub use profile::{
    AnalysisSettings, AudioSettings, GpuPolicy, InterpolationMethod, InterpolationSettings,
    LoudnessTarget, OutputSettings, RestorationProfile, RestorationSettings,
    VALID_TEMPORAL_BATCHES,
};
pub use runner::{PipelineRunner, QcCheck, QcReport, RunnerOptions};
