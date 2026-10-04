//! # sr-core
//!
//! The media engine behind `sr-gui` and `sr-cli`.
//!
//! Design rules this crate is built around:
//!
//! 1. **The UI never touches media.** `sr-core` owns probing, the timeline,
//!    decoding, analysis, DSP, the job state machine and every FFmpeg/child
//!    process. Front ends subscribe to [`Event`]s and read [`state::Store`].
//! 2. **No Python, no CUDA, no vendor SDK is *required*.** FFmpeg is the only
//!    hard external dependency. GPU inference is an optional, pluggable
//!    accelerator ([`infer`]) — correctness never depends on it.
//! 3. **Rational time everywhere.** No `frame_index / fps` float games; see
//!    [`time`].
//! 4. **Failure degrades, it does not exit.** Out-of-memory and "unavailable
//!    backend" conditions walk the [`pipeline::policy`] ladder instead of
//!    aborting the job.
//! 5. **Every stage is resumable.** Progress is committed to SQLite with
//!    atomic file renames; see [`state`].
//!
//! Layout:
//!
//! | module | responsibility |
//! |---|---|
//! | [`time`] | exact rational timestamps / frame rates |
//! | [`ffmpeg`] | tool discovery, arg building, process supervision, `-progress` parsing |
//! | [`media`] | probe/manifest, temporal classification, scene cuts, loudness, dialogue |
//! | [`audio`] | Rust-native DSP: BS.1770 measurement, dialogue rider, band ducking, WAV |
//! | [`pipeline`] | plan, VRAM policy, stage runner, events, checkpointing |
//! | [`infer`] | vendor-neutral inference ABI (FFmpeg baseline + optional plugin) |
//! | [`state`] | SQLite job store |

pub mod audio;
pub mod error;
pub mod events;
pub mod ffmpeg;
pub mod gpu;
pub mod infer;
pub mod media;
pub mod pipeline;
pub mod state;
pub mod time;

pub use error::{Error, Result};
pub use events::{
    Event, EventBus, JobOutcome, JobState, Level, LogRecord, Reporter, Stage, StageProgress,
    StageStatus,
};
pub use time::{Rational, Timestamp};

/// Version of the crate, surfaced in the UI header and job manifests.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
