//! Durable job state.
//!
//! SQLite in WAL mode, one file, no server. A single workstation with one GPU
//! does not need a queue broker; it needs a database that survives a power cut
//! and tells the operator exactly which chunk to resume from.
//!
//! The commit rule everywhere in this crate is: **write the artifact, fsync it,
//! rename it into place, then record it.** A row never claims success before the
//! bytes are on disk.

pub mod store;

pub use store::{
    commit_file_atomic, ChunkRow, JobRow, NewJob, ResumePoint, StageRow, Store,
};
