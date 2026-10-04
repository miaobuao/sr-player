//! Media analysis: manifest, temporal classification and shot detection.
//!
//! These modules answer the questions that must be answered *before* any pixel
//! is touched, because getting them wrong cannot be fixed later:
//!
//! * What is actually in this file? ([`manifest`], [`probe`])
//! * Is it progressive, telecined or genuinely interlaced? ([`classify`])
//! * Where are the cuts, so no frame is ever synthesised across one? ([`scene`])
//!
//! Audio analysis lives in [`crate::audio`].

pub mod classify;
pub mod manifest;
pub mod probe;
pub mod scene;

pub use classify::{ClassifyOptions, TemporalMode, TemporalReport};
pub use manifest::{AudioStream, MediaManifest, VideoStream};
pub use probe::probe;
pub use scene::{detect_scenes, SceneOptions, SceneReport, Shot};
