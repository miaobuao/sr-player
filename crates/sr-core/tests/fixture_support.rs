//! The fixture builder's own tests.
//!
//! `tests/support/mod.rs` is a module, not a test target, so it needs a file to
//! include it. The point of these tests is that a fixture can be *trusted*
//! without asking FFmpeg what it contains: the synthesis and the measurements
//! share one buffer.

mod support;
