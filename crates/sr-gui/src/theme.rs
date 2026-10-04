//! The whole visual vocabulary of the app: a small dark palette plus the one
//! function that maps a log level onto a colour. Nothing here knows about
//! `sr-core`; keep it that way so the look can be changed in one file.

/// Window background.
pub const BG: u32 = 0x17171a;
/// Header, status bar and cards.
pub const PANEL: u32 = 0x1e1e23;
/// Controls sitting on top of a panel.
pub const PANEL_ALT: u32 = 0x26262c;
/// Hover state for those controls.
pub const PANEL_HOVER: u32 = 0x30303a;

/// Hairline separators.
pub const BORDER: u32 = 0x2c2c33;

/// Primary text.
pub const TEXT: u32 = 0xe8e8ea;
/// Secondary text.
pub const TEXT_DIM: u32 = 0x9a9aa5;
/// Placeholders, disabled controls, trace output.
pub const TEXT_FAINT: u32 = 0x6e6e7a;

/// Interactive accent: the primary button, the progress fill, the active chip.
pub const ACCENT: u32 = 0x3f7fe0;
/// Accent hover.
pub const ACCENT_HOVER: u32 = 0x5590ec;
/// Tinted background for the stage that is currently running.
pub const ACCENT_SOFT: u32 = 0x1d2a42;

/// Success.
pub const OK: u32 = 0x46b877;
/// Success banner background.
pub const OK_SOFT: u32 = 0x17281e;

/// Warnings and degraded stages.
pub const WARN: u32 = 0xd9a03a;

/// Failures.
pub const ERR: u32 = 0xe05a5a;
/// Failure banner background.
pub const ERR_SOFT: u32 = 0x2b191b;

/// Skipped / inert stages.
pub const MUTED: u32 = 0x8b8b98;

/// Colour used to print one log line.
pub fn level_color(level: sr_core::Level) -> u32 {
    match level {
        sr_core::Level::Trace | sr_core::Level::Debug => TEXT_FAINT,
        sr_core::Level::Info => TEXT_DIM,
        sr_core::Level::Warn => WARN,
        sr_core::Level::Error => ERR,
    }
}
