//! # sr-gui
//!
//! The presentation layer of sr-player: **pick a file → convert → progress →
//! logs**. There is no playback here — that is delegated to an external player.
//!
//! Every piece of media work happens in `sr-core`; this crate owns a window, a
//! single event subscription and a handful of worker threads.

mod app;
mod bridge;
mod state;
mod theme;
mod widgets;

use gpui::{App, AppContext as _, Application, Bounds, WindowBounds, WindowOptions, px, size};

fn main() {
    init_tracing();

    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1180.), px(820.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |window, cx| cx.new(|cx| app::AppView::new(window, cx)),
        )
        .expect("could not open the sr-player window");
        cx.activate(true);
    });
}

/// `sr-core` mirrors every engine log record into `tracing`; this makes those
/// visible on stderr when `RUST_LOG` asks for them.
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("sr_core=info,sr_gui=info"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();

    tracing::info!("sr-gui {} starting", sr_core::VERSION);
}
