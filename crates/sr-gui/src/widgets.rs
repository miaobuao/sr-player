//! Reusable presentation: the buttons, chips, badges and panels the app is
//! assembled from. Everything returns `AnyElement` so callers can drop them
//! into `Vec`s and mix shapes freely.

use gpui::{
    AnyElement, App, ClickEvent, Context, SharedString, Window, div, prelude::*, px, relative, rgb,
};
use sr_core::StageStatus;

use crate::state;
use crate::theme;

/// Wraps a view method into the `Fn(&ClickEvent, &mut Window, &mut App)`
/// shape GPUI's click handlers want. Written out explicitly (rather than via
/// `Context::listener`) so the event type never has to be inferred.
pub fn on_click<T: 'static>(
    cx: &Context<T>,
    handler: impl Fn(&mut T, &mut Context<T>) + 'static,
) -> impl Fn(&ClickEvent, &mut Window, &mut App) + 'static {
    let view = cx.entity().downgrade();
    move |_event, _window, cx| {
        view.update(cx, |view, cx| handler(view, cx)).ok();
    }
}

/// A small pill for versions and one-line summaries.
pub fn tag(text: impl Into<SharedString>) -> AnyElement {
    div()
        .flex_none()
        .px_2()
        .py_1()
        .rounded_sm()
        .bg(rgb(theme::PANEL_ALT))
        .border_1()
        .border_color(rgb(theme::BORDER))
        .text_xs()
        .text_color(rgb(theme::TEXT_DIM))
        .child(text.into())
        .into_any_element()
}

/// A push button. Disabled buttons are dimmed and inert (no click handler).
pub fn button(
    id: &'static str,
    label: impl Into<SharedString>,
    enabled: bool,
    primary: bool,
    handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> AnyElement {
    let (background, border, foreground, hover) = if primary {
        (
            theme::ACCENT,
            theme::ACCENT,
            theme::TEXT,
            theme::ACCENT_HOVER,
        )
    } else {
        (
            theme::PANEL_ALT,
            theme::BORDER,
            theme::TEXT,
            theme::PANEL_HOVER,
        )
    };

    let mut element = div()
        .id(id)
        .flex_none()
        .px_3()
        .py_1()
        .rounded_sm()
        .border_1()
        .text_sm()
        .bg(rgb(background))
        .border_color(rgb(border))
        .text_color(rgb(if enabled { foreground } else { theme::TEXT_FAINT }))
        .child(label.into());

    if enabled {
        element = element
            .cursor_pointer()
            .hover(move |style| style.bg(rgb(hover)).border_color(rgb(hover)))
            .on_click(handler);
    } else {
        element = element.opacity(0.45);
    }

    element.into_any_element()
}

/// A selectable pill, used for the restoration profiles and the log filter.
pub fn chip(
    id: (&'static str, usize),
    label: impl Into<SharedString>,
    selected: bool,
    enabled: bool,
    handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> AnyElement {
    let mut element = div()
        .id(id)
        .flex_none()
        .px_3()
        .py_1()
        .rounded_full()
        .border_1()
        .text_xs()
        .child(label.into());

    if selected {
        element = element
            .bg(rgb(theme::ACCENT))
            .border_color(rgb(theme::ACCENT))
            .text_color(rgb(theme::TEXT));
    } else {
        element = element
            .bg(rgb(theme::PANEL_ALT))
            .border_color(rgb(theme::BORDER))
            .text_color(rgb(theme::TEXT_DIM));
    }

    if enabled {
        element = element.cursor_pointer();
        element = if selected {
            element
        } else {
            element.hover(|style| {
                style
                    .border_color(rgb(theme::ACCENT))
                    .text_color(rgb(theme::TEXT))
            })
        };
        element = element.on_click(handler);
    } else {
        element = element.opacity(0.5);
    }

    element.into_any_element()
}

/// A checkbox-style toggle. The tick is drawn, not typed, so it never depends
/// on a font having ☑.
pub fn toggle(
    id: &'static str,
    label: impl Into<SharedString>,
    checked: bool,
    enabled: bool,
    handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> AnyElement {
    let box_ = div()
        .flex_none()
        .w(px(14.))
        .h(px(14.))
        .rounded_sm()
        .border_1()
        .border_color(rgb(if checked {
            theme::ACCENT
        } else {
            theme::BORDER
        }))
        .when(checked, |this| this.bg(rgb(theme::ACCENT)));

    let mut element = div()
        .id(id)
        .flex_none()
        .flex()
        .items_center()
        .gap_2()
        .px_2()
        .py_1()
        .rounded_sm()
        .border_1()
        .border_color(rgb(theme::BORDER))
        .bg(rgb(theme::PANEL_ALT))
        .text_xs()
        .text_color(rgb(if enabled {
            theme::TEXT_DIM
        } else {
            theme::TEXT_FAINT
        }))
        .child(box_)
        .child(label.into());

    if enabled {
        element = element
            .cursor_pointer()
            .hover(|style| style.border_color(rgb(theme::ACCENT)))
            .on_click(handler);
    } else {
        element = element.opacity(0.5);
    }

    element.into_any_element()
}

/// The coloured status pill of one ladder rung.
pub fn badge(status: StageStatus) -> AnyElement {
    let (foreground, background) = match status {
        StageStatus::Pending => (theme::TEXT_FAINT, theme::PANEL_ALT),
        StageStatus::Running => (theme::TEXT, theme::ACCENT_SOFT),
        StageStatus::Done => (theme::OK, theme::OK_SOFT),
        StageStatus::Skipped => (theme::MUTED, theme::PANEL_ALT),
        StageStatus::Degraded => (theme::WARN, theme::PANEL_ALT),
        StageStatus::Failed => (theme::ERR, theme::ERR_SOFT),
    };

    div()
        .flex_none()
        .px_2()
        .rounded_sm()
        .bg(rgb(background))
        .text_xs()
        .text_color(rgb(foreground))
        .child(state::status_label(status))
        .into_any_element()
}

/// A filled track; `fraction` is clamped, so an over-eager stage cannot
/// overflow the bar.
pub fn progress_bar(fraction: f32, complete: bool) -> AnyElement {
    let fill = if complete { theme::OK } else { theme::ACCENT };
    div()
        .w_full()
        .h(px(10.))
        .rounded_full()
        .bg(rgb(theme::PANEL_ALT))
        .border_1()
        .border_color(rgb(theme::BORDER))
        .overflow_hidden()
        .child(
            div()
                .h_full()
                .w(relative(fraction.clamp(0.0, 1.0)))
                .bg(rgb(fill)),
        )
        .into_any_element()
}

/// Key/value rows for the info panels.
pub fn kv_rows(rows: &[(String, String)]) -> Vec<AnyElement> {
    rows.iter()
        .map(|(key, value)| {
            div()
                .flex()
                .items_start()
                .gap_2()
                .py_1()
                .child(
                    div()
                        .w(px(104.))
                        .flex_shrink_0()
                        .text_xs()
                        .text_color(rgb(theme::TEXT_FAINT))
                        .child(key.clone()),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w(px(0.))
                        .text_xs()
                        .text_color(rgb(theme::TEXT))
                        .child(value.clone()),
                )
                .into_any_element()
        })
        .collect()
}

/// An amber line, used for plan warnings.
pub fn warning(text: impl Into<SharedString>) -> AnyElement {
    div()
        .flex_none()
        .py_1()
        .text_xs()
        .text_color(rgb(theme::WARN))
        .child(text.into())
        .into_any_element()
}

/// Grey italic-ish filler for panels that have nothing to show yet.
pub fn hint(text: impl Into<SharedString>) -> AnyElement {
    div()
        .py_1()
        .text_xs()
        .text_color(rgb(theme::TEXT_FAINT))
        .child(text.into())
        .into_any_element()
}

/// A titled panel.
pub fn card(title: impl Into<SharedString>, body: Vec<AnyElement>) -> AnyElement {
    div()
        .flex()
        .flex_col()
        .flex_shrink_0()
        .rounded_md()
        .border_1()
        .border_color(rgb(theme::BORDER))
        .bg(rgb(theme::PANEL))
        .overflow_hidden()
        .child(
            div()
                .flex_none()
                .px_3()
                .py_2()
                .border_b_1()
                .border_color(rgb(theme::BORDER))
                .text_xs()
                .text_color(rgb(theme::TEXT_DIM))
                .child(title.into()),
        )
        .child(div().flex().flex_col().p_3().children(body))
        .into_any_element()
}

/// A labelled path with its action button, as used by 输入文件 / 输出文件.
pub fn path_field(
    label: &'static str,
    value: String,
    has_value: bool,
    action: AnyElement,
) -> AnyElement {
    div()
        .flex()
        .flex_col()
        .flex_1()
        .min_w(px(0.))
        .gap_1()
        .child(
            div()
                .text_xs()
                .text_color(rgb(theme::TEXT_FAINT))
                .child(label),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(
                    div()
                        .flex_1()
                        .min_w(px(0.))
                        .px_2()
                        .py_1()
                        .rounded_sm()
                        .bg(rgb(theme::PANEL_ALT))
                        .border_1()
                        .border_color(rgb(theme::BORDER))
                        .text_xs()
                        .text_color(rgb(if has_value {
                            theme::TEXT
                        } else {
                            theme::TEXT_FAINT
                        }))
                        .whitespace_nowrap()
                        .overflow_hidden()
                        .text_ellipsis()
                        .child(value),
                )
                .child(action),
        )
        .into_any_element()
}
