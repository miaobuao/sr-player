//! The window itself: state, wiring and rendering.
//!
//! Threading rules this file obeys:
//! * the UI thread never blocks — dialogs are awaited inside `cx.spawn`,
//!   probing runs on the background executor, and the conversion gets its own
//!   `std::thread` inside [`bridge::Engine::spawn_job`];
//! * engine output arrives through one subscription drained by one task
//!   (`spawn_event_pump`), never by polling the runner.

use std::path::PathBuf;
use std::time::Duration;

use gpui::{
    AnyElement, Context, FontWeight, PathPromptOptions, Render, ScrollHandle, Window, div,
    prelude::*, px, rgb,
};
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::RestorationProfile;
use sr_core::pipeline::runner::RunnerOptions;
use sr_core::{Event, Level, StageStatus};

use crate::bridge::{self, Engine, StartupError};
use crate::state::{self, JobView, LevelFilter};
use crate::{theme, widgets};

/// How often the event pump wakes up when the engine is quiet.
const PUMP_INTERVAL: Duration = Duration::from_millis(80);

pub struct AppView {
    /// `None` only when startup failed, in which case the whole UI is disabled.
    engine: Option<Engine>,
    startup_error: Option<StartupError>,

    input: Option<PathBuf>,
    output: Option<PathBuf>,
    profiles: Vec<RestorationProfile>,
    profile_index: usize,
    dry_run: bool,

    manifest_rows: Vec<(String, String)>,
    probing: bool,

    job: JobView,
    log_scroll: ScrollHandle,
    info_scroll: ScrollHandle,
}

impl AppView {
    /// Builds the view: bootstrap the engine, seed the log, wire the pump.
    pub fn new(_window: &mut Window, cx: &mut Context<Self>) -> Self {
        let (engine, startup_error) = match Engine::bootstrap() {
            Ok(engine) => (Some(engine), None),
            Err(err) => (None, Some(err)),
        };

        let mut view = AppView {
            engine,
            startup_error,
            input: None,
            output: None,
            // `builtin()` starts with `safe-16gb`, which is the intended default.
            profiles: RestorationProfile::builtin(),
            profile_index: 0,
            dry_run: false,
            manifest_rows: Vec::new(),
            probing: false,
            job: JobView::new(),
            log_scroll: ScrollHandle::new(),
            info_scroll: ScrollHandle::new(),
        };

        match view.engine.clone() {
            Some(engine) => {
                view.job.note(
                    Level::Info,
                    None,
                    format!("sr-core {} 已加载", sr_core::VERSION),
                );
                view.job
                    .note(Level::Info, None, format!("FFmpeg {}", engine.ffmpeg_version()));
                view.job
                    .note(Level::Info, None, engine.engines().summary());
            }
            None => {
                if let Some(err) = &view.startup_error {
                    view.job.note(
                        Level::Error,
                        None,
                        format!("{}：{}", err.title(), err.detail()),
                    );
                }
            }
        }

        view.spawn_event_pump(cx);
        view
    }

    // ---- event pump -----------------------------------------------------

    /// Subscribes once and drains the bus from a single task. This is the only
    /// place the UI learns anything about a running job.
    fn spawn_event_pump(&mut self, cx: &mut Context<Self>) {
        let Some(engine) = self.engine.clone() else {
            return;
        };
        let rx = engine.bus().subscribe();

        cx.spawn(async move |this, cx| {
            loop {
                while let Ok(event) = rx.try_recv() {
                    let applied = this.update(cx, |view, cx| {
                        if view.handle_event(event) {
                            cx.notify();
                        }
                    });
                    if applied.is_err() {
                        // The window went away; stop draining the bus.
                        return;
                    }
                }
                cx.background_executor().timer(PUMP_INTERVAL).await;
            }
        })
        .detach();
    }

    fn handle_event(&mut self, event: Event) -> bool {
        let is_log = matches!(event, Event::Log(_));
        let changed = self.job.apply(event);
        if is_log && changed {
            self.log_scroll.scroll_to_bottom();
        }
        changed
    }

    // ---- actions --------------------------------------------------------

    fn note_error(&mut self, message: impl Into<String>) {
        self.job.note(Level::Error, None, message);
        self.log_scroll.scroll_to_bottom();
    }

    fn pick_input(&mut self, cx: &mut Context<Self>) {
        let prompt = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("选择要修复的视频".into()),
        });

        cx.spawn(async move |this, cx| {
            let picked = match prompt.await {
                Ok(Ok(paths)) => paths.and_then(|mut paths| paths.pop()),
                Ok(Err(err)) => {
                    let message = format!("打开文件对话框失败：{err}");
                    this.update(cx, |view, cx| {
                        view.note_error(message);
                        cx.notify();
                    })
                    .ok();
                    None
                }
                // Cancelled, or the platform picker vanished.
                Err(_) => None,
            };

            if let Some(path) = picked {
                this.update(cx, |view, cx| view.set_input(path, cx)).ok();
            }
        })
        .detach();
    }

    fn set_input(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.job
            .note(Level::Info, None, format!("输入文件：{}", path.display()));
        if self.output.is_none() {
            self.output = Some(bridge::suggested_output(&path));
        }
        self.input = Some(path.clone());
        self.manifest_rows.clear();
        self.probing = true;
        self.start_probe(path, cx);
        self.log_scroll.scroll_to_bottom();
        cx.notify();
    }

    /// Probes off the UI thread; the result lands back on the view.
    fn start_probe(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let Some(engine) = self.engine.clone() else {
            self.probing = false;
            return;
        };
        let ff = engine.ffmpeg();

        cx.spawn(async move |this, cx| {
            let result = cx
                .background_executor()
                .spawn(async move { bridge::probe_rows(ff, path) })
                .await;

            this.update(cx, |view, cx| {
                view.probing = false;
                match result {
                    Ok(rows) => {
                        view.manifest_rows = rows;
                        view.job.note(Level::Info, None, "媒体信息已更新");
                    }
                    Err(err) => {
                        view.manifest_rows.clear();
                        view.job
                            .note(Level::Error, None, format!("探测失败：{err}"));
                    }
                }
                view.log_scroll.scroll_to_bottom();
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn pick_output(&mut self, cx: &mut Context<Self>) {
        let start_dir = self
            .input
            .as_ref()
            .and_then(|path| path.parent().map(PathBuf::from))
            .or_else(|| {
                self.output
                    .as_ref()
                    .and_then(|path| path.parent().map(PathBuf::from))
            })
            .unwrap_or_else(|| PathBuf::from("."));
        let suggested = self
            .input
            .as_ref()
            .map(|path| bridge::suggested_output_name(path))
            .unwrap_or_else(|| "output.mkv".into());

        let prompt = cx.prompt_for_new_path(&start_dir, Some(&suggested));

        cx.spawn(async move |this, cx| {
            let picked = match prompt.await {
                Ok(Ok(path)) => path,
                Ok(Err(err)) => {
                    let message = format!("打开保存对话框失败：{err}");
                    this.update(cx, |view, cx| {
                        view.note_error(message);
                        cx.notify();
                    })
                    .ok();
                    None
                }
                Err(_) => None,
            };

            if let Some(path) = picked {
                this.update(cx, |view, cx| view.set_output(path, cx)).ok();
            }
        })
        .detach();
    }

    fn set_output(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.job
            .note(Level::Info, None, format!("输出文件：{}", path.display()));
        self.output = Some(path);
        self.log_scroll.scroll_to_bottom();
        cx.notify();
    }

    fn select_profile(&mut self, index: usize, cx: &mut Context<Self>) {
        if index >= self.profiles.len() {
            return;
        }
        self.profile_index = index;
        let described = self
            .profiles
            .get(index)
            .map(|profile| format!("{} — {}", profile.name, profile.description));
        if let Some(described) = described {
            self.job
                .note(Level::Info, None, format!("修复方案：{described}"));
        }
        self.log_scroll.scroll_to_bottom();
        cx.notify();
    }

    fn toggle_dry_run(&mut self, cx: &mut Context<Self>) {
        self.dry_run = !self.dry_run;
        cx.notify();
    }

    fn start_job(&mut self, cx: &mut Context<Self>) {
        let (Some(engine), Some(input), Some(output)) =
            (self.engine.clone(), self.input.clone(), self.output.clone())
        else {
            self.note_error("请先选择输入文件与输出文件");
            cx.notify();
            return;
        };

        let profile = self
            .profiles
            .get(self.profile_index)
            .cloned()
            .unwrap_or_default();

        // A cancel from a previous run must not leak into this one.
        engine.clear_cancel();

        let job_id = bridge::new_job_id();
        self.job.begin(&job_id, &profile.name);
        self.job.note(
            Level::Info,
            None,
            format!(
                "任务 {job_id} 已提交 · 方案 {} · {}",
                profile.name,
                if self.dry_run {
                    "试运行（仅分析）"
                } else {
                    "正式转换"
                }
            ),
        );

        let request = PlanRequest {
            job_id,
            input,
            output,
            profile,
        };
        let options = RunnerOptions {
            dry_run: self.dry_run,
            ..Default::default()
        };

        if let Err(err) = engine.spawn_job(request, options) {
            self.job.running = false;
            self.job
                .note(Level::Error, None, format!("无法启动转换线程：{err}"));
        }

        self.log_scroll.scroll_to_bottom();
        cx.notify();
    }

    fn cancel_job(&mut self, cx: &mut Context<Self>) {
        if let Some(engine) = &self.engine {
            engine.request_cancel();
        }
        self.job
            .note(Level::Warn, None, "已请求取消，正在等待当前步骤安全退出…");
        self.log_scroll.scroll_to_bottom();
        cx.notify();
    }

    fn clear_logs(&mut self, cx: &mut Context<Self>) {
        self.job.clear_logs();
        cx.notify();
    }

    fn set_level_filter(&mut self, filter: LevelFilter, cx: &mut Context<Self>) {
        self.job.level_filter = filter;
        cx.notify();
    }

    // ---- rendering ------------------------------------------------------

    fn render_fatal(&self, error: &StartupError) -> AnyElement {
        div()
            .flex()
            .flex_col()
            .flex_1()
            .items_center()
            .justify_center()
            .gap_3()
            .p_8()
            .text_center()
            .child(
                div()
                    .text_xl()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(theme::ERR))
                    .child(error.title()),
            )
            .child(
                div()
                    .max_w(px(720.))
                    .text_sm()
                    .text_color(rgb(theme::TEXT))
                    .child(error.hint()),
            )
            .child(
                div()
                    .max_w(px(720.))
                    .p_3()
                    .rounded_md()
                    .bg(rgb(theme::PANEL))
                    .border_1()
                    .border_color(rgb(theme::BORDER))
                    .font_family("Consolas")
                    .text_xs()
                    .text_color(rgb(theme::TEXT_DIM))
                    .child(error.detail().to_string()),
            )
            .into_any_element()
    }

    fn render_header(&self) -> AnyElement {
        // Chrome shows only a readiness indicator; the versions and the
        // per-engine detail live in the 推理引擎 panel, which is where someone
        // debugging a backend will look for them.
        let engines = match self.engine.as_ref() {
            Some(engine) => {
                let (ready, total) = engine.engine_readiness();
                format!("引擎 {ready}/{total} 就绪")
            }
            None => "引擎不可用".into(),
        };

        div()
            .flex()
            .flex_none()
            .items_center()
            .justify_between()
            .px_4()
            .py_2()
            .bg(rgb(theme::PANEL))
            .border_b_1()
            .border_color(rgb(theme::BORDER))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .text_lg()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("sr-player · 无人值守修复"),
                    )
                    .child(widgets::tag(format!("v{}", sr_core::VERSION))),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(widgets::tag(engines)),
            )
            .into_any_element()
    }

    fn render_config(&self, cx: &mut Context<Self>) -> AnyElement {
        let ready = self.engine.is_some();
        let idle = !self.job.running;
        let can_pick = ready && idle;
        let can_run = ready && idle && self.input.is_some() && self.output.is_some();

        let input_path = self.input.clone();
        let output_path = self.output.clone();

        let paths = div()
            .flex()
            .gap_3()
            .child(widgets::path_field(
                "输入文件",
                path_text(&input_path),
                input_path.is_some(),
                widgets::button(
                    "pick-input",
                    "选择文件",
                    can_pick,
                    false,
                    widgets::on_click(cx, |view, cx| view.pick_input(cx)),
                ),
            ))
            .child(widgets::path_field(
                "输出文件",
                path_text(&output_path),
                output_path.is_some(),
                widgets::button(
                    "pick-output",
                    "选择输出",
                    can_pick,
                    false,
                    widgets::on_click(cx, |view, cx| view.pick_output(cx)),
                ),
            ));

        let chips: Vec<AnyElement> = self
            .profiles
            .iter()
            .enumerate()
            .map(|(index, profile)| {
                widgets::chip(
                    ("profile", index),
                    profile.name.clone(),
                    index == self.profile_index,
                    can_pick,
                    widgets::on_click(cx, move |view, cx| view.select_profile(index, cx)),
                )
            })
            .collect();

        let description = self
            .profiles
            .get(self.profile_index)
            .map(|profile| profile.description.clone())
            .unwrap_or_default();

        let controls = div()
            .flex()
            .items_center()
            .gap_3()
            .child(widgets::button(
                "start-job",
                "开始转换",
                can_run,
                true,
                widgets::on_click(cx, |view, cx| view.start_job(cx)),
            ))
            .child(widgets::button(
                "cancel-job",
                "取消",
                self.job.running,
                false,
                widgets::on_click(cx, |view, cx| view.cancel_job(cx)),
            ))
            .child(widgets::toggle(
                "dry-run",
                "试运行（仅分析）",
                self.dry_run,
                ready && idle,
                widgets::on_click(cx, |view, cx| view.toggle_dry_run(cx)),
            ))
            .child(div().flex_1())
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::TEXT_DIM))
                    .child(if self.probing {
                        "正在探测媒体信息…"
                    } else {
                        ""
                    }),
            );

        div()
            .flex()
            .flex_col()
            .flex_none()
            .gap_2()
            .px_4()
            .py_3()
            .bg(rgb(theme::PANEL))
            .border_b_1()
            .border_color(rgb(theme::BORDER))
            .child(paths)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .flex_none()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_FAINT))
                            .child("修复方案"),
                    )
                    .children(chips),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::TEXT_DIM))
                    .child(description),
            )
            .child(controls)
            .into_any_element()
    }

    fn render_banner(&self, cx: &mut Context<Self>) -> AnyElement {
        let Some(outcome) = self.job.outcome.as_ref() else {
            return div().into_any_element();
        };

        let (accent, background, title) = if outcome.ok {
            (theme::OK, theme::OK_SOFT, "转换完成")
        } else if self.job.was_cancelled() {
            (theme::WARN, theme::PANEL_ALT, "已取消")
        } else {
            (theme::ERR, theme::ERR_SOFT, "转换失败")
        };

        let mut head = div()
            .flex()
            .items_center()
            .gap_3()
            .child(
                div()
                    .flex_none()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(accent))
                    .child(title),
            )
            .child(
                div()
                    .flex_none()
                    .text_xs()
                    .text_color(rgb(theme::TEXT_DIM))
                    .child(format!("耗时 {}", state::format_duration(outcome.elapsed))),
            );

        if let Some(path) = outcome.output.as_ref() {
            head = head.child(
                div()
                    .flex_1()
                    .min_w(px(0.))
                    .text_xs()
                    .text_color(rgb(theme::TEXT))
                    .whitespace_nowrap()
                    .overflow_hidden()
                    .text_ellipsis()
                    .child(path.display().to_string()),
            );
        }

        let mut banner = div()
            .flex()
            .flex_col()
            .flex_none()
            .gap_2()
            .p_3()
            .rounded_md()
            .border_1()
            .border_color(rgb(accent))
            .bg(rgb(background))
            .child(head);

        if !outcome.ok {
            banner = banner.child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::ERR))
                    .child(outcome.message.clone()),
            );
        }

        if !outcome.degraded.is_empty() {
            banner = banner.child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::WARN))
                    .child(format!("降级步骤：{}", outcome.degraded.join("、"))),
            );
        }

        if let Some(path) = outcome.output.clone() {
            let reveal = path.clone();
            let play = path;
            banner = banner.child(
                div()
                    .flex()
                    .gap_2()
                    .child(widgets::button(
                        "reveal-output",
                        "打开所在文件夹",
                        true,
                        false,
                        widgets::on_click(cx, move |_view, cx| cx.reveal_path(&reveal)),
                    ))
                    .child(widgets::button(
                        "play-output",
                        "播放（外部播放器）",
                        true,
                        false,
                        widgets::on_click(cx, move |_view, cx| cx.open_with_system(&play)),
                    )),
            );
        }

        banner.into_any_element()
    }

    fn render_progress(&self) -> AnyElement {
        let fraction = self.job.overall_fraction();
        let percent = (fraction * 100.0).round() as i32;
        let finished = self.job.finished_stages();
        let complete = finished >= state::STAGE_COUNT;

        let current = self
            .job
            .running_stage()
            .map(state::stage_label)
            .unwrap_or("—");
        let detail = self
            .job
            .progress
            .as_ref()
            .map(|progress| progress.detail.clone())
            .unwrap_or_default();

        let mut headline = format!("当前阶段：{current}");
        if !detail.is_empty() {
            headline.push_str(" · ");
            headline.push_str(&detail);
        }

        let mut meta: Vec<String> = Vec::new();
        if let Some(progress) = self.job.progress.as_ref() {
            if let Some(fps) = progress.fps {
                meta.push(format!("{fps:.1} fps"));
            }
            if let Some(speed) = progress.speed {
                meta.push(format!("{speed:.2}x"));
            }
            if let Some(frames) = progress.frames {
                meta.push(format!("{frames} 帧"));
            }
            if let Some(out_time) = progress.out_time {
                meta.push(format!("已输出 {}", state::format_duration(out_time)));
            }
            if let Some(eta) = progress.eta {
                meta.push(format!("剩余 {}", state::format_duration(eta)));
            }
        }
        if let Some(elapsed) = self.job.elapsed() {
            meta.push(format!("已用时 {}", state::format_duration(elapsed)));
        }

        div()
            .flex()
            .flex_col()
            .flex_none()
            .gap_2()
            .p_3()
            .rounded_md()
            .border_1()
            .border_color(rgb(theme::BORDER))
            .bg(rgb(theme::PANEL))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_DIM))
                            .child("总体进度"),
                    )
                    .child(
                        div().text_xs().text_color(rgb(theme::TEXT)).child(format!(
                            "{finished}/{} · {percent}%",
                            state::STAGE_COUNT
                        )),
                    ),
            )
            .child(widgets::progress_bar(fraction, complete))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::TEXT))
                    .whitespace_nowrap()
                    .overflow_hidden()
                    .text_ellipsis()
                    .child(headline),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme::TEXT_FAINT))
                    .child(meta.join(" · ")),
            )
            .into_any_element()
    }

    fn render_ladder(&self) -> AnyElement {
        let rows: Vec<AnyElement> = self
            .job
            .stages
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let tint = match row.status {
                    StageStatus::Running => Some(theme::ACCENT_SOFT),
                    StageStatus::Failed => Some(theme::ERR_SOFT),
                    _ => None,
                };

                let mut line = div()
                    .flex()
                    .flex_none()
                    .items_center()
                    .gap_3()
                    .px_3()
                    .py_1()
                    .border_b_1()
                    .border_color(rgb(theme::BORDER))
                    .child(
                        div()
                            .w(px(20.))
                            .flex_shrink_0()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_FAINT))
                            .child(format!("{:02}", index + 1)),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.))
                            .text_sm()
                            .text_color(rgb(if row.status == StageStatus::Pending {
                                theme::TEXT_DIM
                            } else {
                                theme::TEXT
                            }))
                            .child(state::stage_label(row.stage)),
                    )
                    .child(
                        div()
                            .flex_none()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_FAINT))
                            .child(row.stage.id()),
                    )
                    .child(widgets::badge(row.status));

                if let Some(background) = tint {
                    line = line.bg(rgb(background));
                }

                let mut column = div().flex().flex_col().flex_none().child(line);
                if let Some(note) = row.note.as_ref().filter(|note| !note.is_empty()) {
                    column = column.child(
                        div()
                            .flex_none()
                            .pl(px(47.))
                            .pr_3()
                            .pb_1()
                            .text_xs()
                            .text_color(rgb(if row.status == StageStatus::Failed {
                                theme::ERR
                            } else {
                                theme::WARN
                            }))
                            .child(note.clone()),
                    );
                }
                column.into_any_element()
            })
            .collect();

        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.))
            .rounded_md()
            .border_1()
            .border_color(rgb(theme::BORDER))
            .bg(rgb(theme::PANEL))
            .overflow_hidden()
            .child(
                div()
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_3()
                    .py_2()
                    .border_b_1()
                    .border_color(rgb(theme::BORDER))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_DIM))
                            .child("阶段进度"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_FAINT))
                            .child(format!(
                                "{}/{}",
                                self.job.finished_stages(),
                                state::STAGE_COUNT
                            )),
                    ),
            )
            .child(
                div()
                    .id("ladder-scroll")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_h(px(0.))
                    .overflow_y_scroll()
                    .children(rows),
            )
            .into_any_element()
    }

    fn render_logs(&self, cx: &mut Context<Self>) -> AnyElement {
        let total = self.job.logs.len();
        let shown = self.job.visible_logs().count();
        let dropped = self.job.dropped_logs;

        let lines: Vec<AnyElement> = self
            .job
            .visible_logs()
            .map(|line| {
                div()
                    .flex_none()
                    .font_family("Consolas")
                    .text_xs()
                    .text_color(rgb(theme::level_color(line.level)))
                    .child(line.text.clone())
                    .into_any_element()
            })
            .collect();

        let filters: Vec<AnyElement> = LevelFilter::ALL
            .iter()
            .copied()
            .enumerate()
            .map(|(index, filter)| {
                widgets::chip(
                    ("log-filter", index),
                    filter.label(),
                    filter == self.job.level_filter,
                    true,
                    widgets::on_click(cx, move |view, cx| view.set_level_filter(filter, cx)),
                )
            })
            .collect();

        let mut count = format!("{shown}/{total} 行");
        if dropped > 0 {
            count.push_str(&format!("（已丢弃 {dropped} 行）"));
        }

        let body = if lines.is_empty() {
            vec![widgets::hint("（暂无日志）")]
        } else {
            lines
        };

        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.))
            .rounded_md()
            .border_1()
            .border_color(rgb(theme::BORDER))
            .bg(rgb(theme::PANEL))
            .overflow_hidden()
            .child(
                div()
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap_2()
                    .px_3()
                    .py_2()
                    .border_b_1()
                    .border_color(rgb(theme::BORDER))
                    .child(
                        div()
                            .flex_none()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_DIM))
                            .child("运行日志"),
                    )
                    .child(
                        div()
                            .flex_none()
                            .text_xs()
                            .text_color(rgb(theme::TEXT_FAINT))
                            .child(count),
                    )
                    .child(div().flex_1())
                    .children(filters)
                    .child(widgets::button(
                        "clear-logs",
                        "清空",
                        true,
                        false,
                        widgets::on_click(cx, |view, cx| view.clear_logs(cx)),
                    )),
            )
            .child(
                div()
                    .id("log-scroll")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_h(px(0.))
                    .gap_1()
                    .px_3()
                    .py_2()
                    .overflow_y_scroll()
                    .track_scroll(&self.log_scroll)
                    .children(body),
            )
            .into_any_element()
    }

    fn render_info(&self) -> AnyElement {
        let media = if self.manifest_rows.is_empty() {
            vec![widgets::hint(if self.probing {
                "正在探测…"
            } else {
                "尚未选择输入文件"
            })]
        } else {
            widgets::kv_rows(&self.manifest_rows)
        };

        let mut plan = widgets::kv_rows(&self.job.plan_rows);
        if plan.is_empty() {
            plan.push(widgets::hint(if self.job.running {
                "方案尚未生成"
            } else {
                "尚未生成方案"
            }));
        }
        for warning in &self.job.plan_warnings {
            plan.push(widgets::warning(format!("⚠ {warning}")));
        }

        let engine_rows = self
            .engine
            .as_ref()
            .map(|engine| engine.engines().report_rows())
            .unwrap_or_default();
        let engines = if engine_rows.is_empty() {
            vec![widgets::hint("没有可用的推理引擎")]
        } else {
            // The panel owns the full detail: the exact FFmpeg build line and what
            // the registry concluded, before the per-engine rows.
            let mut rows = match self.engine.as_ref() {
                Some(engine) => vec![
                    ("FFmpeg".to_string(), engine.ffmpeg_version().to_string()),
                    (
                        "结论".to_string(),
                        engine.engines().summary(),
                    ),
                ],
                None => Vec::new(),
            };
            rows.extend(engine_rows);
            widgets::kv_rows(&rows)
        };

        div()
            .id("info-scroll")
            .flex()
            .flex_col()
            .gap_3()
            .w(px(400.))
            .flex_shrink_0()
            .min_h(px(0.))
            .pr_1()
            .overflow_y_scroll()
            .track_scroll(&self.info_scroll)
            .child(widgets::card("媒体信息", media))
            .child(widgets::card("转换计划", plan))
            .child(widgets::card("推理引擎", engines))
            .into_any_element()
    }

    fn render_status(&self) -> AnyElement {
        let job = self
            .job
            .job_id
            .clone()
            .unwrap_or_else(|| "（无任务）".into());
        let job_state = self
            .job
            .state
            .map(state::job_state_label)
            .unwrap_or("空闲");

        // No log count here: the log panel's own header shows `shown/total 行`
        // and, when the ring buffer drops lines, how many were dropped.
        let left = format!("任务：{job}　状态：{job_state}");
        // Footer right is build provenance only: which engine build produced the
        // output, for reproducibility. The sr-core version is already the header
        // tag, and the full FFmpeg line lives in the 推理引擎 panel, so neither is
        // repeated here.
        let right = match self.engine.as_ref() {
            Some(engine) => format!("FFmpeg {}", engine.ffmpeg_version_short()),
            None => "引擎不可用".to_string(),
        };

        div()
            .flex()
            .flex_none()
            .items_center()
            .justify_between()
            .gap_3()
            .px_4()
            .h(px(26.))
            .bg(rgb(theme::PANEL))
            .border_t_1()
            .border_color(rgb(theme::BORDER))
            .text_xs()
            .text_color(rgb(theme::TEXT_DIM))
            .child(
                div()
                    .flex_none()
                    .whitespace_nowrap()
                    .overflow_hidden()
                    .text_ellipsis()
                    .child(left),
            )
            .child(
                div()
                    .flex_none()
                    .whitespace_nowrap()
                    .overflow_hidden()
                    .text_ellipsis()
                    .child(right),
            )
            .into_any_element()
    }

    fn render_body(&self, cx: &mut Context<Self>) -> AnyElement {
        div()
            .flex()
            .flex_1()
            .min_h(px(0.))
            .gap_3()
            .p_4()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_w(px(0.))
                    .min_h(px(0.))
                    .gap_3()
                    .child(self.render_banner(cx))
                    .child(self.render_progress())
                    .child(self.render_ladder())
                    .child(self.render_logs(cx)),
            )
            .child(self.render_info())
            .into_any_element()
    }
}

impl Render for AppView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut root = div()
            .flex()
            .flex_col()
            .size_full()
            .overflow_hidden()
            .bg(rgb(theme::BG))
            .text_color(rgb(theme::TEXT))
            .text_sm();

        match self.startup_error.as_ref() {
            Some(error) => {
                root = root
                    .child(self.render_header())
                    .child(self.render_fatal(error));
            }
            None => {
                root = root
                    .child(self.render_header())
                    .child(self.render_config(cx))
                    .child(self.render_body(cx))
                    .child(self.render_status());
            }
        }

        root
    }
}

/// `Some(path)` → the path, `None` → the 未选择 placeholder.
fn path_text(path: &Option<PathBuf>) -> String {
    match path {
        Some(path) => path.display().to_string(),
        None => "未选择".into(),
    }
}
