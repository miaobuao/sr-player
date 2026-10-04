//! Everything the window needs to draw, plus the fold from `sr_core::Event`
//! into it. Pure data and pure functions: no GPUI types live here, which keeps
//! the event pump trivial and the rendering purely a function of this state.

use sr_core::{Event, JobOutcome, JobState, Level, LogRecord, Stage, StageProgress, StageStatus};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Hard cap on retained log lines; the oldest ones are dropped first.
pub const MAX_LOG_LINES: usize = 2000;

/// How many of the 13 pipeline stages the ladder shows.
pub const STAGE_COUNT: usize = Stage::ALL.len();

/// Which severities the log panel displays.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LevelFilter {
    All,
    Info,
    Warn,
    Error,
}

impl LevelFilter {
    pub const ALL: [LevelFilter; 4] = [
        LevelFilter::All,
        LevelFilter::Info,
        LevelFilter::Warn,
        LevelFilter::Error,
    ];

    pub fn label(self) -> &'static str {
        match self {
            LevelFilter::All => "全部",
            LevelFilter::Info => "信息",
            LevelFilter::Warn => "警告",
            LevelFilter::Error => "错误",
        }
    }

    fn accepts(self, level: Level) -> bool {
        match self {
            LevelFilter::All => true,
            LevelFilter::Info => level >= Level::Info,
            LevelFilter::Warn => level >= Level::Warn,
            LevelFilter::Error => level >= Level::Error,
        }
    }
}

/// Chinese name of a pipeline stage. `sr_core::Stage::label` is English and is
/// used in the logs; the ladder is operator-facing, hence the second table.
pub fn stage_label(stage: Stage) -> &'static str {
    match stage {
        Stage::Probe => "探测源",
        Stage::Temporal => "判断场序/胶片节奏",
        Stage::Scenes => "镜头切分",
        Stage::AudioAnalysis => "音频分析",
        Stage::Plan => "生成方案",
        Stage::Restore => "修复/放大",
        Stage::Interpolate => "补帧",
        Stage::Regrain => "重加颗粒",
        Stage::AudioProcess => "对白重制",
        Stage::Encode => "编码",
        Stage::Mux => "封装",
        Stage::Qc => "质量校验",
        Stage::Done => "完成",
    }
}

/// Chinese badge for a stage status.
pub fn status_label(status: StageStatus) -> &'static str {
    match status {
        StageStatus::Pending => "待处理",
        StageStatus::Running => "进行中",
        StageStatus::Done => "完成",
        StageStatus::Skipped => "跳过",
        StageStatus::Degraded => "降级",
        StageStatus::Failed => "失败",
    }
}

/// Chinese badge for the job state machine.
pub fn job_state_label(state: JobState) -> &'static str {
    match state {
        JobState::Queued => "排队中",
        JobState::Running => "运行中",
        JobState::Completed => "已完成",
        JobState::Failed => "失败",
        JobState::Cancelled => "已取消",
    }
}

/// One line in the log panel. The text is pre-formatted by `sr-core`, so the
/// GUI and the CLI show byte-identical lines.
#[derive(Clone, Debug)]
pub struct LogLine {
    pub level: Level,
    pub text: String,
}

impl LogLine {
    /// A line originating in the GUI itself.
    pub fn local(level: Level, stage: Option<Stage>, message: impl Into<String>) -> Self {
        LogLine {
            level,
            text: LogRecord::now(0, level, stage, message).format_line(),
        }
    }

    fn from_record(record: &LogRecord) -> Self {
        LogLine {
            level: record.level,
            text: record.format_line(),
        }
    }
}

/// One rung of the ladder.
#[derive(Clone, Debug)]
pub struct StageRow {
    pub stage: Stage,
    pub status: StageStatus,
    pub note: Option<String>,
}

impl StageRow {
    /// Stages that will not report again: they count as a full step of progress.
    fn counts_as_finished(&self) -> bool {
        matches!(
            self.status,
            StageStatus::Done
                | StageStatus::Skipped
                | StageStatus::Degraded
                | StageStatus::Failed
        )
    }
}

/// Everything known about the current (or most recent) job.
pub struct JobView {
    pub running: bool,
    pub job_id: Option<String>,
    pub state: Option<JobState>,
    pub profile_name: Option<String>,
    pub stages: Vec<StageRow>,
    pub progress: Option<StageProgress>,
    pub started_at: Option<Instant>,
    pub outcome: Option<JobOutcome>,
    pub plan_rows: Vec<(String, String)>,
    pub plan_warnings: Vec<String>,
    pub logs: VecDeque<LogLine>,
    pub level_filter: LevelFilter,
    /// How many lines fell off the front of the ring buffer.
    pub dropped_logs: usize,
}

impl Default for JobView {
    fn default() -> Self {
        JobView::new()
    }
}

impl JobView {
    pub fn new() -> Self {
        JobView {
            running: false,
            job_id: None,
            state: None,
            profile_name: None,
            stages: Stage::ALL
                .iter()
                .map(|stage| StageRow {
                    stage: *stage,
                    status: StageStatus::Pending,
                    note: None,
                })
                .collect(),
            progress: None,
            started_at: None,
            outcome: None,
            plan_rows: Vec::new(),
            plan_warnings: Vec::new(),
            logs: VecDeque::new(),
            level_filter: LevelFilter::All,
            dropped_logs: 0,
        }
    }

    /// Resets the per-run state at the moment a job is handed to the engine.
    /// The log is kept: it doubles as the session history.
    pub fn begin(&mut self, job_id: &str, profile_name: &str) {
        self.running = true;
        self.job_id = Some(job_id.to_string());
        self.state = Some(JobState::Queued);
        self.profile_name = Some(profile_name.to_string());
        self.stages = Stage::ALL
            .iter()
            .map(|stage| StageRow {
                stage: *stage,
                status: StageStatus::Pending,
                note: None,
            })
            .collect();
        self.progress = None;
        self.started_at = Some(Instant::now());
        self.outcome = None;
        self.plan_rows.clear();
        self.plan_warnings.clear();
    }

    /// Appends one line, trimming the ring buffer and reporting whether the
    /// panel actually changed.
    pub fn push_log(&mut self, line: LogLine) -> bool {
        self.logs.push_back(line);
        while self.logs.len() > MAX_LOG_LINES {
            self.logs.pop_front();
            self.dropped_logs += 1;
        }
        true
    }

    /// Logs a line the GUI produced itself (dialog failures, probe failures…).
    pub fn note(&mut self, level: Level, stage: Option<Stage>, message: impl Into<String>) {
        self.push_log(LogLine::local(level, stage, message));
    }

    pub fn clear_logs(&mut self) {
        self.logs.clear();
        self.dropped_logs = 0;
    }

    /// The lines the current filter lets through, oldest first.
    pub fn visible_logs(&self) -> impl Iterator<Item = &LogLine> {
        let filter = self.level_filter;
        self.logs.iter().filter(move |line| filter.accepts(line.level))
    }

    /// Wall-clock time: ticking while the job runs, frozen once it is over.
    pub fn elapsed(&self) -> Option<Duration> {
        if self.running {
            self.started_at.map(|started| started.elapsed())
        } else {
            self.outcome.as_ref().map(|outcome| outcome.elapsed)
        }
    }

    /// How many rungs are already behind us.
    pub fn finished_stages(&self) -> usize {
        self.stages.iter().filter(|row| row.counts_as_finished()).count()
    }

    /// `(finished stages + fraction of the running one) / 13`.
    pub fn overall_fraction(&self) -> f32 {
        let mut done = 0.0f32;
        for row in &self.stages {
            if row.counts_as_finished() {
                done += 1.0;
            } else if row.status == StageStatus::Running {
                done += self
                    .progress
                    .as_ref()
                    .filter(|progress| progress.stage == row.stage)
                    .and_then(|progress| progress.fraction)
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
            }
        }
        (done / STAGE_COUNT as f32).clamp(0.0, 1.0)
    }

    /// The stage whose badge says 进行中, if any.
    pub fn running_stage(&self) -> Option<Stage> {
        self.stages
            .iter()
            .find(|row| row.status == StageStatus::Running)
            .map(|row| row.stage)
    }

    pub fn was_cancelled(&self) -> bool {
        self.state == Some(JobState::Cancelled)
    }

    /// Folds one engine event into the view state. Returns `true` when the
    /// window needs to be redrawn.
    pub fn apply(&mut self, event: Event) -> bool {
        match event {
            Event::Log(record) => self.push_log(LogLine::from_record(&record)),

            Event::Stage {
                stage,
                status,
                note,
            } => {
                if let Some(row) = self.stages.iter_mut().find(|row| row.stage == stage) {
                    row.status = status;
                    row.note = note;
                    true
                } else {
                    false
                }
            }

            Event::Progress(progress) => {
                // A stage can stream progress before its Stage event lands;
                // keep the ladder honest either way.
                if let Some(row) = self.stages.iter_mut().find(|row| row.stage == progress.stage) {
                    if row.status == StageStatus::Pending {
                        row.status = StageStatus::Running;
                    }
                }
                self.progress = Some(progress);
                true
            }

            Event::Plan(plan) => {
                self.plan_rows = plan.summary_rows();
                self.plan_warnings = plan.warnings.clone();
                true
            }

            Event::Job { job_id, state } => {
                self.job_id = Some(job_id);
                self.state = Some(state);
                if matches!(
                    state,
                    JobState::Completed | JobState::Failed | JobState::Cancelled
                ) {
                    self.running = false;
                }
                true
            }

            Event::Finished(outcome) => {
                self.running = false;
                if !self.was_cancelled() {
                    self.state = Some(if outcome.ok {
                        JobState::Completed
                    } else {
                        JobState::Failed
                    });
                }
                self.progress = None;
                self.outcome = Some(outcome);
                true
            }
        }
    }
}

/// `1:02:03`, `02:03` — used for elapsed time, ETA and `out_time`.
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let (hours, minutes, seconds) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}
