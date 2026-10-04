//! Live event stream and the pipeline vocabulary (stages, levels, statuses).
//!
//! The front end subscribes with [`EventBus::subscribe`] and never blocks the
//! engine: the bus is an unbounded broadcast, and the durable record of what
//! happened lives in SQLite via [`crate::state::Store`]. The [`Reporter`] writes
//! to both at once, which is what the pipeline uses everywhere.

use crate::pipeline::plan::ConversionPlan;
use crate::state::Store;
use crossbeam_channel::{unbounded, Receiver, Sender};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Log severity. Ordered so `Level::Warn >= Level::Info` works for filtering.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl Level {
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Trace => "TRACE",
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
        }
    }

    pub fn parse(s: &str) -> Level {
        match s.to_ascii_uppercase().as_str() {
            "TRACE" => Level::Trace,
            "DEBUG" => Level::Debug,
            "WARN" | "WARNING" => Level::Warn,
            "ERROR" | "FATAL" => Level::Error,
            _ => Level::Info,
        }
    }
}

/// The fixed pipeline. The GUI renders exactly this list as the progress ladder.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Probe,
    Temporal,
    Scenes,
    AudioAnalysis,
    Plan,
    Restore,
    Interpolate,
    Regrain,
    AudioProcess,
    Encode,
    Mux,
    Qc,
    Done,
}

impl Stage {
    pub const ALL: [Stage; 13] = [
        Stage::Probe,
        Stage::Temporal,
        Stage::Scenes,
        Stage::AudioAnalysis,
        Stage::Plan,
        Stage::Restore,
        Stage::Interpolate,
        Stage::Regrain,
        Stage::AudioProcess,
        Stage::Encode,
        Stage::Mux,
        Stage::Qc,
        Stage::Done,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Stage::Probe => "probe",
            Stage::Temporal => "temporal",
            Stage::Scenes => "scenes",
            Stage::AudioAnalysis => "audio_analysis",
            Stage::Plan => "plan",
            Stage::Restore => "restore",
            Stage::Interpolate => "interpolate",
            Stage::Regrain => "regrain",
            Stage::AudioProcess => "audio_process",
            Stage::Encode => "encode",
            Stage::Mux => "mux",
            Stage::Qc => "qc",
            Stage::Done => "done",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Stage::Probe => "Probe source",
            Stage::Temporal => "Classify cadence",
            Stage::Scenes => "Detect shots",
            Stage::AudioAnalysis => "Analyse audio",
            Stage::Plan => "Resolve plan",
            Stage::Restore => "Restore / upscale",
            Stage::Interpolate => "Interpolate",
            Stage::Regrain => "Re-grain",
            Stage::AudioProcess => "Dialogue remaster",
            Stage::Encode => "Encode video",
            Stage::Mux => "Mux output",
            Stage::Qc => "Quality control",
            Stage::Done => "Finished",
        }
    }

    pub fn order(self) -> usize {
        Stage::ALL.iter().position(|s| *s == self).unwrap_or(0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Pending,
    Running,
    Done,
    Skipped,
    Degraded,
    Failed,
}

impl StageStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            StageStatus::Done | StageStatus::Skipped | StageStatus::Failed
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobState {
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Completed => "completed",
            JobState::Failed => "failed",
            JobState::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> JobState {
        match s {
            "running" => JobState::Running,
            "completed" => JobState::Completed,
            "failed" => JobState::Failed,
            "cancelled" => JobState::Cancelled,
            _ => JobState::Queued,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogRecord {
    pub seq: u64,
    pub at_ms: u64,
    pub level: Level,
    pub stage: Option<Stage>,
    pub message: String,
}

impl LogRecord {
    pub fn now(seq: u64, level: Level, stage: Option<Stage>, message: impl Into<String>) -> Self {
        LogRecord {
            seq,
            at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            level,
            stage,
            message: message.into(),
        }
    }

    /// `12:34:56.789  INFO   [scenes] message`
    pub fn format_line(&self) -> String {
        let ms = self.at_ms % 86_400_000;
        let (h, m, s, milli) = (
            ms / 3_600_000,
            (ms / 60_000) % 60,
            (ms / 1000) % 60,
            ms % 1000,
        );
        match self.stage {
            Some(stage) => format!(
                "{h:02}:{m:02}:{s:02}.{milli:03} {:5} [{}] {}",
                self.level.as_str(),
                stage.id(),
                self.message
            ),
            None => format!(
                "{h:02}:{m:02}:{s:02}.{milli:03} {:5} {}",
                self.level.as_str(),
                self.message
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageProgress {
    pub stage: Stage,
    /// 0.0..=1.0, or `None` when the stage cannot estimate itself.
    pub fraction: Option<f32>,
    pub detail: String,
    pub frames: Option<u64>,
    pub fps: Option<f64>,
    pub speed: Option<f64>,
    pub out_time: Option<Duration>,
    pub eta: Option<Duration>,
}

impl StageProgress {
    pub fn new(stage: Stage, detail: impl Into<String>) -> Self {
        StageProgress {
            stage,
            fraction: None,
            detail: detail.into(),
            frames: None,
            fps: None,
            speed: None,
            out_time: None,
            eta: None,
        }
    }

    pub fn with_fraction(mut self, f: f32) -> Self {
        self.fraction = Some(f.clamp(0.0, 1.0));
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobOutcome {
    pub job_id: String,
    pub ok: bool,
    pub output: Option<PathBuf>,
    pub message: String,
    pub elapsed: Duration,
    /// Stages that had to walk down the degrade ladder, for the report.
    pub degraded: Vec<String>,
}

/// Everything the engine tells the outside world.
#[derive(Clone, Debug)]
pub enum Event {
    Log(LogRecord),
    Stage {
        stage: Stage,
        status: StageStatus,
        note: Option<String>,
    },
    Progress(StageProgress),
    Plan(Box<ConversionPlan>),
    Job {
        job_id: String,
        state: JobState,
    },
    Finished(JobOutcome),
}

struct BusInner {
    subs: Mutex<Vec<Sender<Event>>>,
    seq: AtomicU64,
}

/// Broadcast bus. Cloning shares the same subscribers.
#[derive(Clone)]
pub struct EventBus {
    inner: Arc<BusInner>,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    pub fn new() -> Self {
        EventBus {
            inner: Arc::new(BusInner {
                subs: Mutex::new(Vec::new()),
                seq: AtomicU64::new(1),
            }),
        }
    }

    pub fn subscribe(&self) -> Receiver<Event> {
        let (tx, rx) = unbounded();
        self.inner.subs.lock().push(tx);
        rx
    }

    pub fn subscriber_count(&self) -> usize {
        self.inner.subs.lock().len()
    }

    pub fn next_seq(&self) -> u64 {
        self.inner.seq.fetch_add(1, Ordering::Relaxed)
    }

    pub fn emit(&self, event: Event) {
        let mut subs = self.inner.subs.lock();
        subs.retain(|tx| match tx.send(event.clone()) {
            Ok(()) => true,
            // a dropped receiver means the UI went away; stop feeding it
            Err(_) => false,
        });
    }
}

/// Writes a log line to the live bus *and* to the durable store.
#[derive(Clone)]
pub struct Reporter {
    bus: EventBus,
    store: Option<Arc<Store>>,
    job_id: Option<String>,
    /// Mirror everything into `tracing` as well, so `RUST_LOG` debugging works.
    mirror: bool,
}

impl Reporter {
    pub fn new(bus: EventBus) -> Self {
        Reporter {
            bus,
            store: None,
            job_id: None,
            // The product surfaces (GUI, CLI) render events themselves; the
            // tracing mirror exists for `RUST_LOG` debugging and would otherwise
            // print every line twice.
            mirror: false,
        }
    }

    pub fn with_store(bus: EventBus, store: Arc<Store>, job_id: impl Into<String>) -> Self {
        Reporter {
            bus,
            store: Some(store),
            job_id: Some(job_id.into()),
            mirror: false,
        }
    }

    /// Also forward every line to `tracing`.
    pub fn with_mirror(mut self, mirror: bool) -> Self {
        self.mirror = mirror;
        self
    }

    pub fn bus(&self) -> &EventBus {
        &self.bus
    }

    pub fn log(&self, level: Level, stage: Option<Stage>, message: impl Into<String>) {
        let record = LogRecord::now(self.bus.next_seq(), level, stage, message);
        if self.mirror {
            let line = record.format_line();
            match level {
                Level::Error => tracing::error!("{line}"),
                Level::Warn => tracing::warn!("{line}"),
                Level::Info => tracing::info!("{line}"),
                Level::Debug => tracing::debug!("{line}"),
                Level::Trace => tracing::trace!("{line}"),
            }
        }
        if let (Some(store), Some(job_id)) = (&self.store, &self.job_id) {
            // Logging must never take the pipeline down.
            if let Err(err) = store.append_log(job_id, &record) {
                tracing::warn!("failed to persist log line: {err}");
            }
        }
        self.bus.emit(Event::Log(record));
    }

    pub fn trace(&self, stage: Option<Stage>, message: impl Into<String>) {
        self.log(Level::Trace, stage, message);
    }

    pub fn debug(&self, stage: Option<Stage>, message: impl Into<String>) {
        self.log(Level::Debug, stage, message);
    }

    pub fn info(&self, stage: Option<Stage>, message: impl Into<String>) {
        self.log(Level::Info, stage, message);
    }

    pub fn warn(&self, stage: Option<Stage>, message: impl Into<String>) {
        self.log(Level::Warn, stage, message);
    }

    pub fn error(&self, stage: Option<Stage>, message: impl Into<String>) {
        self.log(Level::Error, stage, message);
    }

    pub fn stage(&self, stage: Stage, status: StageStatus, note: Option<String>) {
        if let Some(store) = &self.store {
            if let Some(job_id) = &self.job_id {
                let _ = store.set_stage_status(job_id, stage, status, note.as_deref());
            }
        }
        self.bus.emit(Event::Stage {
            stage,
            status,
            note,
        });
    }

    pub fn progress(&self, progress: StageProgress) {
        self.bus.emit(Event::Progress(progress));
    }

    pub fn emit(&self, event: Event) {
        self.bus.emit(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_broadcasts_to_every_subscriber() {
        let bus = EventBus::new();
        let a = bus.subscribe();
        let b = bus.subscribe();
        bus.emit(Event::Stage {
            stage: Stage::Probe,
            status: StageStatus::Running,
            note: None,
        });
        assert!(matches!(a.try_recv(), Ok(Event::Stage { .. })));
        assert!(matches!(b.try_recv(), Ok(Event::Stage { .. })));
    }

    #[test]
    fn dropped_subscribers_are_forgotten() {
        let bus = EventBus::new();
        {
            let _rx = bus.subscribe();
        }
        assert_eq!(bus.subscriber_count(), 1);
        bus.emit(Event::Job {
            job_id: "j".into(),
            state: JobState::Running,
        });
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn stage_ids_are_unique_and_ordered() {
        let mut ids: Vec<&str> = Stage::ALL.iter().map(|s| s.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), Stage::ALL.len());
        assert!(Stage::Probe.order() < Stage::Qc.order());
        assert_eq!(Stage::Done.order(), Stage::ALL.len() - 1);
    }

    #[test]
    fn log_line_formatting_carries_level_and_stage() {
        let rec = LogRecord::now(1, Level::Warn, Some(Stage::Encode), "using hevc_nvenc");
        let line = rec.format_line();
        assert!(line.contains("WARN"));
        assert!(line.contains("[encode]"));
        assert!(line.contains("using hevc_nvenc"));
    }
}
