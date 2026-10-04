//! The only place that owns engine handles and worker threads.
//!
//! `sr-gui` never blocks its UI thread: discovering FFmpeg and probing happen
//! here (on the caller's executor), and the conversion runs on a dedicated
//! `std::thread` because `PipelineRunner::run` is blocking by design.

use sr_core::ffmpeg::Ffmpeg;
use sr_core::media::probe;
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::{Event, EventBus, JobOutcome};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Why the app cannot run a job at all. Both variants disable the whole UI.
pub enum StartupError {
    /// FFmpeg is the one hard external dependency: without it there is no app.
    Ffmpeg(String),
    /// The SQLite job/checkpoint store could not be opened.
    Store(String),
}

impl StartupError {
    pub fn title(&self) -> &'static str {
        match self {
            StartupError::Ffmpeg(_) => "未找到 FFmpeg",
            StartupError::Store(_) => "无法打开任务数据库",
        }
    }

    pub fn hint(&self) -> &'static str {
        match self {
            StartupError::Ffmpeg(_) => {
                "sr-player 依赖 FFmpeg 完成全部媒体工作。请安装 FFmpeg，\
                 并确保 ffmpeg.exe 与 ffprobe.exe 在 PATH 中（或用 SR_FFMPEG 指定路径），\
                 然后重新启动本程序。"
            }
            StartupError::Store(_) => {
                "作业断点数据库无法打开，转换将无法开始。请检查该路径的写入权限，\
                 或设置 SR_STATE_DB 指向一个可写位置后重新启动本程序。"
            }
        }
    }

    pub fn detail(&self) -> &str {
        match self {
            StartupError::Ffmpeg(detail) | StartupError::Store(detail) => detail,
        }
    }
}

/// Engine handles. Cloning is cheap (all `Arc`s) and every field is `Send`, so
/// a clone can be moved into the conversion thread.
#[derive(Clone)]
pub struct Engine {
    ff: Arc<Ffmpeg>,
    store: Arc<Store>,
    bus: EventBus,
    cancel: Arc<AtomicBool>,
}

impl Engine {
    /// Discovers FFmpeg and opens the job store. Called once, from
    /// `AppView::new`.
    ///
    /// There is no inference probe to run alongside it any more: the AI runtime is
    /// a statically linked part of the binary, not something to be discovered, so
    /// the only things that can fail here are the two external dependencies.
    pub fn bootstrap() -> Result<Engine, StartupError> {
        let ff = Ffmpeg::discover().map_err(|err| StartupError::Ffmpeg(err.to_string()))?;
        let ff = Arc::new(ff);

        let store_path = Store::default_path();
        let store = Store::open(&store_path).map_err(|err| {
            StartupError::Store(format!("{} — {err}", store_path.display()))
        })?;

        Ok(Engine {
            ff,
            store: Arc::new(store),
            bus: EventBus::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn bus(&self) -> &EventBus {
        &self.bus
    }

    pub fn ffmpeg(&self) -> Arc<Ffmpeg> {
        Arc::clone(&self.ff)
    }

    pub fn ffmpeg_version(&self) -> &str {
        &self.ff.version
    }

    /// Compact form for the status bar; the full line goes in the runtime panel.
    pub fn ffmpeg_version_short(&self) -> String {
        self.ff.version_short()
    }

    /// What the AI layer can do, for the header indicator.
    ///
    /// It is a status line rather than a `(ready, total)` pair because there is
    /// nothing to count any more: exactly one runtime exists, and either it is
    /// built into this binary or the two model tasks are refused.
    pub fn ai_runtime(&self) -> &'static str {
        sr_core::pipeline::ai_runtime_line()
    }

    /// Asks the running job (if any) to stop at the next safe point.
    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Called before every new job so a previous cancel cannot poison it.
    pub fn clear_cancel(&self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    /// Runs one job on a dedicated thread. Everything the thread needs is moved
    /// in; the engine itself stays usable for the next run.
    pub fn spawn_job(
        &self,
        request: PlanRequest,
        options: RunnerOptions,
    ) -> std::io::Result<()> {
        let scratch = scratch_dir(&request.output);
        let runner = PipelineRunner::new(
            Arc::clone(&self.ff),
            self.bus.clone(),
            Arc::clone(&self.store),
            Arc::clone(&self.cancel),
            scratch.clone(),
        );

        let bus = self.bus.clone();
        let job_id = request.job_id.clone();

        std::thread::Builder::new()
            .name("sr-gui-job".into())
            .spawn(move || {
                let started = Instant::now();
                if let Err(err) = std::fs::create_dir_all(&scratch) {
                    bus.emit(Event::Finished(JobOutcome {
                        job_id,
                        ok: false,
                        output: None,
                        message: format!("无法创建临时目录 {}：{err}", scratch.display()),
                        elapsed: started.elapsed(),
                        degraded: Vec::new(),
                    }));
                    return;
                }
                // `PipelineRunner::run` emits `Event::Finished` itself on every
                // path that reaches its end; an `Err` here means it bailed out
                // before that, so the UI still needs to hear about it.
                if let Err(err) = runner.run(request, &options) {
                    bus.emit(Event::Finished(JobOutcome {
                        job_id,
                        ok: false,
                        output: None,
                        message: err.to_string(),
                        elapsed: started.elapsed(),
                        degraded: Vec::new(),
                    }));
                }
            })
            .map(|_| ())
    }
}

/// Probes one file. Blocking by design — always call it on an executor.
pub fn probe_rows(ff: Arc<Ffmpeg>, path: PathBuf) -> Result<Vec<(String, String)>, String> {
    probe(&ff, &path)
        .map(|manifest| manifest.summary_lines())
        .map_err(|err| err.to_string())
}

/// `<output parent>/.sr-scratch`, where the pipeline keeps its intermediates.
pub fn scratch_dir(output: &Path) -> PathBuf {
    output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(".sr-scratch")
}

/// `gui-<unix seconds>-<counter>`: unique within a session, sortable, and
/// readable in the SQLite store without pulling in a uuid dependency.
pub fn new_job_id() -> String {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("gui-{seconds}-{counter}")
}

/// `movie.mkv` → `movie.restored.mkv`, next to the source like the CLI does.
pub fn suggested_output(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".into());
    input
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}.restored.mkv"))
}

/// The file name suggested in the "save as" dialog.
pub fn suggested_output_name(input: &Path) -> String {
    suggested_output(input)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "output.mkv".into())
}
