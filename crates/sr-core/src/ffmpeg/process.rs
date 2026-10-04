//! Child-process supervision.
//!
//! Every external tool invocation in the engine goes through here so that
//! cancellation, log capture, OOM classification and progress reporting behave
//! identically for `ffmpeg`, `ffprobe` and any optional plugin.

use crate::error::{looks_like_oom, Error, Result};
use crate::events::{Level, Reporter, Stage, StageProgress};
use crate::ffmpeg::progress::ProgressParser;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Max bytes of stderr we keep for OOM classification / failure reporting.
const STDERR_HARD_CAP: usize = 256 * 1024;
/// Progress is throttled to this rate so a fast encode cannot flood the UI.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(200);

type Sink = Arc<dyn Fn(&str) + Send + Sync>;

/// How one child invocation should be run and reported.
pub struct RunSpec<'a> {
    pub stage: Stage,
    pub label: &'a str,
    pub expected_duration: Option<Duration>,
    pub total_frames: Option<u64>,
    /// Receives every stderr line (parsers for `idet`, `ebur128`, `showinfo`).
    pub stderr_sink: Option<Sink>,
    /// Substrings that suppress logging (the sink still sees the line).
    pub stderr_quiet: Vec<String>,
    pub log_stderr: bool,
    /// Severity used when logging stderr lines.
    ///
    /// Analysis passes run FFmpeg at `info` level because that is where `idet`
    /// and `ebur128` print, and that also dumps the whole input layout. Those
    /// passes ask for [`Level::Debug`] so the product log stays readable while
    /// `--verbose` still has everything.
    pub stderr_level: Level,
    pub stderr_tail: usize,
}

impl<'a> RunSpec<'a> {
    pub fn new(stage: Stage, label: &'a str) -> Self {
        RunSpec {
            stage,
            label,
            expected_duration: None,
            total_frames: None,
            stderr_sink: None,
            stderr_quiet: Vec::new(),
            log_stderr: true,
            stderr_level: Level::Warn,
            stderr_tail: 40,
        }
    }

    pub fn stderr_level(mut self, level: Level) -> Self {
        self.stderr_level = level;
        self
    }

    pub fn with_duration(mut self, d: Duration) -> Self {
        self.expected_duration = Some(d);
        self
    }

    pub fn with_frames(mut self, frames: u64) -> Self {
        self.total_frames = Some(frames);
        self
    }

    pub fn with_sink(mut self, sink: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.stderr_sink = Some(Arc::new(sink));
        self
    }

    pub fn quiet(mut self, pattern: &str) -> Self {
        self.stderr_quiet.push(pattern.to_ascii_lowercase());
        self
    }
}

#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub code: i32,
    pub stderr_tail: String,
    pub elapsed: Duration,
}

/// Shared stderr drain: forwards to the sink, logs what survives the quiet
/// filter, and remembers the tail plus a bounded copy of everything.
struct StderrPump {
    tail: Arc<Mutex<VecDeque<String>>>,
    all: Arc<Mutex<String>>,
}

impl StderrPump {
    fn start<R: Read + Send + 'static>(
        reader: R,
        spec: StderrSinkConfig,
        reporter: Reporter,
    ) -> (Self, JoinHandle<()>) {
        let tail = Arc::new(Mutex::new(VecDeque::with_capacity(spec.tail_cap)));
        let all = Arc::new(Mutex::new(String::new()));
        let pump = StderrPump {
            tail: Arc::clone(&tail),
            all: Arc::clone(&all),
        };
        let handle = std::thread::Builder::new()
            .name(format!("stderr-{}", spec.label))
            .spawn(move || {
                let reader = BufReader::new(reader);
                for line in reader.lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(_) => break,
                    };
                    if let Some(sink) = &spec.sink {
                        sink(&line);
                    }
                    {
                        let mut all = all.lock();
                        if all.len() < STDERR_HARD_CAP {
                            all.push_str(&line);
                            all.push('\n');
                        }
                    }
                    {
                        let mut tail = tail.lock();
                        if tail.len() == spec.tail_cap {
                            tail.pop_front();
                        }
                        tail.push_back(line.clone());
                    }
                    if spec.log_stderr && !spec.is_quiet(&line) {
                        reporter.log(spec.level, Some(spec.stage), line);
                    }
                }
            })
            .expect("spawn stderr pump");
        (pump, handle)
    }

    fn tail_string(&self) -> String {
        self.tail.lock().iter().cloned().collect::<Vec<_>>().join("\n")
    }

    fn looks_like_oom(&self) -> bool {
        looks_like_oom(&self.all.lock())
    }
}

struct StderrSinkConfig {
    label: String,
    stage: Stage,
    sink: Option<Sink>,
    quiet: Vec<String>,
    log_stderr: bool,
    level: Level,
    tail_cap: usize,
}

impl StderrSinkConfig {
    fn is_quiet(&self, line: &str) -> bool {
        if self.quiet.is_empty() {
            return false;
        }
        let lower = line.to_ascii_lowercase();
        self.quiet.iter().any(|q| lower.contains(q.as_str()))
    }
}

/// Runs a short-lived tool and returns its stdout (ffprobe, `ffmpeg -h`, ...).
pub fn capture(program: &Path, args: &[String]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error::io(program.to_path_buf(), e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Ffmpeg {
            code: output.status.code().unwrap_or(-1),
            stderr: tail_of(&stderr, 20),
            oom: looks_like_oom(&stderr),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Runs a tool to completion with progress reporting and cancellation.
pub fn run_tool(
    program: &Path,
    args: &[String],
    reporter: &Reporter,
    cancel: &AtomicBool,
    spec: &RunSpec<'_>,
) -> Result<RunOutcome> {
    let started = Instant::now();
    reporter.debug(
        Some(spec.stage),
        format!("exec: {}", format_command(program, args)),
    );

    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Error::io(program.to_path_buf(), e))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let (pump, stderr_thread) = match stderr {
        Some(stderr) => {
            let cfg = StderrSinkConfig {
                label: spec.label.to_string(),
                stage: spec.stage,
                sink: spec.stderr_sink.clone(),
                quiet: spec.stderr_quiet.clone(),
                log_stderr: spec.log_stderr,
                level: spec.stderr_level,
                tail_cap: spec.stderr_tail.max(1),
            };
            let (pump, handle) = StderrPump::start(stderr, cfg, reporter.clone());
            (Some(pump), Some(handle))
        }
        None => (None, None),
    };

    let progress_thread = stdout.map(|stdout| {
        let reporter = reporter.clone();
        let stage = spec.stage;
        let label = spec.label.to_string();
        let expected = spec.expected_duration;
        let total_frames = spec.total_frames;
        std::thread::Builder::new()
            .name(format!("progress-{label}"))
            .spawn(move || {
                let reader = BufReader::new(stdout);
                let mut parser = ProgressParser::new();
                let mut last = Instant::now() - PROGRESS_INTERVAL;
                for line in reader.lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(_) => break,
                    };
                    if let Some(tick) = parser.push_line(&line) {
                        if last.elapsed() >= PROGRESS_INTERVAL || tick.end {
                            last = Instant::now();
                            reporter.progress(build_progress(
                                stage,
                                &tick,
                                expected,
                                total_frames,
                            ));
                        }
                    }
                }
            })
            .expect("spawn progress pump")
    });

    let code = loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            // let the pumps drain so the log ends cleanly
            if let Some(handle) = progress_thread {
                let _ = handle.join();
            }
            if let Some(handle) = stderr_thread {
                let _ = handle.join();
            }
            return Err(Error::Cancelled);
        }
        match child.try_wait().map_err(Error::BareIo)? {
            Some(status) => {
                let code = status.code().unwrap_or(-1);
                if let Some(handle) = progress_thread {
                    let _ = handle.join();
                }
                if let Some(handle) = stderr_thread {
                    let _ = handle.join();
                }
                break code;
            }
            None => std::thread::sleep(Duration::from_millis(40)),
        }
    };

    let elapsed = started.elapsed();
    let (tail, oom) = match &pump {
        Some(pump) => (pump.tail_string(), pump.looks_like_oom()),
        None => (String::new(), false),
    };
    if code != 0 {
        return Err(Error::Ffmpeg {
            code,
            stderr: if tail.trim().is_empty() {
                format!("{} failed with no diagnostic output", spec.label)
            } else {
                tail
            },
            oom,
        });
    }
    Ok(RunOutcome {
        code,
        stderr_tail: tail,
        elapsed,
    })
}

fn build_progress(
    stage: Stage,
    tick: &crate::ffmpeg::progress::ProgressTick,
    expected: Option<Duration>,
    total_frames: Option<u64>,
) -> StageProgress {
    let out_time = tick.out_time();
    let fraction = match (out_time, expected) {
        (Some(done), Some(total)) if total.as_micros() > 0 => {
            Some(done.as_micros() as f32 / total.as_micros() as f32)
        }
        _ => match (tick.frame, total_frames) {
            (Some(frame), Some(total)) if total > 0 => Some(frame as f32 / total as f32),
            _ => None,
        },
    };
    let eta = match (out_time, expected, tick.speed) {
        (Some(done), Some(total), Some(speed)) if speed > 0.01 && total > done => {
            let remaining = total.saturating_sub(done).as_secs_f64();
            Some(Duration::from_secs_f64(remaining / speed))
        }
        _ => None,
    };
    let mut detail = String::new();
    if let Some(frame) = tick.frame {
        detail.push_str(&format!("frame {frame}"));
    }
    if let Some(fps) = tick.fps {
        detail.push_str(&format!(" · {fps:.1} fps"));
    }
    if let Some(speed) = tick.speed {
        detail.push_str(&format!(" · {speed:.2}x"));
    }
    if let (Some(drop), true) = (tick.drop_frames, tick.drop_frames.unwrap_or(0) > 0) {
        detail.push_str(&format!(" · {drop} dropped"));
    }
    StageProgress {
        stage,
        fraction,
        detail: if detail.is_empty() {
            "working".to_string()
        } else {
            detail
        },
        frames: tick.frame,
        fps: tick.fps,
        speed: tick.speed,
        out_time,
        eta,
    }
}

/// A child whose stdout/stdin the caller drives (frame reads, PCM writes).
pub struct StreamingChild {
    child: Child,
    pub stdout: Option<BufReader<ChildStdout>>,
    pub stdin: Option<ChildStdin>,
    pump: Option<StderrPump>,
    stderr_thread: Option<JoinHandle<()>>,
    label: String,
    stage: Stage,
    started: Instant,
}

impl StreamingChild {
    pub fn spawn(
        program: &Path,
        args: &[String],
        reporter: &Reporter,
        spec: &RunSpec<'_>,
        pipe_stdin: bool,
    ) -> Result<Self> {
        reporter.debug(
            Some(spec.stage),
            format!("exec (stream): {}", format_command(program, args)),
        );
        let mut child = Command::new(program)
            .args(args)
            .stdin(if pipe_stdin {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::io(program.to_path_buf(), e))?;

        let stdout = child.stdout.take().map(BufReader::new);
        let stdin = child.stdin.take();
        let (pump, stderr_thread) = match child.stderr.take() {
            Some(stderr) => {
                let cfg = StderrSinkConfig {
                    label: spec.label.to_string(),
                    stage: spec.stage,
                    sink: spec.stderr_sink.clone(),
                    quiet: spec.stderr_quiet.clone(),
                    log_stderr: spec.log_stderr,
                    level: spec.stderr_level,
                    tail_cap: spec.stderr_tail.max(1),
                };
                let (pump, handle) = StderrPump::start(stderr, cfg, reporter.clone());
                (Some(pump), Some(handle))
            }
            None => (None, None),
        };

        Ok(StreamingChild {
            child,
            stdout,
            stdin,
            pump,
            stderr_thread,
            label: spec.label.to_string(),
            stage: spec.stage,
            started: Instant::now(),
        })
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Returns `Err(Cancelled)` when the caller should stop reading.
    pub fn check_cancel(&self, cancel: &AtomicBool) -> Result<()> {
        if cancel.load(Ordering::Relaxed) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
    }

    pub fn wait(mut self) -> Result<RunOutcome> {
        let status = self.child.wait().map_err(Error::BareIo)?;
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
        let elapsed = self.started.elapsed();
        let tail = self
            .pump
            .as_ref()
            .map(|p| p.tail_string())
            .unwrap_or_default();
        let oom = self.pump.as_ref().map(|p| p.looks_like_oom()).unwrap_or(false);
        let code = status.code().unwrap_or(-1);
        if code != 0 {
            return Err(Error::Ffmpeg {
                code,
                stderr: if tail.trim().is_empty() {
                    format!("{} failed with no diagnostic output", self.label)
                } else {
                    tail
                },
                oom,
            });
        }
        Ok(RunOutcome {
            code,
            stderr_tail: tail,
            elapsed,
        })
    }

    /// Kills the child and reaps it; used on the error paths.
    pub fn abort(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Drains a reader into `f`, checking for cancellation between lines.
pub fn read_lines<F: FnMut(&str)>(
    reader: &mut BufReader<impl Read>,
    cancel: &AtomicBool,
    mut f: F,
) -> Result<()> {
    let mut buf = String::new();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        buf.clear();
        let n = reader.read_line(&mut buf).map_err(Error::BareIo)?;
        if n == 0 {
            return Ok(());
        }
        f(buf.trim_end_matches(['\r', '\n']));
    }
}

/// Reads exactly `buf.len()` bytes unless the stream ends first.
pub fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::BareIo(e)),
        }
    }
    Ok(filled)
}

pub fn write_all<W: Write>(writer: &mut W, buf: &[u8]) -> Result<()> {
    writer.write_all(buf).map_err(Error::BareIo)
}

/// Human-readable command line for the log (`program "a b" c`).
pub fn format_command(program: &Path, args: &[String]) -> String {
    let mut out = String::new();
    out.push_str(&quote(&program.display().to_string()));
    for arg in args {
        out.push(' ');
        out.push_str(&quote(arg));
    }
    out
}

fn quote(s: &str) -> String {
    if s.is_empty() {
        return "\"\"".to_string();
    }
    if s.chars().any(|c| c.is_whitespace() || c == '"') {
        format!("\"{}\"", s.replace('"', "\\\""))
    } else {
        s.to_string()
    }
}

fn tail_of(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().collect();
    let start = all.len().saturating_sub(lines);
    all[start..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_formatting_quotes_only_when_needed() {
        let cmd = format_command(
            Path::new("ffmpeg"),
            &[
                "-vf".into(),
                "scale=1920:1080".into(),
                "-metadata".into(),
                "title=My Film".into(),
            ],
        );
        assert_eq!(
            cmd,
            "ffmpeg -vf scale=1920:1080 -metadata \"title=My Film\""
        );
    }

    #[test]
    fn progress_fraction_comes_from_out_time_when_known() {
        let mut tick = crate::ffmpeg::progress::ProgressTick::default();
        tick.out_time_us = Some(30_000_000);
        tick.frame = Some(720);
        tick.speed = Some(2.0);
        let p = build_progress(
            Stage::Encode,
            &tick,
            Some(Duration::from_secs(60)),
            Some(1440),
        );
        assert_eq!(p.fraction, Some(0.5));
        assert_eq!(p.eta, Some(Duration::from_secs(15)));
        assert!(p.detail.contains("720"));
    }

    #[test]
    fn missing_duration_falls_back_to_frame_count() {
        let mut tick = crate::ffmpeg::progress::ProgressTick::default();
        tick.frame = Some(50);
        let p = build_progress(Stage::Encode, &tick, None, Some(100));
        assert_eq!(p.fraction, Some(0.5));
        assert_eq!(p.eta, None);
    }

    #[test]
    fn unknown_progress_is_none_not_zero() {
        let tick = crate::ffmpeg::progress::ProgressTick::default();
        let p = build_progress(Stage::Encode, &tick, None, None);
        assert_eq!(p.fraction, None);
        assert_eq!(p.detail, "working");
    }

    #[test]
    fn tail_keeps_the_last_lines() {
        let text = "a\nb\nc\nd\ne";
        assert_eq!(tail_of(text, 2), "d\ne");
        assert_eq!(tail_of(text, 99), "a\nb\nc\nd\ne");
    }

    #[test]
    fn quiet_filters_match_case_insensitively() {
        let config = StderrSinkConfig {
            label: "idet".into(),
            stage: Stage::Scenes,
            sink: None,
            quiet: RunSpec::new(Stage::Scenes, "idet")
                .quiet("Parsed_idet")
                .stderr_quiet,
            log_stderr: true,
            level: Level::Debug,
            tail_cap: 4,
        };
        assert!(config.is_quiet("[Parsed_idet_0 @ 0x1] Repeated Fields: Neither: 10"));
        assert!(!config.is_quiet("frame= 100 fps=0.0"));
    }
}
