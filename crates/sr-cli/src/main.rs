//! `sr-cli` — the headless driver for the sr-core media engine.
//!
//! Everything the GUI does, minus the window: probe a file, analyse it, run the
//! restoration pipeline, and inspect the job store. This is also the surface the
//! unattended/scripted use case actually wants.

use clap::{Parser, Subcommand};
use sr_core::audio::{analyze, AnalyzeOptions};
use sr_core::events::{Event, Level, StageStatus};
use sr_core::ffmpeg::Ffmpeg;
use sr_core::gpu;
use sr_core::infer::EngineRegistry;
use sr_core::media::classify::{classify, ClassifyOptions};
use sr_core::media::probe;
use sr_core::media::scene::{detect_scenes, SceneOptions};
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::RestorationProfile;
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::{EventBus, VERSION};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(
    name = "sr-cli",
    version,
    about = "Unattended film restoration: probe, analyse and convert"
)]
struct Cli {
    /// Job database (defaults to the per-user state directory).
    #[arg(long, global = true)]
    state_db: Option<PathBuf>,

    /// Print every log line, including debug detail.
    #[arg(long, short, global = true)]
    verbose: bool,

    /// Override the FFmpeg binary.
    #[arg(long, global = true)]
    ffmpeg: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show what a file actually contains.
    Probe {
        file: PathBuf,
        /// Emit the raw manifest as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run the analysis stages without encoding anything.
    Analyze {
        file: PathBuf,
        /// Emit the analysis reports as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Convert a file: the full pipeline.
    Convert {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// safe-16gb | deterministic | preview
        #[arg(long)]
        profile: Option<String>,
        /// Analyse and plan, but do not touch a pixel.
        #[arg(long)]
        dry_run: bool,
        /// Ignore committed checkpoints and start over.
        #[arg(long)]
        no_resume: bool,
        /// Keep the intermediate WAV / scratch files.
        #[arg(long)]
        keep_intermediates: bool,
        /// off | duplicate | minterpolate | plugin
        #[arg(long)]
        interpolate: Option<String>,
        /// Skip the audio remaster entirely (tracks are copied).
        #[arg(long)]
        no_audio: bool,
        /// Encoder quality target (CQ/CRF).
        #[arg(long)]
        quality: Option<i32>,
    },
    /// List recent jobs from the state database.
    Jobs {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Print a job's log.
    Log {
        job_id: String,
        #[arg(long, default_value_t = 200)]
        limit: usize,
    },
    /// Show the inference engines and GPU telemetry this machine offers.
    Engines,
    /// List the built-in profiles.
    Profiles,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let level = if cli.verbose { "debug" } else { "info" };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level)),
        )
        .with_target(false)
        .try_init();

    match run(&cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli) -> sr_core::Result<ExitCode> {
    let state_db = cli
        .state_db
        .clone()
        .unwrap_or_else(Store::default_path);

    match &cli.command {
        Command::Profiles => {
            for profile in RestorationProfile::builtin() {
                println!("{}\n  {}\n", profile.name, profile.description);
                for (key, value) in profile.summary_rows() {
                    println!("  {key:<18} {value}");
                }
                println!();
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Engines => {
            let ff = Arc::new(discover_ffmpeg(cli)?);
            let engines = EngineRegistry::probe(Arc::clone(&ff));
            println!("sr-core {VERSION}");
            println!("{}", ff.version);
            println!("GPU: {}", gpu::describe(&gpu::probe()));
            println!("{}\n", engines.summary());
            for (name, value) in engines.report_rows() {
                println!("{name:<22} {value}");
            }
            println!("\nhardware decode: {}", ff.hwaccels().join(", "));
            Ok(ExitCode::SUCCESS)
        }
        Command::Jobs { limit } => {
            let store = Store::open(&state_db)?;
            let jobs = store.list_jobs(*limit)?;
            if jobs.is_empty() {
                println!("no jobs recorded in {}", state_db.display());
                return Ok(ExitCode::SUCCESS);
            }
            println!(
                "{:<28} {:<10} {:<9} {}",
                "job", "state", "elapsed", "input"
            );
            for job in jobs {
                let elapsed = job
                    .elapsed_ms
                    .map(|ms| format!("{:.1}s", ms as f64 / 1000.0))
                    .unwrap_or_else(|| "-".into());
                println!(
                    "{:<28} {:<10} {:<9} {}",
                    job.id,
                    job.state.as_str(),
                    elapsed,
                    job.input.display()
                );
                if let Some(message) = job.message {
                    println!("{:>49} {}", "", message);
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Log { job_id, limit } => {
            let store = Store::open(&state_db)?;
            for record in store.logs(job_id, *limit, Level::Trace)? {
                println!("{}", record.format_line());
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Probe { file, json } => {
            let ff = discover_ffmpeg(cli)?;
            let manifest = probe(&ff, file)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            } else {
                println!("{}", file.display());
                for (key, value) in manifest.summary_lines() {
                    println!("  {key:<22} {value}");
                }
                let problems = manifest.validate();
                if !problems.is_empty() {
                    println!("\nnotes:");
                    for problem in problems {
                        println!("  - {problem}");
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Analyze { file, json } => {
            let ff = Arc::new(discover_ffmpeg(cli)?);
            let bus = EventBus::new();
            attach_printer(&bus, cli.verbose);
            let reporter = sr_core::Reporter::new(bus);
            let cancel = AtomicBool::new(false);

            let manifest = probe(&ff, file)?;
            let temporal = classify(
                &ff,
                &manifest,
                &reporter,
                &cancel,
                &ClassifyOptions::default(),
            )?;
            let scenes = detect_scenes(
                &ff,
                &manifest,
                &reporter,
                &cancel,
                &SceneOptions::default(),
            )?;
            let audio = if manifest.audio.is_empty() {
                None
            } else {
                Some(analyze(
                    &ff,
                    &manifest,
                    &AnalyzeOptions::default(),
                    &reporter,
                    &cancel,
                )?)
            };

            if *json {
                let report = serde_json::json!({
                    "manifest": manifest,
                    "temporal": temporal,
                    "scenes": scenes,
                    "audio": audio,
                });
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("cadence: {}", temporal.summary());
                for note in &temporal.notes {
                    println!("  - {note}");
                }
                println!("shots:   {}", scenes.summary());
                for note in &scenes.notes {
                    println!("  - {note}");
                }
                match &audio {
                    Some(audio) => {
                        println!("audio:   {}", audio.summary());
                        for note in &audio.notes {
                            println!("  - {note}");
                        }
                    }
                    None => println!("audio:   no audio stream"),
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Convert {
            input,
            output,
            profile,
            dry_run,
            no_resume,
            keep_intermediates,
            interpolate,
            no_audio,
            quality,
        } => {
            let output = match output {
                Some(path) => path.clone(),
                None => default_output_path(input),
            };
            let mut profile = match profile {
                Some(name) => RestorationProfile::by_name(name).ok_or_else(|| {
                    sr_core::Error::Other(format!("unknown profile `{name}`"))
                })?,
                None => RestorationProfile::safe_16gb(),
            };
            if let Some(method) = interpolate {
                profile.interpolation.method =
                    sr_core::pipeline::profile::InterpolationMethod::parse(method).ok_or_else(
                        || sr_core::Error::Other(format!("unknown interpolation `{method}`")),
                    )?;
            }
            if *no_audio {
                profile.audio.enabled = false;
            }
            if let Some(quality) = quality {
                profile.output.quality = *quality;
            }

            let ff = Arc::new(discover_ffmpeg(cli)?);
            let engines = Arc::new(EngineRegistry::probe(Arc::clone(&ff)));
            let store = Arc::new(Store::open(&state_db)?);
            let bus = EventBus::new();
            attach_printer(&bus, cli.verbose);

            let job_id = new_job_id("cli");
            let scratch = output
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(".sr-scratch");

            let cancel = Arc::new(AtomicBool::new(false));
            let runner = PipelineRunner::new(
                ff,
                engines,
                bus.clone(),
                Arc::clone(&store),
                Arc::clone(&cancel),
                scratch,
            );

            let request = PlanRequest {
                job_id: job_id.clone(),
                input: input.clone(),
                output: output.clone(),
                profile,
            };
            let options = RunnerOptions {
                resume: !*no_resume,
                keep_intermediates: *keep_intermediates,
                dry_run: *dry_run,
                ..Default::default()
            };

            let outcome = runner.run(request, &options)?;
            println!();
            if outcome.ok {
                println!(
                    "completed in {:.1}s → {}",
                    outcome.elapsed.as_secs_f64(),
                    outcome
                        .output
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                );
                Ok(ExitCode::SUCCESS)
            } else {
                println!("job {} failed: {}", outcome.job_id, outcome.message);
                Ok(ExitCode::FAILURE)
            }
        }
    }
}

fn discover_ffmpeg(cli: &Cli) -> sr_core::Result<Ffmpeg> {
    match &cli.ffmpeg {
        Some(path) => Ffmpeg::from_paths(
            path.clone(),
            path.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" }),
        ),
        None => Ffmpeg::discover(),
    }
}

fn default_output_path(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".into());
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{stem}.restored.mkv"))
}

fn new_job_id(prefix: &str) -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{prefix}-{seconds}")
}

/// Prints engine events to the terminal: logs always, progress on one line.
fn attach_printer(bus: &EventBus, verbose: bool) {
    let rx = bus.subscribe();
    std::thread::Builder::new()
        .name("cli-events".into())
        .spawn(move || {
            let mut last_progress = std::time::Instant::now() - Duration::from_secs(5);
            while let Ok(event) = rx.recv() {
                match event {
                    Event::Log(record) => {
                        if record.level < Level::Info && !verbose {
                            continue;
                        }
                        println!("{}", record.format_line());
                    }
                    Event::Stage { stage, status, note } => {
                        let badge = match status {
                            StageStatus::Running => "▶",
                            StageStatus::Done => "✓",
                            StageStatus::Skipped => "–",
                            StageStatus::Degraded => "▼",
                            StageStatus::Failed => "✗",
                            StageStatus::Pending => "·",
                        };
                        match note {
                            Some(note) => println!("{badge} {} — {note}", stage.label()),
                            None => println!("{badge} {}", stage.label()),
                        }
                    }
                    Event::Progress(progress) => {
                        // Throttle: one line per second unless the stage changed.
                        if last_progress.elapsed() < Duration::from_secs(1) {
                            continue;
                        }
                        last_progress = std::time::Instant::now();
                        let percent = progress
                            .fraction
                            .map(|f| format!("{:>5.1}%", f * 100.0))
                            .unwrap_or_else(|| "     ".into());
                        let eta = progress
                            .eta
                            .map(|e| format!(" eta {:.0}s", e.as_secs_f64()))
                            .unwrap_or_default();
                        println!(
                            "  {} [{}] {}{eta}",
                            percent,
                            progress.stage.id(),
                            progress.detail
                        );
                    }
                    Event::Plan(plan) => {
                        println!("plan: {}", plan.describe());
                    }
                    Event::Job { job_id, state } => {
                        println!("job {job_id}: {}", state.as_str());
                    }
                    Event::Finished(outcome) => {
                        println!(
                            "finished: ok={} in {:.1}s — {}",
                            outcome.ok,
                            outcome.elapsed.as_secs_f64(),
                            outcome.message
                        );
                        break;
                    }
                }
            }
        })
        .expect("spawn event printer");
}
