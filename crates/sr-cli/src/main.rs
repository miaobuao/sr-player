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
use sr_core::media::classify::{classify, ClassifyOptions};
use sr_core::media::probe;
use sr_core::media::scene::{detect_scenes, SceneDecode, SceneOptions};
use sr_core::pipeline::native::ChunkEncoding;
use sr_core::pipeline::plan::PlanRequest;
use sr_core::pipeline::profile::RestorationProfile;
use sr_core::pipeline::runner::{PipelineRunner, RunnerOptions};
use sr_core::state::Store;
use sr_core::{EventBus, VERSION};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

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
        /// off | rife
        ///
        /// `rife` is the only interpolator, and it is not built yet: asking for it
        /// fails before any pixel moves rather than quietly running something else.
        #[arg(long)]
        interpolate: Option<String>,
        /// How the native executor stores its per-chunk checkpoints:
        /// ffv1 (lossless intermediates, one final encode) or direct (encode each
        /// chunk with the final encoder and concatenate with a stream copy).
        #[arg(long)]
        chunk_encoding: Option<String>,
        /// Skip the audio remaster entirely (tracks are copied).
        #[arg(long)]
        no_audio: bool,
        /// Encoder quality target (CQ/CRF).
        #[arg(long)]
        quality: Option<i32>,
        /// Enable re-grain. Any positive value switches the per-shot estimator
        /// on, and the strength actually applied is then *measured* from each
        /// shot rather than taken from this number. The number is used directly
        /// only on the single-pass path, which has one filter chain for the whole
        /// film and therefore cannot vary it.
        ///
        /// This is also what selects the chunked executor, so it is what makes
        /// per-chunk resume reachable from the command line.
        #[arg(long)]
        regrain: Option<f32>,
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
    /// Show the AI runtime, GPU telemetry and hardware decode this machine offers.
    Devices,
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
        Command::Devices => {
            let ff = Arc::new(discover_ffmpeg(cli)?);
            println!("sr-core {VERSION}");
            println!("{}", ff.version);
            println!("GPU: {}", gpu::describe(&gpu::probe()));
            println!("{}", sr_core::pipeline::ai_runtime_line());
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
            let cadence = sr_core::pipeline::plan::cadence_for(&manifest, &temporal)?;
            let scenes = detect_scenes(
                &ff,
                &manifest,
                &reporter,
                &cancel,
                &SceneOptions::default(),
                SceneDecode {
                    pre_chain: cadence.chain_str(),
                    fps: cadence.fps,
                    reason: "decoded cadence, the same chain the encoder uses",
                },
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
            chunk_encoding,
            no_audio,
            quality,
            regrain,
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
                let parsed = sr_core::pipeline::profile::InterpolationMethod::parse(method)
                    .ok_or_else(|| {
                        sr_core::Error::Other(format!("unknown interpolation `{method}`"))
                    })?;
                profile.interpolation.method = parsed;
                // Asking for an interpolator while the multiplier stays at 1 is a
                // request for nothing, and the planner would read it as "off".
                // The product's film mode is 2x, so that is what a bare
                // `--interpolate rife` means.
                if parsed != sr_core::pipeline::profile::InterpolationMethod::Off
                    && profile.interpolation.multiplier <= 1
                {
                    profile.interpolation.multiplier = 2;
                }
            }
            if *no_audio {
                profile.audio.enabled = false;
            }
            if let Some(quality) = quality {
                profile.output.quality = *quality;
            }
            if let Some(strength) = regrain {
                if !strength.is_finite() || *strength < 0.0 {
                    return Err(sr_core::Error::Other(format!(
                        "re-grain strength `{strength}` must be a finite value >= 0"
                    )));
                }
                profile.output.regrain_strength = *strength;
            }

            let ff = Arc::new(discover_ffmpeg(cli)?);
            let store = Arc::new(Store::open(&state_db)?);
            let bus = EventBus::new();
            attach_printer(&bus, cli.verbose);

            let job_id = new_job_id(input, &output);
            let scratch = output
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(".sr-scratch");

            let cancel = Arc::new(AtomicBool::new(false));
            let runner = PipelineRunner::new(
                ff,
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
                chunk_encoding: match chunk_encoding {
                    Some(text) => ChunkEncoding::parse(text).ok_or_else(|| {
                        sr_core::Error::Other(format!(
                            "unknown chunk encoding `{text}`: use ffv1 or direct"
                        ))
                    })?,
                    None => ChunkEncoding::LosslessIntermediate,
                },
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

/// A job id derived from what the job *is*, not from when it started.
///
/// It used to be `cli-<unix seconds>`, which had two consequences and both were
/// wrong:
///
/// * **resume never happened from the command line.** The chunks table is keyed by
///   job id, so a second run of the same file looked for a job that had never
///   existed and redid everything. The resume machinery was unreachable from the
///   only entry point an unattended user has;
/// * **two files started in the same second shared a job id.** A batch loop over a
///   directory does exactly that, and the second run would find the first run's
///   committed chunks — same id, same scratch directory — and happily reuse
///   artifacts built from a different film. Nothing validated that a chunk row
///   belonged to the input being processed.
///
/// Hashing the input and output paths fixes both: the same conversion resumes, and
/// two different conversions cannot collide. FNV-1a rather than `DefaultHasher`,
/// because a job id has to mean the same thing in a later release.
fn new_job_id(input: &Path, output: &Path) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(input.to_string_lossy().as_bytes());
    feed(&[0]);
    feed(output.to_string_lossy().as_bytes());

    // The stem as well, so `sr-cli jobs` and the logs show which film a row belongs
    // to without a lookup.
    let stem: String = input
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_default()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(32)
        .collect();
    if stem.is_empty() {
        format!("cli-{hash:016x}")
    } else {
        format!("cli-{stem}-{hash:016x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The id has to be a function of the job, not of the clock: that is what makes
    /// a second attempt resume instead of starting over.
    #[test]
    fn a_job_id_is_the_same_for_the_same_conversion() {
        let one = new_job_id(Path::new("E:/films/a.mkv"), Path::new("E:/out/a.mkv"));
        let two = new_job_id(Path::new("E:/films/a.mkv"), Path::new("E:/out/a.mkv"));
        assert_eq!(one, two, "the same conversion must resume, not restart");
        assert!(one.starts_with("cli-a-"), "the id should name the film: {one}");
    }

    /// And it has to differ for different work, including two files in the same
    /// second — which is what a batch loop does, and what the old
    /// `cli-<seconds>` scheme could not tell apart.
    #[test]
    fn different_conversions_get_different_ids() {
        let first = new_job_id(Path::new("E:/films/a.mkv"), Path::new("E:/out/a.mkv"));
        let second = new_job_id(Path::new("E:/films/b.mkv"), Path::new("E:/out/b.mkv"));
        assert_ne!(first, second);
        // The same input to a different output is different work as well: the chunks
        // are the same but the film being built is not.
        let third = new_job_id(Path::new("E:/films/a.mkv"), Path::new("E:/out/a-1080.mkv"));
        assert_ne!(first, third);
    }
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
