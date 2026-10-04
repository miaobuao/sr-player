//! The pipeline runner: the thing that actually calls the video processing
//! workflow.
//!
//! Responsibilities, in the order they matter for unattended operation:
//!
//! 1. **Checkpoint everything.** Each analysis stage commits its result to SQLite
//!    and is skipped on a resumed run.
//! 2. **Report continuously.** Stage transitions and FFmpeg progress go to the
//!    event bus; the UI only has to render them.
//! 3. **Degrade instead of exiting.** An out-of-memory failure walks the policy
//!    ladder and retries, rather than failing the job.
//! 4. **Publish atomically.** The output is written to a temporary name, fsynced
//!    and renamed only after a successful mux.
//! 5. **Verify.** A quality-control stage re-probes the result and compares it
//!    with the plan instead of assuming FFmpeg did what it was told.

use crate::audio::loudness::{analyze, measure_ebur128, AnalyzeOptions};
use crate::audio::remaster_to_wav;
use crate::error::{Error, Result};
use crate::events::{
    Event, EventBus, JobOutcome, JobState, Level, Reporter, Stage, StageStatus,
};
use crate::ffmpeg::{args, run_tool, Ffmpeg, RunSpec, SelectedVideoEncoder};
use crate::gpu;
use crate::infer::EngineRegistry;
use crate::media::classify::classify;
use crate::media::manifest::MediaManifest;
use crate::media::probe::probe;
use crate::media::scene::{detect_scenes, SceneDecode, SceneReport};
use crate::pipeline::native::ChunkEncoding;
use crate::pipeline::plan::{build_plan, cadence_for, AudioPlan, ConversionPlan, PlanRequest, VideoPlan};
use crate::pipeline::profile::LoudnessTarget;
use crate::state::{commit_file_atomic, ChunkRow, NewJob, ResumePoint, Store};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunnerOptions {
    pub resume: bool,
    pub keep_intermediates: bool,
    /// Stop after the plan is produced, without touching any pixels.
    pub dry_run: bool,
    pub max_degrade_retries: u32,
    /// How the native executor stores its per-chunk checkpoints.
    pub chunk_encoding: ChunkEncoding,
}

impl Default for RunnerOptions {
    fn default() -> Self {
        RunnerOptions {
            resume: true,
            keep_intermediates: false,
            dry_run: false,
            max_degrade_retries: 4,
            // FFV1 intermediates: one encode at the end, so the published file
            // cannot depend on where the chunk boundaries happened to fall.
            chunk_encoding: ChunkEncoding::LosslessIntermediate,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QcCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QcReport {
    pub output: PathBuf,
    pub checks: Vec<QcCheck>,
    pub passed: bool,
}

impl QcReport {
    pub fn failures(&self) -> Vec<&QcCheck> {
        self.checks.iter().filter(|c| !c.passed).collect()
    }

    pub fn summary(&self) -> String {
        let failed = self.failures().len();
        if failed == 0 {
            format!("{} checks passed", self.checks.len())
        } else {
            format!("{failed} of {} checks failed", self.checks.len())
        }
    }
}

pub struct PipelineRunner {
    ff: Arc<Ffmpeg>,
    engines: Arc<EngineRegistry>,
    bus: EventBus,
    store: Arc<Store>,
    cancel: Arc<AtomicBool>,
    scratch_root: PathBuf,
}

impl PipelineRunner {
    pub fn new(
        ff: Arc<Ffmpeg>,
        engines: Arc<EngineRegistry>,
        bus: EventBus,
        store: Arc<Store>,
        cancel: Arc<AtomicBool>,
        scratch_root: PathBuf,
    ) -> Self {
        PipelineRunner {
            ff,
            engines,
            bus,
            store,
            cancel,
            scratch_root,
        }
    }

    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn clear_cancel(&self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    fn load_or<T: DeserializeOwned>(&self, job_id: &str, stage: Stage) -> Option<T> {
        self.store
            .load_stage_result(job_id, stage)
            .ok()
            .flatten()
            .and_then(|json| serde_json::from_str(&json).ok())
    }

    fn commit<T: Serialize>(&self, job_id: &str, stage: Stage, value: &T) -> Result<()> {
        let json = serde_json::to_string(value)?;
        self.store.commit_stage_result(job_id, stage, &json)
    }

    /// Runs a stage, or restores its committed result on a resumed run.
    fn run_stage<T, F>(
        &self,
        reporter: &Reporter,
        resume: &ResumePoint,
        job_id: &str,
        stage: Stage,
        produce: F,
    ) -> Result<T>
    where
        T: Serialize + DeserializeOwned,
        F: FnOnce(&Reporter) -> Result<T>,
    {
        if resume.is_done(stage) {
            if let Some(value) = self.load_or::<T>(job_id, stage) {
                reporter.info(
                    Some(stage),
                    "checkpoint found: skipping this stage",
                );
                reporter.stage(stage, StageStatus::Done, Some("resumed".into()));
                return Ok(value);
            }
        }
        reporter.stage(stage, StageStatus::Running, None);
        match produce(reporter) {
            Ok(value) => {
                if let Err(err) = self.commit(job_id, stage, &value) {
                    reporter.warn(
                        Some(stage),
                        format!("could not checkpoint this stage: {err}"),
                    );
                }
                reporter.stage(stage, StageStatus::Done, None);
                Ok(value)
            }
            Err(err) => {
                reporter.stage(stage, StageStatus::Failed, Some(err.to_string()));
                Err(err)
            }
        }
    }

    /// The whole job.
    ///
    /// Contract: a call to `run` always produces **exactly one**
    /// [`Event::Finished`] and one terminal [`Event::Job`], including when the
    /// job cannot even be recorded in the database. A front end subscribes once
    /// and can never be left waiting for a job that has already died.
    pub fn run(&self, request: PlanRequest, options: &RunnerOptions) -> Result<JobOutcome> {
        let started = Instant::now();
        let job_id = request.job_id.clone();
        let reporter = Reporter::with_store(
            self.bus.clone(),
            Arc::clone(&self.store),
            job_id.clone(),
        );

        self.bus.emit(Event::Job {
            job_id: job_id.clone(),
            state: JobState::Running,
        });

        let result = self.run_job(&request, options, &reporter, started);
        let (state, message) = match &result {
            Ok(_) => (JobState::Completed, "completed".to_string()),
            Err(Error::Cancelled) => (
                JobState::Cancelled,
                "cancelled by the operator".to_string(),
            ),
            Err(err) => (JobState::Failed, err.to_string()),
        };
        let elapsed = started.elapsed();

        // Recording the outcome must never suppress the terminal event.
        if let Err(err) =
            self.store
                .finish_job(&job_id, state, Some(&message), elapsed.as_millis() as i64)
        {
            reporter.warn(None, format!("could not record the job outcome: {err}"));
        }
        self.bus.emit(Event::Job {
            job_id: job_id.clone(),
            state,
        });

        let outcome = match result {
            Ok(output_path) => {
                reporter.stage(Stage::Done, StageStatus::Done, None);
                JobOutcome {
                    job_id: job_id.clone(),
                    ok: true,
                    output: Some(output_path),
                    message: format!("completed in {}", format_duration(elapsed)),
                    elapsed,
                    degraded: Vec::new(),
                }
            }
            Err(err) => {
                let level = if matches!(err, Error::Cancelled) {
                    Level::Warn
                } else {
                    Level::Error
                };
                reporter.log(level, None, format!("job ended: {err}"));
                JobOutcome {
                    job_id: job_id.clone(),
                    ok: false,
                    output: None,
                    message: err.to_string(),
                    elapsed,
                    degraded: Vec::new(),
                }
            }
        };
        self.bus.emit(Event::Finished(outcome.clone()));
        Ok(outcome)
    }

    /// Setup plus the stage sequence. Everything that can fail early lives here
    /// so that [`Self::run`] has a single place to build the terminal event.
    fn run_job(
        &self,
        request: &PlanRequest,
        options: &RunnerOptions,
        reporter: &Reporter,
        started: Instant,
    ) -> Result<PathBuf> {
        let job_id = request.job_id.as_str();
        self.store.create_job(&NewJob {
            id: job_id.to_string(),
            input: request.input.clone(),
            output: Some(request.output.clone()),
            profile: request.profile.name.clone(),
        })?;
        self.store
            .set_job_state(job_id, JobState::Running, Some("starting"))?;

        reporter.info(None, format!("sr-core {} · {}", crate::VERSION, self.ff.version));
        reporter.info(None, format!("GPU: {}", gpu::describe(&gpu::probe())));
        reporter.info(None, self.engines.summary());
        reporter.info(
            None,
            format!(
                "input: {} → output: {}",
                request.input.display(),
                request.output.display()
            ),
        );
        reporter.info(None, format!("profile: {}", request.profile.description));

        let resume = if options.resume {
            let point = self.store.resume_point(job_id)?;
            if !point.completed.is_empty() {
                reporter.info(
                    None,
                    format!(
                        "resuming: {} stage(s) already committed",
                        point.completed.len()
                    ),
                );
            }
            point
        } else {
            ResumePoint::default()
        };

        self.run_inner(request, options, reporter, &resume, started)
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    fn run_inner(
        &self,
        request: &PlanRequest,
        options: &RunnerOptions,
        reporter: &Reporter,
        resume: &ResumePoint,
        started: Instant,
    ) -> Result<PathBuf> {
        let job_id = request.job_id.as_str();
        let profile = &request.profile;

        // ---- analysis -----------------------------------------------------
        let manifest: MediaManifest = self.run_stage(reporter, resume, job_id, Stage::Probe, |_| {
            probe(&self.ff, &request.input)
        })?;
        for problem in manifest.validate() {
            reporter.warn(Some(Stage::Probe), problem);
        }
        reporter.info(
            Some(Stage::Probe),
            format!(
                "{} video, {} audio, {} subtitle, {} attachment stream(s), {} chapter(s)",
                manifest.video.len(),
                manifest.audio.len(),
                manifest.subtitles.len(),
                manifest.attachments.len(),
                manifest.chapters.len()
            ),
        );

        let temporal = self.run_stage(reporter, resume, job_id, Stage::Temporal, |r| {
            classify(
                &self.ff,
                &manifest,
                r,
                &self.cancel,
                &profile.analysis.classify,
            )
        })?;

        // Shot boundaries are measured on the *decoded* stream — the same cadence
        // chain the encoder and the model will see — so `Shot::start_frame` means
        // "the Nth frame the model gets", not "the Nth frame in the file".
        let cadence = cadence_for(&manifest, &temporal)?;
        let scenes = self.run_stage(reporter, resume, job_id, Stage::Scenes, |r| {
            detect_scenes(
                &self.ff,
                &manifest,
                r,
                &self.cancel,
                &profile.analysis.scene,
                SceneDecode {
                    pre_chain: cadence.chain_str(),
                    fps: cadence.fps,
                    reason: if cadence.renumbers_frames {
                        "decoded cadence after inverse telecine"
                    } else {
                        "decoded cadence"
                    },
                },
            )
        })?;

        let audio = if profile.audio.enabled && !manifest.audio.is_empty() {
            Some(self.run_stage(reporter, resume, job_id, Stage::AudioAnalysis, |r| {
                analyze(
                    &self.ff,
                    &manifest,
                    &AnalyzeOptions {
                        stream_index: profile.audio.stream_index.min(manifest.audio.len() - 1),
                        sample_rate: profile.audio.sample_rate,
                        dialogue: profile.audio.dialogue.clone(),
                        rider: profile.audio.rider.clone(),
                    },
                    r,
                    &self.cancel,
                )
            })?)
        } else {
            if manifest.audio.is_empty() {
                reporter.warn(Some(Stage::AudioAnalysis), "the file has no audio stream");
            }
            reporter.stage(
                Stage::AudioAnalysis,
                StageStatus::Skipped,
                Some("audio processing disabled".into()),
            );
            None
        };

        // ---- plan ---------------------------------------------------------
        let plan = self.run_stage(reporter, resume, job_id, Stage::Plan, |_| {
            build_plan(
                &self.ff,
                &self.engines,
                &manifest,
                &temporal,
                &scenes,
                audio.as_ref(),
                request,
            )
        })?;
        if let Ok(json) = serde_json::to_string(&plan) {
            let _ = self.store.set_plan(job_id, &json);
        }
        reporter.info(Some(Stage::Plan), format!("plan: {}", plan.describe()));
        for note in &plan.notes {
            reporter.info(Some(Stage::Plan), note.clone());
        }
        for warning in &plan.warnings {
            reporter.warn(Some(Stage::Plan), warning.clone());
        }
        self.bus.emit(Event::Plan(Box::new(plan.clone())));

        if options.dry_run {
            reporter.info(None, "dry run: stopping before any pixel is touched");
            return Ok(request.output.clone());
        }
        self.check_cancel()?;

        // ---- stages inside the video work ----------------------------------
        //
        // This is where the executor is chosen for real. A plan that asks for
        // model restoration or model interpolation is executed by pushing frames
        // through a session; anything else is one FFmpeg pass. The log says which,
        // because those two are not the same product.
        self.announce_video_stages(reporter, &plan);

        // ---- audio remaster ------------------------------------------------
        let scratch = self.scratch_root.join(format!("job-{job_id}"));
        std::fs::create_dir_all(&scratch).map_err(|e| Error::io(&scratch, e))?;
        let remastered = if plan.audio.enabled && !manifest.audio.is_empty() {
            let wav = scratch.join("enhanced-audio.wav");
            let stats = match self.run_stage(reporter, resume, job_id, Stage::AudioProcess, |r| {
                remaster_to_wav(
                    &self.ff,
                    &manifest,
                    audio.as_ref().expect("analysis present"),
                    &profile.audio.rider,
                    &wav,
                    r,
                    &self.cancel,
                )
            }) {
                Ok(stats) => stats,
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(err) => {
                    // Losing the audio remaster must not lose the video work:
                    // fall back to copying the original tracks.
                    reporter.warn(
                        Some(Stage::AudioProcess),
                        format!("audio remaster failed ({err}); original tracks will be copied"),
                    );
                    reporter.stage(
                        Stage::AudioProcess,
                        StageStatus::Degraded,
                        Some(err.to_string()),
                    );
                    return self.encode(request, &plan, None, reporter, options, &scratch);
                }
            };
            self.store.commit_chunk(
                job_id,
                &ChunkRow {
                    stage: Stage::AudioProcess,
                    chunk_index: 0,
                    start_frame: None,
                    end_frame: None,
                    status: "committed".into(),
                    artifact: Some(stats.path.clone()),
                    updated_ms: 0,
                },
            )?;
            reporter.info(
                Some(Stage::AudioProcess),
                format!(
                    "enhanced track written: {:.0}s, {}{}",
                    stats.duration_seconds,
                    if stats.applied {
                        "dialogue rider applied"
                    } else {
                        "dynamics untouched"
                    },
                    if stats.max_gain_db != 0.0 {
                        format!(" ({:+.1} dB)", stats.max_gain_db)
                    } else {
                        String::new()
                    }
                ),
            );
            Some(stats.path)
        } else {
            None
        };

        // ---- encode + mux + qc --------------------------------------------
        let output = if plan.video.uses_model() {
            // A failure here is *not* silently replaced by a deterministic encode.
            // That fallback used to exist and it produced a file the plan's own QC
            // then rejected — "planned 48 fps, got 24" — which hides the real
            // reason. The model path resumes from its committed chunks, so the
            // cheap recovery is to run the job again, not to publish something
            // nobody asked for.
            match self.native_video(
                request,
                &plan,
                &scenes,
                remastered.as_deref(),
                reporter,
                options,
                &scratch,
            ) {
                Ok(output) => output,
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(err) => {
                    reporter.error(
                        Some(Stage::Encode),
                        format!(
                            "the native inference path failed: {err}\n  the chunks that finished \
                             are committed, so running this job again resumes from the last one; \
                             `--profile deterministic` runs the model-free path instead"
                        ),
                    );
                    reporter.stage(
                        Stage::Restore,
                        StageStatus::Failed,
                        Some(err.to_string()),
                    );
                    return Err(err);
                }
            }
        } else {
            self.encode(
                request,
                &plan,
                remastered.as_deref(),
                reporter,
                options,
                &scratch,
            )?
        };

        let qc = self.run_stage(reporter, resume, job_id, Stage::Qc, |r| {
            self.quality_check(&manifest, &plan, &output, r)
        })?;
        if !qc.passed {
            for check in qc.failures() {
                reporter.error(
                    Some(Stage::Qc),
                    format!("{}: {}", check.name, check.detail),
                );
            }
            return Err(Error::Stage {
                stage: Stage::Qc.id().into(),
                detail: format!("{} of {} checks failed", qc.failures().len(), qc.checks.len()),
            });
        }
        reporter.info(Some(Stage::Qc), qc.summary());

        if !options.keep_intermediates {
            if let Err(err) = std::fs::remove_dir_all(&scratch) {
                reporter.debug(
                    Some(Stage::Done),
                    format!("could not remove scratch directory: {err}"),
                );
            }
        }
        reporter.info(
            None,
            format!(
                "finished in {} → {}",
                format_duration(started.elapsed()),
                output.display()
            ),
        );
        Ok(output)
    }

    /// States, before any pixel moves, which stages the chosen executor really
    /// runs.
    ///
    /// The baseline engine folds restoration, interpolation and muxing into a
    /// single FFmpeg pass. The native executor does something categorically
    /// different: it decodes frames, pushes them through a model, and checkpoints
    /// each chunk. The ladder must reflect which one is about to happen, because
    /// "interpolation: done" means two very different things in those two cases.
    fn announce_video_stages(&self, reporter: &Reporter, plan: &ConversionPlan) {
        if !plan.video.uses_model() {
            if plan.video.upscale_factor > 1.01 {
                reporter.stage(
                    Stage::Restore,
                    StageStatus::Skipped,
                    Some(format!(
                        "no model backend: the {:.2}x upscale is a deterministic Lanczos resample \
                         folded into the encode, and invents no detail",
                        plan.video.upscale_factor
                    )),
                );
            } else {
                reporter.stage(
                    Stage::Restore,
                    StageStatus::Skipped,
                    Some("source resolution kept; no restoration engine requested".into()),
                );
            }
            if plan.video.interpolation.enabled {
                reporter.stage(
                    Stage::Interpolate,
                    StageStatus::Done,
                    Some(format!(
                        "{} folded into the encode; cut detection is FFmpeg's, not the engine's",
                        plan.video.interpolation.method.as_str()
                    )),
                );
            } else {
                reporter.stage(
                    Stage::Interpolate,
                    StageStatus::Skipped,
                    Some("interpolation disabled".into()),
                );
            }
            reporter.stage(
                Stage::Regrain,
                StageStatus::Skipped,
                Some(if plan.video.regrain_strength > 0.0 {
                    format!(
                        "re-grain strength {:.0} folded into the encode (FFmpeg noise, not a \
                         per-shot grain model)",
                        plan.video.regrain_strength
                    )
                } else {
                    "re-grain disabled (no per-shot grain estimator in this build)".to_string()
                }),
            );
            return;
        }

        let inference = plan.video.inference.as_ref().expect("native plan");
        if inference.describes("restore") {
            reporter.stage(
                Stage::Restore,
                StageStatus::Running,
                Some(format!(
                    "{} restores every frame through a model session at {}x{}, strength {:.2}",
                    inference.engine_id,
                    inference.width,
                    inference.height,
                    inference.restore_strength
                )),
            );
        } else {
            reporter.stage(
                Stage::Restore,
                StageStatus::Skipped,
                Some("the model handles interpolation only; no restoration was requested".into()),
            );
        }
        if inference.describes("interpolate") {
            reporter.stage(
                Stage::Interpolate,
                StageStatus::Running,
                Some(format!(
                    "{} synthesises {:.3} fps from {:.3} fps, one shot at a time: a frame pair \
                     that straddles a cut is never handed to the model",
                    inference.engine_id,
                    plan.video.interpolation.target_fps.to_f64(),
                    plan.video.interpolation.source_fps.to_f64()
                )),
            );
        } else {
            reporter.stage(
                Stage::Interpolate,
                StageStatus::Skipped,
                Some(format!(
                    "{} (no model interpolation selected)",
                    plan.video.interpolation.method.as_str()
                )),
            );
        }
        reporter.stage(
            Stage::Regrain,
            StageStatus::Skipped,
            Some(if plan.video.regrain_strength > 0.0 {
                format!(
                    "re-grain strength {:.0} applied by FFmpeg after the model (not a per-shot \
                     grain model)",
                    plan.video.regrain_strength
                )
            } else {
                "re-grain disabled (no per-shot grain estimator in this build)".to_string()
            }),
        );
    }

    fn encode(
        &self,
        request: &PlanRequest,
        plan: &ConversionPlan,
        remastered: Option<&Path>,
        reporter: &Reporter,
        options: &RunnerOptions,
        scratch: &Path,
    ) -> Result<PathBuf> {
        // Measure the enhanced track to calibrate the final gain.
        let loudness_chain = self.prepare_loudness_chain(remastered, plan, reporter)?;

        let temp_output = scratch.join(format!(
            "output.{}",
            crate::pipeline::plan::output_extension(&plan.output_settings)
        ));

        let mut working_set = plan.working_set.clone();
        let mut attempt = 0u32;
        let mut encoder_index = 0usize;
        let mut last_error: Option<Error> = None;
        reporter.stage(Stage::Encode, StageStatus::Running, None);
        loop {
            self.check_cancel()?;
            let encoder = match plan.video.encoder_chain.get(encoder_index) {
                Some(encoder) => encoder.clone(),
                None => {
                    return Err(last_error.unwrap_or_else(|| Error::Stage {
                        stage: Stage::Encode.id().into(),
                        detail: "no usable video encoder remained".into(),
                    }))
                }
            };
            let filter_chain = filter_chain_for(&plan.video, &encoder);
            let argv = build_encode_args(
                &self.ff,
                &request.input,
                remastered,
                &filter_chain,
                &encoder,
                &plan.audio,
                &plan.inventory,
                loudness_chain.as_deref(),
                &plan.output_settings,
                &temp_output,
            );
            let spec = RunSpec::new(Stage::Encode, "encode")
                .with_duration(Duration::from_secs_f64(
                    plan.video.source.duration_seconds.max(1.0),
                ))
                .with_frames(plan.video.estimated_frames.max(1));
            match run_tool(&self.ff.ffmpeg, &argv, reporter, &self.cancel, &spec) {
                Ok(outcome) => {
                    reporter.info(
                        Some(Stage::Encode),
                        format!(
                            "encode finished in {} with {}",
                            format_duration(outcome.elapsed),
                            encoder.name
                        ),
                    );
                    break;
                }
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(err) if err.is_oom() && attempt < options.max_degrade_retries => {
                    match working_set.degrade() {
                        Some(change) => {
                            attempt += 1;
                            reporter.warn(
                                Some(Stage::Encode),
                                format!(
                                    "out of memory: {change} (attempt {}/{})",
                                    attempt, options.max_degrade_retries
                                ),
                            );
                            // Emitted as a stage event as well as stored, so the
                            // UI's "degraded" badge reflects what actually
                            // happened rather than only the database knowing.
                            reporter.stage(
                                Stage::Encode,
                                StageStatus::Degraded,
                                Some(change.clone()),
                            );
                            let _ = std::fs::remove_file(&temp_output);
                            continue;
                        }
                        None => {
                            reporter.error(
                                Some(Stage::Encode),
                                "out of memory and the degrade ladder is exhausted",
                            );
                            return Err(err);
                        }
                    }
                }
                Err(err) => {
                    // An encoder that is advertised but cannot open (a vendor
                    // encoder on another vendor's GPU, a driver too old, a
                    // missing device) must cost a retry, not the job.
                    match plan.video.encoder_chain.get(encoder_index + 1) {
                        Some(next) => {
                            reporter.warn(
                                Some(Stage::Encode),
                                format!(
                                    "{} failed ({}); retrying with {}",
                                    encoder.name,
                                    first_line(&err.to_string()),
                                    next.name
                                ),
                            );
                            reporter.stage(
                                Stage::Encode,
                                StageStatus::Degraded,
                                Some(format!("fell back to {}", next.name)),
                            );
                            encoder_index += 1;
                            last_error = Some(err);
                            let _ = std::fs::remove_file(&temp_output);
                            continue;
                        }
                        None => return Err(err),
                    }
                }
            }
        }
        reporter.stage(Stage::Encode, StageStatus::Done, None);

        reporter.stage(
            Stage::Mux,
            StageStatus::Done,
            Some(format!(
                "muxed in the same pass: {} audio, {} subtitle, {} attachment stream(s), {} chapter(s)",
                if remastered.is_some() {
                    plan.inventory.audio_streams + 1
                } else {
                    plan.inventory.audio_streams
                },
                plan.inventory.subtitle_streams,
                plan.inventory.attachments,
                plan.inventory.chapters
            )),
        );

        commit_file_atomic(&temp_output, &request.output)?;
        Ok(request.output.clone())
    }

    /// Measures the enhanced track and turns the measurement into the final gain
    /// chain. Shared by both video executors, so the audio result cannot depend on
    /// which one ran.
    fn prepare_loudness_chain(
        &self,
        remastered: Option<&Path>,
        plan: &ConversionPlan,
        reporter: &Reporter,
    ) -> Result<Option<String>> {
        let Some(wav) = remastered else {
            return Ok(None);
        };
        match measure_wav_loudness(&self.ff, wav, reporter, &self.cancel) {
            Ok(report) => match report.integrated_lufs {
                Some(measured) => {
                    let (chain, gain_db, limiting) =
                        build_gain_chain(measured, report.true_peak_dbtp, &plan.audio.loudness);
                    reporter.info(
                        Some(Stage::AudioProcess),
                        format!(
                            "final loudness: {} → gain {:+.2} dB{}",
                            report.summary(),
                            gain_db,
                            if limiting {
                                " (true-peak limiter engaged: dynamics touched)"
                            } else {
                                " (pure gain: dynamics preserved)"
                            }
                        ),
                    );
                    Ok(Some(chain))
                }
                None => {
                    reporter.warn(
                        Some(Stage::AudioProcess),
                        "the enhanced track has no measurable loudness; encoding without normalisation",
                    );
                    Ok(None)
                }
            },
            Err(Error::Cancelled) => Err(Error::Cancelled),
            Err(err) => {
                reporter.warn(
                    Some(Stage::AudioProcess),
                    format!("loudness measurement failed ({err}); encoding without normalisation"),
                );
                Ok(None)
            }
        }
    }

    /// The native path: frames go through a model session, chunk by chunk.
    #[allow(clippy::too_many_arguments)]
    fn native_video(
        &self,
        request: &PlanRequest,
        plan: &ConversionPlan,
        scenes: &SceneReport,
        remastered: Option<&Path>,
        reporter: &Reporter,
        options: &RunnerOptions,
        scratch: &Path,
    ) -> Result<PathBuf> {
        let loudness_chain = self.prepare_loudness_chain(remastered, plan, reporter)?;
        let executor = crate::pipeline::native::NativeExecutor::new(
            &self.ff,
            &self.engines,
            &self.store,
            &self.cancel,
        );
        reporter.stage(Stage::Encode, StageStatus::Running, None);
        let context = crate::pipeline::native::NativeContext {
            job_id: request.job_id.as_str(),
            request,
            plan,
            scenes,
            reporter,
            remastered,
            loudness_chain,
            working: plan.working_set.clone(),
            chunk_encoding: options.chunk_encoding,
            max_degrade_retries: options.max_degrade_retries,
            workdir: scratch,
        };
        let outcome = executor.run(&context)?;
        for message in &outcome.degradations {
            reporter.warn(Some(Stage::Encode), message.clone());
        }
        reporter.stage(Stage::Encode, StageStatus::Done, None);
        reporter.info(
            Some(Stage::Encode),
            format!(
                "{} chunk(s) written, {} reused from a previous run, {} model call(s), \
                 {} output frame(s) from {} input frame(s)",
                outcome.chunks_written,
                outcome.chunks_resumed,
                outcome.model_calls,
                outcome.output_frames,
                outcome.input_frames
            ),
        );
        reporter.stage(
            Stage::Mux,
            StageStatus::Done,
            Some(format!(
                "chunks concatenated with `-c:v {}` and muxed with {} audio, {} subtitle, \
                 {} attachment stream(s), {} chapter(s)",
                if outcome.chunk_encoding
                    == crate::pipeline::native::ChunkEncoding::DirectFinalCodec
                {
                    "copy"
                } else {
                    plan.video.encoder.name.as_str()
                },
                if remastered.is_some() {
                    plan.inventory.audio_streams + 1
                } else {
                    plan.inventory.audio_streams
                },
                plan.inventory.subtitle_streams,
                plan.inventory.attachments,
                plan.inventory.chapters
            )),
        );
        Ok(outcome.output)
    }

    fn quality_check(
        &self,
        source: &MediaManifest,
        plan: &ConversionPlan,
        output: &Path,
        reporter: &Reporter,
    ) -> Result<QcReport> {
        let mut checks = Vec::new();
        let out_manifest = probe(&self.ff, output)?;
        let duration_tolerance = 0.05f64.max(plan.video.source.duration_seconds * 0.001);

        checks.push(QcCheck {
            name: "output exists and probes".into(),
            passed: out_manifest.primary_video().is_some(),
            detail: out_manifest
                .primary_video()
                .map(|v| v.describe())
                .unwrap_or_else(|| "no video stream".into()),
        });

        // Both sides measured as content: the source's plan duration already
        // excludes a leading offset, so the output must be measured the same way
        // or a correct transcode of a late-starting file looks short.
        let drift =
            (out_manifest.content_duration_seconds() - plan.video.source.duration_seconds).abs();
        checks.push(QcCheck {
            name: "duration drift".into(),
            passed: drift <= duration_tolerance.max(0.5),
            detail: format!(
                "source {:.3}s, output {:.3}s, drift {:.3}s (tolerance {:.3}s)",
                plan.video.source.duration_seconds,
                out_manifest.content_duration_seconds(),
                drift,
                duration_tolerance.max(0.5)
            ),
        });

        let out_video = out_manifest.primary_video();
        let expected_pixels = (plan.video.target_width, plan.video.target_height);
        let actual_pixels = out_video.and_then(|v| v.size());
        checks.push(QcCheck {
            name: "resolution matches the plan".into(),
            passed: actual_pixels
                .map(|(w, h)| (w as u32, h as u32) == expected_pixels)
                .unwrap_or(false),
            detail: format!(
                "planned {}x{}, got {}",
                expected_pixels.0,
                expected_pixels.1,
                actual_pixels
                    .map(|(w, h)| format!("{w}x{h}"))
                    .unwrap_or_else(|| "?".into())
            ),
        });

        let expected_audio = if plan.audio.enabled && !source.audio.is_empty() {
            if plan.audio.keep_original {
                source.audio.len() + 1
            } else {
                1
            }
        } else {
            source.audio.len()
        };
        checks.push(QcCheck {
            name: "audio tracks preserved".into(),
            passed: out_manifest.audio.len() >= expected_audio,
            detail: format!(
                "expected at least {expected_audio}, found {}",
                out_manifest.audio.len()
            ),
        });

        if plan.audio.keep_original {
            checks.push(QcCheck {
                name: "original audio present".into(),
                passed: source
                    .audio
                    .iter()
                    .all(|a| out_manifest.audio.iter().any(|b| b.base.codec_name == a.base.codec_name)),
                detail: format!(
                    "source codecs {:?}",
                    source
                        .audio
                        .iter()
                        .filter_map(|a| a.base.codec_name.clone())
                        .collect::<Vec<_>>()
                ),
            });
        }

        checks.push(QcCheck {
            name: "subtitles preserved".into(),
            passed: out_manifest.subtitles.len() >= source.subtitles.len(),
            detail: format!(
                "source {}, output {}",
                source.subtitles.len(),
                out_manifest.subtitles.len()
            ),
        });

        checks.push(QcCheck {
            name: "chapters preserved".into(),
            passed: out_manifest.chapters.len() >= source.chapters.len(),
            detail: format!(
                "source {}, output {}",
                source.chapters.len(),
                out_manifest.chapters.len()
            ),
        });

        checks.push(QcCheck {
            name: "attachments preserved".into(),
            passed: out_manifest.attachments.len() >= source.attachments.len(),
            detail: format!(
                "source {}, output {}",
                source.attachments.len(),
                out_manifest.attachments.len()
            ),
        });

        if plan.video.interpolation.enabled {
            let planned = plan.video.interpolation.target_fps.to_f64();
            let actual = out_video.and_then(|v| v.fps()).map(|f| f.to_f64()).unwrap_or(0.0);
            let tolerance = planned * 0.01;
            checks.push(QcCheck {
                name: "frame rate matches the plan".into(),
                passed: (actual - planned).abs() <= tolerance.max(0.05),
                detail: format!("planned {planned:.3} fps, got {actual:.3} fps"),
            });
        }

        // True peak on the encoded enhanced track: the limiter must have done
        // its job, and this is the only way to know.
        if plan.audio.enabled && !out_manifest.audio.is_empty() {
            match measure_ebur128(&self.ff, &out_manifest, 0, reporter, &self.cancel) {
                Ok(report) => {
                    let peak_ok = report
                        .true_peak_dbtp
                        .map(|tp| tp <= plan.audio.loudness.true_peak_dbtp as f64 + 0.2)
                        .unwrap_or(true);
                    checks.push(QcCheck {
                        name: "true peak within target".into(),
                        passed: peak_ok,
                        detail: format!(
                            "{} (target {} dBTP)",
                            report
                                .true_peak_dbtp
                                .map(|v| format!("{v:.2} dBTP"))
                                .unwrap_or_else(|| "not measured".into()),
                            plan.audio.loudness.true_peak_dbtp
                        ),
                    });
                    let loudness_ok = report
                        .integrated_lufs
                        .map(|i| (i - plan.audio.loudness.i_lufs as f64).abs() <= 1.0)
                        .unwrap_or(true);
                    checks.push(QcCheck {
                        name: "integrated loudness within 1 LU of target".into(),
                        passed: loudness_ok,
                        detail: format!(
                            "{} (target {:.1} LUFS)",
                            report
                                .integrated_lufs
                                .map(|v| format!("{v:.1} LUFS"))
                                .unwrap_or_else(|| "not measured".into()),
                            plan.audio.loudness.i_lufs
                        ),
                    });
                }
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(err) => {
                    checks.push(QcCheck {
                        name: "loudness verification".into(),
                        passed: true,
                        detail: format!("not measured: {err}"),
                    });
                }
            }
        }

        let passed = checks.iter().all(|c| c.passed);
        Ok(QcReport {
            output: output.to_path_buf(),
            checks,
            passed,
        })
    }
}

// ---- argument construction ------------------------------------------------

/// The filter chain changes with the encoder (VAAPI needs a device upload), so
/// it is rebuilt whenever the runner falls back to a different encoder.
fn filter_chain_for(video: &VideoPlan, encoder: &SelectedVideoEncoder) -> String {
    let mut copy = video.clone();
    copy.encoder = encoder.clone();
    copy.pix_fmt = encoder.pix_fmt.clone();
    crate::pipeline::plan::build_filter_chain(
        &copy,
        (video.source.square_width, video.source.square_height),
    )
}

/// First line only: FFmpeg failures carry a multi-line stderr tail.
fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").trim().to_string()
}

#[allow(clippy::too_many_arguments)]
fn build_encode_args(
    ff: &Ffmpeg,
    input: &Path,
    remastered: Option<&Path>,
    filter_chain: &str,
    encoder: &SelectedVideoEncoder,
    audio: &AudioPlan,
    inventory: &crate::media::manifest::PreservationInventory,
    loudness_chain: Option<&str>,
    settings: &crate::pipeline::profile::OutputSettings,
    output: &Path,
) -> Vec<String> {
    let mut argv = args(&["-hide_banner", "-nostdin", "-progress", "pipe:1", "-y"]);
    argv.extend(encoder.input_args.iter().cloned());
    argv.push("-i".into());
    argv.push(input.display().to_string());
    if let Some(wav) = remastered {
        argv.push("-i".into());
        argv.push(wav.display().to_string());
    }

    // Video first, then the enhanced track, then everything else.
    argv.extend(args(&["-map", "0:v:0"]));
    if remastered.is_some() {
        argv.extend(args(&["-map", "1:a:0"]));
    }
    if audio.keep_original && inventory.audio_streams > 0 {
        argv.extend(args(&["-map", "0:a?"]));
    }
    if settings.preserve_subtitles && inventory.subtitle_streams > 0 {
        argv.extend(args(&["-map", "0:s?"]));
    }
    if settings.preserve_attachments && inventory.attachments > 0 {
        argv.extend(args(&["-map", "0:t?"]));
    }
    argv.extend(args(&["-map_metadata", "0"]));
    if settings.preserve_chapters && inventory.chapters > 0 {
        argv.extend(args(&["-map_chapters", "0"]));
    }

    if !filter_chain.is_empty() {
        argv.extend(args(&["-vf", filter_chain]));
    }
    argv.extend(args(&["-c:v", &encoder.name]));
    argv.extend(encoder.quality_args.iter().cloned());
    argv.extend(args(&["-pix_fmt", &encoder.pix_fmt]));

    // Audio: copied by default, re-encoded only for the first track.
    if inventory.audio_streams > 0 {
        argv.extend(args(&["-c:a", "copy"]));
    }
    if remastered.is_some() {
        argv.extend(args(&["-c:a:0", &audio.encoder.name]));
        argv.extend(audio.encoder.args.iter().cloned());
        if let Some(chain) = loudness_chain {
            argv.extend(args(&["-filter:a:0", chain]));
        }
        argv.extend(args(&[
            "-metadata:s:a:0",
            "title=Dialogue enhanced (LDR adapted)",
        ]));
        argv.extend(args(&["-disposition:a:0", "default"]));
        // Every source track that follows must lose any inherited `default`
        // flag, or players would pick the untouched original.
        for index in 1..=inventory.audio_streams {
            argv.push(format!("-disposition:a:{index}"));
            argv.push("0".to_string());
        }
    }
    if inventory.subtitle_streams > 0 {
        argv.extend(args(&["-c:s", "copy"]));
    }
    if inventory.attachments > 0 {
        argv.extend(args(&["-c:t", "copy"]));
    }
    argv.extend(args(&["-max_muxing_queue_size", "4096"]));
    argv.push("-f".into());
    argv.push(settings.container.clone());
    argv.push(output.display().to_string());

    let _ = ff;
    argv
}

// ---- loudness normalisation (always the last audio step) ------------------

/// Measures the enhanced track with FFmpeg's `ebur128`.
///
/// The final gain is computed from `ebur128`'s numbers rather than from
/// `loudnorm=print_format=json`. That report is not trustworthy for every input
/// (a valid 5.1 float WAV has been observed to report `+29 LUFS / +685 dBTP`
/// while `ebur128` and `volumedetect` both read it as `-21.9 LUFS`, peak
/// `0 dBFS`). `ebur128` is the same meter the analysis stage uses and the one
/// the native BS.1770 implementation is validated against, so the pipeline has
/// exactly one source of truth for loudness.
pub fn measure_wav_loudness(
    ff: &Ffmpeg,
    wav: &Path,
    reporter: &Reporter,
    cancel: &AtomicBool,
) -> Result<crate::audio::Ebur128Report> {
    let manifest = probe(ff, wav)?;
    let mut report = measure_ebur128(ff, &manifest, 0, reporter, cancel)?;
    // Meters can report physically impossible values on unusual inputs (an
    // `ebur128` true peak of +597 dBTP has been observed on a valid float WAV
    // that `volumedetect` reads as 0 dBFS peak). An implausible reading is worse
    // than no reading: it would engage the limiter for no reason, so it is
    // dropped and reported.
    if let Some(peak) = report.true_peak_dbtp {
        if !(peak > -120.0 && peak < 6.0) {
            reporter.warn(
                Some(Stage::AudioProcess),
                format!(
                    "the loudness meter reported an impossible true peak ({peak:.1} dBTP); \
                     normalising the loudness without a peak constraint"
                ),
            );
            report.true_peak_dbtp = None;
        }
    }
    Ok(report)
}

/// Builds the final chain: a calibrated gain, then a true-peak limiter.
///
/// Returns `(filter_chain, gain_db, limiting_required)`.
pub fn build_gain_chain(
    measured_i_lufs: f64,
    measured_tp_dbtp: Option<f64>,
    target: &LoudnessTarget,
) -> (String, f64, bool) {
    let gain_db = target.i_lufs as f64 - measured_i_lufs;
    let projected_peak = measured_tp_dbtp.map(|tp| tp + gain_db);
    let limiting = projected_peak
        .map(|peak| peak > target.true_peak_dbtp as f64)
        .unwrap_or(false);
    let limiter = format!(
        "alimiter=limit={:.4}:level=disabled:attack=5:release=50",
        10f64.powf(target.true_peak_dbtp as f64 / 20.0)
    );
    let chain = if gain_db.abs() < 0.05 {
        // Already on target: the limiter alone is the safety net.
        limiter
    } else {
        format!("volume={gain_db:.3}dB,{limiter}")
    };
    (chain, gain_db, limiting)
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let (h, m, s) = (secs / 3600, (secs / 60) % 60, secs % 60);
    if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::RemasterDecision;
    use crate::ffmpeg::{SelectedAudioEncoder, SelectedVideoEncoder, VideoCodec};
    use crate::pipeline::plan::{InterpolationPlan, VideoSourceInfo};
    use crate::time::Rational;

    fn sample_video_plan() -> VideoPlan {
        VideoPlan {
            source: VideoSourceInfo {
                width: 720,
                height: 480,
                square_width: 720,
                square_height: 480,
                sar: Rational::ONE,
                fps: Rational::new(24000, 1001).unwrap(),
                duration_seconds: 60.0,
                frames: 1440,
                pix_fmt: Some("yuv420p".into()),
                is_hdr: false,
            },
            temporal_mode: crate::media::classify::TemporalMode::Progressive,
            temporal_note: "progressive".into(),
            cadence: crate::pipeline::plan::CadencePlan::plan(
                crate::media::classify::TemporalMode::Progressive,
                Rational::new(24000, 1001).unwrap(),
                true,
            ),
            ivtc: false,
            deinterlace: None,
            target_width: 1440,
            target_height: 960,
            upscale_factor: 2.0,
            interpolation: InterpolationPlan {
                enabled: true,
                method: crate::pipeline::profile::InterpolationMethod::Duplicate,
                source_fps: Rational::new(24000, 1001).unwrap(),
                target_fps: Rational::new(48000, 1001).unwrap(),
                multiplier: 2,
                engine: None,
                scene_cuts_respected: true,
                note: "duplication".into(),
            },
            executor: crate::pipeline::plan::VideoExecutor::FfmpegSinglePass,
            inference: None,
            regrain_strength: 0.0,
            regrain_per_shot: Vec::new(),
            filter_chain: "scale=1440:960:flags=lanczos,framerate=fps=47.952048".into(),
            decode_chain: String::new(),
            encoder: SelectedVideoEncoder {
                name: "av1_nvenc".into(),
                codec: VideoCodec::Av1,
                vendor: Some("nvidia".into()),
                hardware: true,
                pix_fmt: "p010le".into(),
                quality_args: vec!["-cq".into(), "24".into()],
                input_args: vec![],
                requires_hwupload: false,
                note: String::new(),
            },
            encoder_chain: vec![SelectedVideoEncoder {
                name: "av1_nvenc".into(),
                codec: VideoCodec::Av1,
                vendor: Some("nvidia".into()),
                hardware: true,
                pix_fmt: "p010le".into(),
                quality_args: vec!["-cq".into(), "24".into()],
                input_args: vec![],
                requires_hwupload: false,
                note: String::new(),
            }],
            pix_fmt: "p010le".into(),
            estimated_frames: 2877,
        }
    }

    fn sample_audio_plan() -> AudioPlan {
        AudioPlan {
            enabled: true,
            summary: None,
            encoder: SelectedAudioEncoder {
                name: "flac".into(),
                lossless: true,
                args: vec![],
            },
            loudness: LoudnessTarget::default(),
            keep_original: true,
            note: String::new(),
        }
    }

    fn sample_inventory() -> crate::media::manifest::PreservationInventory {
        crate::media::manifest::PreservationInventory {
            video_streams: 1,
            audio_streams: 2,
            subtitle_streams: 3,
            attachments: 1,
            chapters: 12,
            attachment_names: vec!["Font.ttf".into()],
        }
    }

    #[test]
    fn encode_args_map_the_enhanced_track_first_and_copy_the_rest() {
        let video = sample_video_plan();
        let encoder = video.encoder.clone();
        let chain = filter_chain_for(&video, &encoder);
        let argv = build_encode_args(
            &fake_ffmpeg(),
            Path::new("in.mkv"),
            Some(Path::new("enhanced.wav")),
            &chain,
            &encoder,
            &sample_audio_plan(),
            &sample_inventory(),
            Some("volume=7dB,alimiter=limit=0.84"),
            &crate::pipeline::profile::OutputSettings::default(),
            Path::new("out.mkv"),
        );
        let joined = argv.join(" ");
        // enhanced track is output audio 0, originals follow as copied streams
        let enhanced_at = joined.find("-map 1:a:0").expect("enhanced mapped");
        let originals_at = joined.find("-map 0:a?").expect("originals mapped");
        assert!(enhanced_at < originals_at);
        assert!(joined.contains("-c:a copy"));
        assert!(joined.contains("-c:a:0 flac"));
        assert!(joined.contains("-filter:a:0 volume=7dB,alimiter"));
        assert!(joined.contains("-map 0:s?"));
        assert!(joined.contains("-map 0:t?"));
        assert!(joined.contains("-map_chapters 0"));
        assert!(joined.contains("-vf scale=1440:960"));
        assert!(joined.contains("-c:v av1_nvenc"));
        assert!(joined.contains("-pix_fmt p010le"));
        assert!(joined.contains("-f matroska"));
        // both source tracks must lose the default flag
        assert!(joined.contains("-disposition:a:1 0"));
        assert!(joined.contains("-disposition:a:2 0"));
    }

    #[test]
    fn encode_args_without_a_remaster_just_copy_every_audio_stream() {
        let video = sample_video_plan();
        let encoder = video.encoder.clone();
        let chain = filter_chain_for(&video, &encoder);
        let argv = build_encode_args(
            &fake_ffmpeg(),
            Path::new("in.mkv"),
            None,
            &chain,
            &encoder,
            &sample_audio_plan(),
            &sample_inventory(),
            None,
            &crate::pipeline::profile::OutputSettings::default(),
            Path::new("out.mkv"),
        );
        let joined = argv.join(" ");
        assert!(!joined.contains("1:a:0"));
        assert!(!joined.contains("-c:a:0"));
        assert!(joined.contains("-map 0:a?"));
        assert!(joined.contains("-c:a copy"));
    }

    #[test]
    fn final_chain_uses_a_pure_gain_when_the_peak_allows_it() {
        let target = LoudnessTarget::default();
        // -26 LUFS measured, -12 dBTP peak: +7 dB lands at -5 dBTP, inside target.
        let (chain, gain, limiting) = build_gain_chain(-26.0, Some(-12.0), &target);
        assert!((gain - 7.0).abs() < 1e-9);
        assert!(!limiting, "no limiting should be needed");
        assert!(chain.starts_with("volume=7.000dB"));
        assert!(chain.contains("alimiter=limit=0.8414"));
        assert!(!chain.contains("loudnorm"));
    }

    #[test]
    fn final_chain_reports_when_the_limiter_must_work() {
        let target = LoudnessTarget::default();
        // +7 dB of gain on a -2 dBTP peak would land at +5 dBTP.
        let (chain, _, limiting) = build_gain_chain(-26.0, Some(-2.0), &target);
        assert!(limiting);
        assert!(chain.contains("alimiter="));
    }

    #[test]
    fn a_track_already_on_target_gets_no_gain_but_keeps_the_limiter() {
        let target = LoudnessTarget::default();
        let (chain, gain, limiting) = build_gain_chain(target.i_lufs as f64, Some(-6.0), &target);
        assert!(gain.abs() < 0.05);
        assert!(!limiting);
        assert_eq!(chain, "alimiter=limit=0.8414:level=disabled:attack=5:release=50");
    }

    #[test]
    fn unknown_true_peak_still_normalises_the_loudness() {
        let target = LoudnessTarget::default();
        let (chain, gain, limiting) = build_gain_chain(-25.0, None, &target);
        assert!((gain - 6.0).abs() < 1e-9);
        assert!(!limiting, "we cannot claim limiting is needed without knowing the peak");
        assert!(chain.starts_with("volume=6.000dB"));
    }

    #[test]
    fn duration_formatting_is_readable() {
        assert_eq!(format_duration(Duration::from_secs(45)), "45s");
        assert_eq!(format_duration(Duration::from_secs(125)), "2m05s");
        assert_eq!(format_duration(Duration::from_secs(3725)), "1h02m05s");
    }

    #[test]
    fn qc_summary_counts_failures() {
        let report = QcReport {
            output: PathBuf::from("out.mkv"),
            checks: vec![
                QcCheck {
                    name: "a".into(),
                    passed: true,
                    detail: String::new(),
                },
                QcCheck {
                    name: "b".into(),
                    passed: false,
                    detail: "bad".into(),
                },
            ],
            passed: false,
        };
        assert_eq!(report.failures().len(), 1);
        assert!(report.summary().contains("1 of 2"));
    }

    fn fake_ffmpeg() -> Ffmpeg {
        Ffmpeg::from_paths("ffmpeg".into(), "ffprobe".into())
            .unwrap_or_else(|_| panic!("this test only needs the encoder tables"))
    }

    #[test]
    fn remaster_decision_summaries_are_human_readable() {
        let skipped = RemasterDecision::skipped("LDR within target", Some(3.2), 1.0);
        assert!(skipped.summary().contains("no dynamic adaptation"));
    }
}
