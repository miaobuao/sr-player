//! The native inference executor: decode, model session, per-chunk encode.
//!
//! This is the part that was missing. Before it existed, a plan could say
//! `InterpolationMethod::Plugin` and the runner would hand the whole job to a
//! single FFmpeg pass, which quietly ran `minterpolate` instead of the model: the
//! plugin was discovered, its capabilities were logged, and nothing was ever
//! executed. Here is what actually happens now:
//!
//! ```text
//! ffmpeg -i in -vf <cadence>,format=rgb24 -f rawvideo   (stdout)
//!        |
//!        +- per segment:  carry -> window of at most N frames
//!        |                restore      (model session)
//!        |                interpolate  (model session)
//!        |
//!        +- per chunk:    raw frames -> ffmpeg -> chunk-00042.mkv  (checkpointed)
//!                                              |
//!                            concat -----------+--> final encode + mux of audio,
//!                                                   subtitles, chapters, attachments
//! ```
//!
//! Three properties are worth stating explicitly, because they are the difference
//! between "a model ran" and "a model was supposed to run":
//!
//! * **The model never sees a cut.** Segments come from the shot list, and a
//!   segment containing a boundary is executed as `Hold`: frames are repeated
//!   instead of synthesised. Neither FFmpeg nor the model gets to make that
//!   decision.
//! * **Resume is per chunk.** Each chunk is its own encode, committed to the
//!   `chunks` table with its artifact before the next one starts. A job that dies
//!   at 95% re-runs one chunk, not the film.
//! * **Out of memory degrades, it does not fail.** A backend answering
//!   `SR_ERR_OUT_OF_MEMORY` walks the working-set ladder, tells the session about
//!   the smaller working set, and the same segment runs again.
//!
//! Frame accounting is exact: `m` times interpolation of `n` frames emits
//! `m*(n-1)+1` frames, and the executor fails loudly if what it wrote disagrees
//! with the plan. A short output is never published quietly.

use crate::error::{Error, Result};
use crate::events::{Reporter, Stage, StageProgress, StageStatus};
use crate::ffmpeg::process::{write_all, StreamingChild};
use crate::ffmpeg::{args, run_tool, Ffmpeg, RunSpec, SelectedAudioEncoder};
use crate::infer::abi::{FrameBuffer, JobOptions, SessionRequest};
use crate::infer::{EngineRegistry, EngineSession, WorkingSetRequest};
use crate::media::scene::SceneReport;
use crate::pipeline::plan::{
    model_output_geometry, output_extension, ConversionPlan, PlanRequest, VideoPlan,
};
use crate::pipeline::policy::WorkingSet;
use crate::pipeline::profile::InterpolationMethod;
use crate::pipeline::segments::{plan_run, Chunk, RunPlan, Segment, SegmentKind, SegmentOptions};
use crate::state::{commit_file_atomic, ChunkRow, Store};
use crate::time::Rational;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How a chunk on disk is encoded.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkEncoding {
    /// FFV1 lossless intermediates, then one final encode pass.
    ///
    /// The default: the published file is a single encode with one encoder
    /// session, so two chunks can never disagree about codec parameters and the
    /// quality is what a one-pass encode would have produced. It costs a second
    /// pass over the video and a lot of scratch disk.
    LosslessIntermediate,
    /// Encode each chunk with the final encoder and concatenate with `-c:v copy`.
    ///
    /// Half the work and a fraction of the disk, at the cost of relying on every
    /// chunk producing bit-compatible stream parameters.
    DirectFinalCodec,
}

impl ChunkEncoding {
    pub fn as_str(self) -> &'static str {
        match self {
            ChunkEncoding::LosslessIntermediate => "ffv1-intermediate",
            ChunkEncoding::DirectFinalCodec => "direct",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "ffv1" | "ffv1-intermediate" | "lossless" => Some(ChunkEncoding::LosslessIntermediate),
            "direct" | "final" => Some(ChunkEncoding::DirectFinalCodec),
            _ => None,
        }
    }

    /// Rough scratch-space estimate, so an unattended run can say what it needs
    /// before it fills a disk.
    pub fn describe(self, seconds: f64, width: u32, height: u32) -> String {
        let frames = seconds * 24.0;
        let raw = frames * width as f64 * height as f64 * 3.0;
        match self {
            ChunkEncoding::LosslessIntermediate => format!(
                "lossless intermediates, roughly {:.0}-{:.0} GiB of scratch space for this file",
                raw / 1_073_741_824.0 * 0.35,
                raw / 1_073_741_824.0 * 0.6
            ),
            ChunkEncoding::DirectFinalCodec => {
                "encoded chunks concatenated with a stream copy: scratch stays small".to_string()
            }
        }
    }
}

/// Everything the executor needs that the runner owns.
pub struct NativeContext<'a> {
    pub job_id: &'a str,
    pub request: &'a PlanRequest,
    pub plan: &'a ConversionPlan,
    pub scenes: &'a SceneReport,
    pub reporter: &'a Reporter,
    pub remastered: Option<&'a Path>,
    pub loudness_chain: Option<String>,
    pub working: WorkingSet,
    pub chunk_encoding: ChunkEncoding,
    pub max_degrade_retries: u32,
    /// Directory for chunk files; created if missing.
    pub workdir: &'a Path,
}

pub struct NativeOutcome {
    pub output: PathBuf,
    pub chunks_written: u32,
    pub chunks_resumed: u32,
    pub input_frames: u64,
    pub output_frames: u64,
    pub model_calls: u64,
    pub degradations: Vec<String>,
    pub chunk_encoding: ChunkEncoding,
}

pub struct NativeExecutor<'a> {
    ff: &'a Arc<Ffmpeg>,
    engines: &'a Arc<EngineRegistry>,
    store: &'a Arc<Store>,
    cancel: &'a Arc<AtomicBool>,
}

impl<'a> NativeExecutor<'a> {
    pub fn new(
        ff: &'a Arc<Ffmpeg>,
        engines: &'a Arc<EngineRegistry>,
        store: &'a Arc<Store>,
        cancel: &'a Arc<AtomicBool>,
    ) -> Self {
        NativeExecutor {
            ff,
            engines,
            store,
            cancel,
        }
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Runs the whole native video path and returns the finished output path.
    pub fn run(&self, ctx: &NativeContext<'_>) -> Result<NativeOutcome> {
        let video = &ctx.plan.video;
        let inference = video.inference.as_ref().ok_or_else(|| Error::Stage {
            stage: Stage::Encode.id().into(),
            detail: "the plan selected the native executor without an inference plan".into(),
        })?;
        let engine = self.engines.model_engine().ok_or_else(|| Error::Stage {
            stage: Stage::Encode.id().into(),
            detail: "the plan selected the native executor but no model plugin is loaded".into(),
        })?;

        let interpolate = video.interpolation.enabled
            && video.interpolation.method == InterpolationMethod::Plugin;
        let restore = inference.describes("restore");
        if !interpolate && !restore {
            return Err(Error::Stage {
                stage: Stage::Encode.id().into(),
                detail: "the native executor was selected with no model task to run".into(),
            });
        }

        // Every frame the decode pass will produce, according to the shot
        // analysis, which ran on the same cadence chain: the numbering matches by
        // construction rather than by hope.
        let input_frames = ctx.scenes.frames_analyzed;
        let multiplier = if interpolate {
            video.interpolation.multiplier.max(1)
        } else {
            1
        };
        let run_plan = plan_run(
            input_frames,
            multiplier,
            &ctx.scenes.shots,
            &SegmentOptions {
                max_input_frames: inference.temporal_window,
                target_chunk_frames: 1000,
                dissolve_guard_frames: 12,
                interpolate,
            },
        );
        for note in &run_plan.notes {
            ctx.reporter.info(Some(Stage::Plan), note.clone());
        }
        ctx.reporter.info(
            Some(Stage::Plan),
            format!(
                "native plan: {} input frame(s) to {} output frame(s) in {} chunk(s), {} \
                 segment(s); {}",
                run_plan.input_frames,
                run_plan.total_output_frames,
                run_plan.chunks.len(),
                run_plan.segments.len(),
                ctx.chunk_encoding.describe(
                    video.source.duration_seconds,
                    video.target_width,
                    video.target_height
                )
            ),
        );
        if run_plan.input_frames == 0 {
            return Err(Error::Stage {
                stage: Stage::Restore.id().into(),
                detail: "the shot analysis counted no frames: there is nothing to process".into(),
            });
        }

        // ---- session ------------------------------------------------------
        let request = session_request(ctx, inference.temporal_window);
        let mut session = engine.open_session(&request).map_err(|err| Error::Stage {
            stage: Stage::Restore.id().into(),
            detail: format!(
                "could not open a model session on {}: {err}",
                engine.display_name()
            ),
        })?;
        ctx.reporter.info(
            Some(Stage::Restore),
            format!(
                "model session open: {} | {} | window {} | budget {} MiB",
                engine.display_name(),
                if inference.model.is_empty() {
                    "unnamed model".to_string()
                } else {
                    inference.model.clone()
                },
                inference.temporal_window,
                ctx.working.estimated_mib()
            ),
        );

        let mut outcome = self.execute_chunks(
            ctx,
            &run_plan,
            session.as_mut(),
            restore,
            interpolate,
            multiplier,
        )?;
        outcome.output = self.finalize(ctx, &run_plan)?;
        // Released explicitly so the log can say the model was gone before the
        // mux started.
        drop(session);
        ctx.reporter.debug(Some(Stage::Restore), "model session closed");
        Ok(outcome)
    }

    fn execute_chunks(
        &self,
        ctx: &NativeContext<'_>,
        run_plan: &RunPlan,
        session: &mut dyn EngineSession,
        restore: bool,
        interpolate: bool,
        multiplier: u32,
    ) -> Result<NativeOutcome> {
        let video = &ctx.plan.video;
        let chunks_dir = ctx.workdir.join("chunks");
        std::fs::create_dir_all(&chunks_dir).map_err(|e| Error::io(&chunks_dir, e))?;

        let decode_argv = build_decode_args(&ctx.request.input, video);
        let spec = RunSpec::new(Stage::Restore, "decode")
            .with_frames(run_plan.input_frames.max(1))
            .quiet("deprecated pixel format");
        let mut decoder =
            StreamingChild::spawn(&self.ff.ffmpeg, &decode_argv, ctx.reporter, &spec, false)?;
        let stdout = decoder.stdout.take().ok_or_else(|| Error::Stage {
            stage: Stage::Restore.id().into(),
            detail: "the decoder produced no stdout pipe".into(),
        })?;

        let (width, height) = (video.source.square_width, video.source.square_height);
        // The model's own output size. The post chain already knows how to turn
        // that into the target raster.
        let model_size = {
            let (w, h) = model_output_geometry(video.inference.as_ref());
            if w == 0 || h == 0 {
                (width, height)
            } else {
                (w, h)
            }
        };
        let frame_bytes = width as usize * height as usize * 3;

        let mut source = FrameSource::new(stdout, frame_bytes, width, height);
        let existing = self.committed_chunks(ctx.job_id)?;

        let mut outcome = NativeOutcome {
            output: ctx.request.output.clone(),
            chunks_written: 0,
            chunks_resumed: 0,
            input_frames: run_plan.input_frames,
            output_frames: 0,
            model_calls: 0,
            degradations: Vec::new(),
            chunk_encoding: ctx.chunk_encoding,
        };
        let mut working = ctx.working.clone();
        let started = Instant::now();
        let mut written_slots = 0u64;

        for chunk in &run_plan.chunks {
            self.check_cancel()?;
            let path = chunk_path(&chunks_dir, chunk);
            let resumed = existing
                .get(&chunk.index)
                .map(|artifact| artifact == &path && path.exists())
                .unwrap_or(false);

            if resumed {
                // The decoder is a stream, so a resumed chunk still has to be read
                // past; nothing is inferred or encoded again.
                source.skip_to(chunk.last_input)?;
                outcome.chunks_resumed += 1;
                written_slots += chunk.outputs;
                ctx.reporter.info(
                    Some(Stage::Encode),
                    format!(
                        "chunk {:04} already committed: {} frame(s) kept from the previous run",
                        chunk.index, chunk.outputs
                    ),
                );
                continue;
            }

            let mut encoder = ChunkEncoder::start(self.ff, ctx, video, &path, model_size)?;
            let mut frames_written = 0u64;
            for segment in &run_plan.segments[chunk.first_segment..=chunk.last_segment] {
                self.check_cancel()?;
                frames_written += self.run_segment(
                    ctx,
                    segment,
                    chunk.index,
                    &mut source,
                    session,
                    restore,
                    interpolate,
                    multiplier,
                    &mut working,
                    &mut encoder,
                    &mut outcome,
                )?;
            }
            encoder.finish()?;

            if frames_written != chunk.outputs {
                // Exact by construction, so this means the decoder produced a
                // different number of frames than the shot analysis counted.
                // Report it loudly instead of publishing a file whose length
                // nobody can explain.
                ctx.reporter.error(
                    Some(Stage::Encode),
                    format!(
                        "chunk {:04} wrote {frames_written} frame(s) but the plan expected {}; \
                         the decode stream and the shot analysis disagree",
                        chunk.index, chunk.outputs
                    ),
                );
                return Err(Error::Stage {
                    stage: Stage::Encode.id().into(),
                    detail: format!(
                        "chunk {} frame count mismatch: {frames_written} written, {} planned",
                        chunk.index, chunk.outputs
                    ),
                });
            }

            self.store.commit_chunk(
                ctx.job_id,
                &ChunkRow {
                    stage: Stage::Encode,
                    chunk_index: chunk.index,
                    start_frame: Some(chunk.first_input as i64),
                    end_frame: Some(chunk.last_input as i64),
                    status: "committed".into(),
                    artifact: Some(path.clone()),
                    updated_ms: 0,
                },
            )?;
            outcome.chunks_written += 1;
            written_slots += frames_written;
            self.report_progress(ctx, run_plan, chunk, written_slots, &outcome, &started);
        }

        // Anything past the plan is a surprise. Draining keeps the decoder from
        // blocking on a full pipe, and the count is reported either way.
        let extra = source.drain_rest()?;
        let decoded = source.frames_read;
        let padded = source.padded;
        let status = decoder.wait();
        if extra > 0 {
            ctx.reporter.warn(
                Some(Stage::Encode),
                format!(
                    "{extra} frame(s) beyond the {decoded} the shot analysis counted were decoded \
                     and discarded"
                ),
            );
        }
        if padded > 0 {
            ctx.reporter.warn(
                Some(Stage::Encode),
                format!(
                    "{padded} frame(s) were missing from the decode stream and were filled by \
                     repeating the previous frame"
                ),
            );
        }
        status?;

        outcome.output_frames = written_slots;
        Ok(outcome)
    }

    fn report_progress(
        &self,
        ctx: &NativeContext<'_>,
        run_plan: &RunPlan,
        chunk: &Chunk,
        written_slots: u64,
        outcome: &NativeOutcome,
        started: &Instant,
    ) {
        let fraction = if run_plan.input_frames > 0 {
            ((chunk.last_input + 1) as f32 / run_plan.input_frames as f32).min(1.0)
        } else {
            1.0
        };
        let elapsed = started.elapsed().as_secs_f64().max(0.001);
        let fps = written_slots as f64 / elapsed;
        ctx.reporter.progress(StageProgress {
            stage: Stage::Encode,
            fraction: Some(fraction),
            detail: format!(
                "chunk {}/{} | shot {:04} | {} model call(s){}",
                chunk.index + 1,
                run_plan.chunks.len(),
                chunk.shot + 1,
                outcome.model_calls,
                if chunk.protected_boundaries > 0 {
                    format!(" | {} protected region(s)", chunk.protected_boundaries)
                } else {
                    String::new()
                }
            ),
            frames: Some(written_slots),
            fps: if fps > 0.0 { Some(fps) } else { None },
            speed: None,
            out_time: None,
            eta: None,
        });
    }

    /// Executes one segment, writing its frames to the chunk encoder.
    ///
    /// Returns the number of frames written.
    #[allow(clippy::too_many_arguments)]
    fn run_segment(
        &self,
        ctx: &NativeContext<'_>,
        segment: &Segment,
        chunk_index: u32,
        source: &mut FrameSource,
        session: &mut dyn EngineSession,
        restore: bool,
        interpolate: bool,
        multiplier: u32,
        working: &mut WorkingSet,
        encoder: &mut ChunkEncoder,
        outcome: &mut NativeOutcome,
    ) -> Result<u64> {
        match segment.kind {
            SegmentKind::Synthesise => {
                let m = multiplier.max(1) as u64;
                let mut frames: Vec<FrameBuffer> =
                    Vec::with_capacity(segment.input_frames() as usize);
                for index in segment.first..=segment.last {
                    frames.push(source.frame(index)?);
                }
                // Input frame k of the segment sits at output slot m*k.
                stamp(ctx, &mut frames, |i| m * (segment.first + i as u64));
                if restore {
                    self.restore_batch(ctx, session, chunk_index, &mut frames, working, outcome)?;
                }
                // The next segment starts on this frame, and it must be the
                // restored version of it.
                let carry = frames.last().cloned();
                let mut outputs: Vec<FrameBuffer> = if interpolate && multiplier > 1 {
                    let produced = (frames.len() - 1) as u32 * multiplier + 1;
                    let mut outputs: Vec<FrameBuffer> = (0..produced)
                        .map(|_| FrameBuffer::new_rgb8(frames[0].width, frames[0].height))
                        .collect();
                    stamp(ctx, &mut outputs, |i| m * segment.first + i as u64);
                    self.call_model(
                        ctx,
                        session,
                        chunk_index,
                        &mut frames,
                        &mut outputs,
                        multiplier,
                        working,
                        outcome,
                    )?;
                    outputs
                } else {
                    frames
                };
                if segment.drop_first && !outputs.is_empty() {
                    outputs.remove(0);
                }
                let mut written = 0u64;
                for frame in &outputs {
                    encoder.write_frame(&frame.data)?;
                    written += 1;
                }
                if let Some(frame) = carry {
                    source.set_carry(segment.last, frame);
                }
                Ok(written)
            }
            SegmentKind::Hold => {
                // No model call: these pairs may not be synthesised, so the frames
                // repeat. Streamed one at a time, so a long held region costs one
                // frame of memory rather than a thousand.
                let mut written = 0u64;
                let mut skip_next = segment.drop_first;
                let mut carry: Option<FrameBuffer> = None;
                for index in segment.first..=segment.last {
                    let frame = source.frame(index)?;
                    let mut frames = vec![frame];
                    if restore {
                        self.restore_batch(ctx, session, chunk_index, &mut frames, working, outcome)?;
                    }
                    let frame = &frames[0];
                    if index == segment.last {
                        carry = Some(frame.clone());
                    }
                    // Slots m*k .. m*k+m-1 are the frame and its repeats; the last
                    // frame of the segment contributes only itself.
                    let repeats = if index == segment.last {
                        0
                    } else {
                        multiplier.saturating_sub(1)
                    };
                    let mut emissions = 1 + repeats;
                    if skip_next {
                        skip_next = false;
                        emissions = emissions.saturating_sub(1);
                    }
                    for _ in 0..emissions {
                        encoder.write_frame(&frame.data)?;
                        written += 1;
                    }
                }
                if let Some(frame) = carry {
                    source.set_carry(segment.last, frame);
                }
                Ok(written)
            }
            SegmentKind::Pass => {
                let frame = source.frame(segment.first)?;
                let mut frames = vec![frame];
                if restore {
                    self.restore_batch(ctx, session, chunk_index, &mut frames, working, outcome)?;
                }
                if segment.drop_first {
                    return Ok(0);
                }
                encoder.write_frame(&frames[0].data)?;
                Ok(1)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn call_model(
        &self,
        ctx: &NativeContext<'_>,
        session: &mut dyn EngineSession,
        chunk_index: u32,
        inputs: &mut [FrameBuffer],
        outputs: &mut [FrameBuffer],
        multiplier: u32,
        working: &mut WorkingSet,
        outcome: &mut NativeOutcome,
    ) -> Result<()> {
        let mut options = JobOptions {
            multiplier,
            strength: restore_strength(ctx),
            tile: tile_for(working),
            seed: 0,
            // The chunk index, so a plugin's own log lines up with the engine's
            // `chunks` table when someone has to work out what happened.
            chunk_id: chunk_index as u64,
        };
        let mut attempt = 0u32;
        loop {
            self.check_cancel()?;
            match session.interpolate(inputs, outputs, &options) {
                Ok(result) => {
                    outcome.model_calls += 1;
                    if let Some(message) = result.message {
                        if !message.is_empty() {
                            ctx.reporter.debug(Some(Stage::Interpolate), message);
                        }
                    }
                    return Ok(());
                }
                Err(err) if err.is_oom() && attempt < ctx.max_degrade_retries => {
                    attempt += 1;
                    match self.degrade(ctx, session, working, outcome, Stage::Interpolate)? {
                        Some(tile) => {
                            options.tile = tile;
                            continue;
                        }
                        None => return Err(err),
                    }
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Applies one rung of the degrade ladder.
    ///
    /// `Some(tile)` means "retry with this tile"; `None` means the ladder is
    /// exhausted and the caller must report the original failure.
    fn degrade(
        &self,
        ctx: &NativeContext<'_>,
        session: &mut dyn EngineSession,
        working: &mut WorkingSet,
        outcome: &mut NativeOutcome,
        stage: Stage,
    ) -> Result<Option<Option<(u32, u32)>>> {
        let Some(change) = working.degrade() else {
            ctx.reporter.error(
                Some(stage),
                "out of memory and the degrade ladder is exhausted",
            );
            return Ok(None);
        };
        let refused = match session.reconfigure(&working_request(working)) {
            Ok(()) => None,
            Err(reason) => Some(reason.to_string()),
        };
        let detail = match &refused {
            None => format!("out of memory: {change}"),
            Some(reason) => format!(
                "out of memory: {change}; the backend cannot apply it in place ({reason}), so the \
                 segment runs again with the original configuration"
            ),
        };
        ctx.reporter.warn(Some(stage), detail.clone());
        ctx.reporter
            .stage(stage, StageStatus::Degraded, Some(detail.clone()));
        outcome.degradations.push(detail);
        Ok(Some(tile_for(working)))
    }

    /// Restores frames in place, walking the degrade ladder on out-of-memory.
    fn restore_batch(
        &self,
        ctx: &NativeContext<'_>,
        session: &mut dyn EngineSession,
        chunk_index: u32,
        frames: &mut Vec<FrameBuffer>,
        working: &mut WorkingSet,
        outcome: &mut NativeOutcome,
    ) -> Result<()> {
        let batch = ctx
            .plan
            .video
            .inference
            .as_ref()
            .map(|i| i.max_batch.max(1) as usize)
            .unwrap_or(1);
        let mut offset = 0usize;
        while offset < frames.len() {
            let end = (offset + batch).min(frames.len());
            let mut attempt = 0u32;
            loop {
                self.check_cancel()?;
                let options = JobOptions {
                    multiplier: 1,
                    strength: restore_strength(ctx),
                    tile: tile_for(working),
                    seed: 0,
                    chunk_id: chunk_index as u64,
                };
                match session.restore(&mut frames[offset..end], &options) {
                    Ok(result) => {
                        outcome.model_calls += 1;
                        if let Some(message) = result.message {
                            if !message.is_empty() {
                                ctx.reporter.debug(Some(Stage::Restore), message);
                            }
                        }
                        break;
                    }
                    Err(err) if err.is_oom() && attempt < ctx.max_degrade_retries => {
                        attempt += 1;
                        if self
                            .degrade(ctx, session, working, outcome, Stage::Restore)?
                            .is_none()
                        {
                            return Err(err);
                        }
                    }
                    Err(err) => return Err(err),
                }
            }
            offset = end;
        }
        Ok(())
    }

    /// Concatenates the committed chunks and muxes everything the source had.
    fn finalize(&self, ctx: &NativeContext<'_>, run_plan: &RunPlan) -> Result<PathBuf> {
        let chunks_dir = ctx.workdir.join("chunks");
        let list_path = ctx.workdir.join("chunks.txt");
        let mut list = String::new();
        let mut concatenated = 0u32;
        for chunk in &run_plan.chunks {
            let path = chunk_path(&chunks_dir, chunk);
            if !path.exists() {
                return Err(Error::Stage {
                    stage: Stage::Mux.id().into(),
                    detail: format!(
                        "chunk {} is missing from {}, so the output would be incomplete",
                        chunk.index,
                        chunks_dir.display()
                    ),
                });
            }
            // The concat demuxer resolves relative paths against the list file's
            // own directory, not the working directory, so the entries are made
            // absolute. Forward slashes because the demuxer's escaping rules are
            // FFmpeg's, not the platform's.
            let absolute = std::path::absolute(&path).unwrap_or_else(|_| path.clone());
            let text = absolute.display().to_string().replace('\\', "/");
            list.push_str(&format!("file '{}'\n", text.replace('\'', "'\\''")));
            concatenated += 1;
        }
        std::fs::write(&list_path, list).map_err(|e| Error::io(&list_path, e))?;
        ctx.reporter.info(
            Some(Stage::Mux),
            format!(
                "concatenating {concatenated} chunk(s) from {} intermediates",
                ctx.chunk_encoding.as_str()
            ),
        );

        let temp_output = ctx.workdir.join(format!(
            "native-output.{}",
            output_extension(&ctx.plan.output_settings)
        ));
        let argv = build_final_args(ctx, &list_path, &temp_output);
        let spec = RunSpec::new(Stage::Mux, "final-mux")
            .with_duration(Duration::from_secs_f64(
                ctx.plan.video.source.duration_seconds.max(1.0),
            ))
            .with_frames(run_plan.total_output_frames.max(1));
        run_tool(&self.ff.ffmpeg, &argv, ctx.reporter, self.cancel, &spec)?;
        commit_file_atomic(&temp_output, &ctx.request.output)?;
        Ok(ctx.request.output.clone())
    }

    fn committed_chunks(&self, job_id: &str) -> Result<HashMap<u32, PathBuf>> {
        let rows = self.store.chunks(job_id)?;
        let mut map = HashMap::new();
        for row in rows {
            if row.stage == Stage::Encode && row.status == "committed" {
                if let Some(artifact) = row.artifact {
                    map.insert(row.chunk_index, artifact);
                }
            }
        }
        Ok(map)
    }
}

// ---- frame plumbing -------------------------------------------------------

/// Exact rational timestamps for a run of frames, so a model that cares about
/// time places a synthetic frame precisely instead of accumulating float drift
/// across a two-hour film.
///
/// `slot_of` yields the output slot index of each frame; the timestamp is that
/// slot over the pipe frame rate, which is exact because both sides are integers.
fn stamp<F: Fn(usize) -> u64>(ctx: &NativeContext<'_>, frames: &mut [FrameBuffer], slot_of: F) {
    let den = ctx.plan.video.pipe_fps().num().max(1);
    for (index, frame) in frames.iter_mut().enumerate() {
        frame.pts_num = slot_of(index) as i64;
        frame.pts_den = den;
    }
}

/// A forward-only frame stream with the single frame of carry the segment plan
/// needs: consecutive segments share their boundary frame.
struct FrameSource {
    reader: std::io::BufReader<std::process::ChildStdout>,
    frame_bytes: usize,
    width: u32,
    height: u32,
    next_index: u64,
    carry: Option<(u64, FrameBuffer)>,
    last: Option<FrameBuffer>,
    frames_read: u64,
    eof: bool,
    /// Frames invented because the stream ended before the plan did.
    padded: u64,
}

impl FrameSource {
    fn new(
        reader: std::io::BufReader<std::process::ChildStdout>,
        frame_bytes: usize,
        width: u32,
        height: u32,
    ) -> Self {
        FrameSource {
            reader,
            frame_bytes,
            width,
            height,
            next_index: 0,
            carry: None,
            last: None,
            frames_read: 0,
            eof: false,
            padded: 0,
        }
    }

    /// The frame at absolute index `index`.
    ///
    /// Callers must not ask for a frame before one they have already consumed;
    /// the carry covers the single frame that consecutive segments share.
    fn frame(&mut self, index: u64) -> Result<FrameBuffer> {
        if let Some((held, _)) = &self.carry {
            if *held == index {
                let (_, frame) = self.carry.take().expect("checked above");
                return Ok(frame);
            }
        }
        if index != self.next_index {
            return Err(Error::Stage {
                stage: Stage::Restore.id().into(),
                detail: format!(
                    "frame {index} was requested out of order (next readable frame is {})",
                    self.next_index
                ),
            });
        }
        let frame = self.read_one()?;
        self.next_index = index + 1;
        Ok(frame)
    }

    fn read_one(&mut self) -> Result<FrameBuffer> {
        if self.eof {
            return self.repeat_last();
        }
        let mut buffer = vec![0u8; self.frame_bytes];
        let read = crate::ffmpeg::process::read_exact_or_eof(&mut self.reader, &mut buffer)?;
        if read < self.frame_bytes {
            // A short read means the stream ended mid-frame: treat the stream as
            // finished and repeat the last good frame from here on.
            self.eof = true;
            return self.repeat_last();
        }
        self.frames_read += 1;
        let frame = FrameBuffer::from_owned(buffer, self.width, self.height);
        self.last = Some(frame.clone());
        Ok(frame)
    }

    fn repeat_last(&mut self) -> Result<FrameBuffer> {
        self.padded += 1;
        self.last.clone().ok_or_else(|| Error::Stage {
            stage: Stage::Restore.id().into(),
            detail: format!(
                "the decoder stopped after {} frame(s), before a single complete frame arrived",
                self.frames_read
            ),
        })
    }

    /// Hands the frame at `index` back to the stream so the next segment, which
    /// starts on the same frame, can pick it up.
    ///
    /// The frame stored is the one the model produced (restoration happens in
    /// place), which is what the next window must contain: interpolating between
    /// a restored frame and an unrestored one would show up as a flicker.
    fn set_carry(&mut self, index: u64, frame: FrameBuffer) {
        self.carry = Some((index, frame));
    }

    /// Consumes frames up to and including `index`, keeping it as the carry so the
    /// next segment can start there.
    fn skip_to(&mut self, index: u64) -> Result<()> {
        if let Some((held, _)) = &self.carry {
            if *held >= index {
                return Ok(());
            }
        }
        while self.next_index <= index {
            let frame = self.read_one()?;
            let at = self.next_index;
            self.next_index = at + 1;
            self.carry = Some((at, frame));
        }
        Ok(())
    }

    /// Reads whatever is left, so the decoder never blocks on a full pipe.
    fn drain_rest(&mut self) -> Result<u64> {
        let mut buffer = vec![0u8; self.frame_bytes.max(1) * 8];
        let mut extra = 0u64;
        let mut leftover = 0usize;
        loop {
            let read = self.reader.read(&mut buffer).map_err(Error::BareIo)?;
            if read == 0 {
                break;
            }
            leftover += read;
            while leftover >= self.frame_bytes {
                leftover -= self.frame_bytes;
                extra += 1;
            }
        }
        Ok(extra)
    }
}

// ---- encoders -------------------------------------------------------------

/// One chunk's encoder process, fed raw frames on stdin.
struct ChunkEncoder {
    child: StreamingChild,
    stdin: Option<std::process::ChildStdin>,
}

impl ChunkEncoder {
    fn start(
        ff: &Ffmpeg,
        ctx: &NativeContext<'_>,
        video: &VideoPlan,
        path: &Path,
        model_size: (u32, u32),
    ) -> Result<Self> {
        let argv = build_chunk_args(ctx, video, path, model_size);
        let spec = RunSpec::new(Stage::Encode, "chunk-encode");
        let mut child = StreamingChild::spawn(&ff.ffmpeg, &argv, ctx.reporter, &spec, true)?;
        let stdin = child.stdin.take().ok_or_else(|| Error::Stage {
            stage: Stage::Encode.id().into(),
            detail: "the chunk encoder produced no stdin pipe".into(),
        })?;
        Ok(ChunkEncoder {
            child,
            stdin: Some(stdin),
        })
    }

    fn write_frame(&mut self, data: &[u8]) -> Result<()> {
        let stdin = self.stdin.as_mut().ok_or_else(|| Error::Stage {
            stage: Stage::Encode.id().into(),
            detail: "the chunk encoder is already closed".into(),
        })?;
        write_all(stdin, data)
    }

    /// Closing stdin is what tells the encoder the stream is finished; the exit
    /// status is the only proof that the chunk is complete.
    fn finish(mut self) -> Result<()> {
        drop(self.stdin.take());
        self.child.wait().map(|_| ())
    }
}

// ---- argument construction ------------------------------------------------

/// The decode side: cadence filters, square pixels, 8-bit RGB.
pub fn build_decode_args(input: &Path, video: &VideoPlan) -> Vec<String> {
    let mut argv = args(&["-hide_banner", "-nostdin", "-y", "-i"]);
    argv.push(input.display().to_string());
    let mut filters: Vec<String> = Vec::new();
    if let Some(chain) = video.cadence.chain_str() {
        if !chain.is_empty() {
            filters.push(chain.to_string());
        }
    }
    // A model must see square pixels: the raster is a storage detail, and feeding
    // it anamorphic pixels would have it restore the wrong geometry.
    if (video.source.width, video.source.height)
        != (video.source.square_width, video.source.square_height)
    {
        filters.push(format!(
            "scale={}:{}:flags=lanczos",
            video.source.square_width, video.source.square_height
        ));
    }
    filters.push("format=rgb24".to_string());
    let chain = filters.join(",");
    argv.extend(args(&[
        "-map",
        "0:v:0",
        "-vf",
        &chain,
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]));
    argv
}

fn build_chunk_args(
    ctx: &NativeContext<'_>,
    video: &VideoPlan,
    path: &Path,
    model_size: (u32, u32),
) -> Vec<String> {
    let mut argv = args(&["-hide_banner", "-nostdin", "-y"]);
    let size = format!("{}x{}", model_size.0, model_size.1);
    let rate = rational_arg(video.pipe_fps());
    argv.extend(args(&[
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        &size,
        "-r",
        &rate,
        "-i",
        "-",
    ]));
    if !video.filter_chain.is_empty() {
        argv.extend(args(&["-vf", &video.filter_chain]));
    }
    match ctx.chunk_encoding {
        ChunkEncoding::LosslessIntermediate => {
            // Planar RGB in FFV1 is genuinely lossless; going through YUV here
            // would silently subsample the chroma of every chunk.
            argv.extend(args(&[
                "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", "-slices", "4", "-slicecrc",
                "1",
            ]));
        }
        ChunkEncoding::DirectFinalCodec => {
            argv.extend(args(&["-c:v", &video.encoder.name]));
            argv.extend(video.encoder.quality_args.iter().cloned());
            let pix_fmt = video.pix_fmt.clone();
            argv.extend(args(&["-pix_fmt", &pix_fmt]));
        }
    }
    argv.push("-f".into());
    argv.push("matroska".into());
    argv.push(path.display().to_string());
    argv
}

/// The final pass: concatenate the chunks, mux everything the source had.
fn build_final_args(ctx: &NativeContext<'_>, list: &Path, output: &Path) -> Vec<String> {
    let plan = ctx.plan;
    let video = &plan.video;
    let inventory = &plan.inventory;
    let settings = &plan.output_settings;
    let audio: &SelectedAudioEncoder = &plan.audio.encoder;

    let mut argv = args(&["-hide_banner", "-nostdin", "-progress", "pipe:1", "-y"]);
    // VAAPI-style device setup belongs before the inputs and applies to all of
    // them, so it stays here rather than next to the video input.
    argv.extend(video.encoder.input_args.iter().cloned());
    argv.extend(args(&["-f", "concat", "-safe", "0", "-i"]));
    argv.push(list.display().to_string());
    if let Some(wav) = ctx.remastered {
        argv.push("-i".into());
        argv.push(wav.display().to_string());
    }
    let remastered_index = ctx.remastered.map(|_| 1u32);
    let source_index = remastered_index.map(|i| i + 1).unwrap_or(1);
    let needs_source = (plan.audio.keep_original && inventory.audio_streams > 0)
        || inventory.subtitle_streams > 0
        || inventory.attachments > 0
        || inventory.chapters > 0;
    if needs_source {
        argv.push("-i".into());
        argv.push(ctx.request.input.display().to_string());
    }

    argv.extend(args(&["-map", "0:v:0"]));
    if let Some(index) = remastered_index {
        let map = format!("{index}:a:0");
        argv.extend(args(&["-map", &map]));
    }
    if needs_source {
        if plan.audio.keep_original && inventory.audio_streams > 0 {
            let map = format!("{source_index}:a?");
            argv.extend(args(&["-map", &map]));
        }
        if settings.preserve_subtitles && inventory.subtitle_streams > 0 {
            let map = format!("{source_index}:s?");
            argv.extend(args(&["-map", &map]));
        }
        if settings.preserve_attachments && inventory.attachments > 0 {
            let map = format!("{source_index}:t?");
            argv.extend(args(&["-map", &map]));
        }
        let metadata = source_index.to_string();
        argv.extend(args(&["-map_metadata", &metadata]));
        if settings.preserve_chapters && inventory.chapters > 0 {
            argv.extend(args(&["-map_chapters", &metadata]));
        }
    }

    match ctx.chunk_encoding {
        ChunkEncoding::LosslessIntermediate => {
            argv.extend(args(&["-c:v", &video.encoder.name]));
            argv.extend(video.encoder.quality_args.iter().cloned());
            let pix_fmt = video.pix_fmt.clone();
            argv.extend(args(&["-pix_fmt", &pix_fmt]));
        }
        ChunkEncoding::DirectFinalCodec => {
            // The chunks already went through the encoder and the post-model
            // filter chain, so the final pass only remuxes them.
            argv.extend(args(&["-c:v", "copy"]));
        }
    }

    if inventory.audio_streams > 0 {
        argv.extend(args(&["-c:a", "copy"]));
    }
    if remastered_index.is_some() {
        argv.extend(args(&["-c:a:0", &audio.name]));
        argv.extend(audio.args.iter().cloned());
        if let Some(chain) = &ctx.loudness_chain {
            argv.extend(args(&["-filter:a:0", chain]));
        }
        argv.extend(args(&[
            "-metadata:s:a:0",
            "title=Dialogue enhanced (LDR adapted)",
        ]));
        argv.extend(args(&["-disposition:a:0", "default"]));
        // Every source track that follows must lose any inherited `default` flag,
        // or a player would pick the untouched original.
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
    argv
}

/// Rational frame rate as FFmpeg wants it: exact, never a float.
pub fn rational_arg(fps: Rational) -> String {
    if fps.den() == 0 || fps.den() == 1 {
        format!("{}", fps.num().max(1))
    } else {
        format!("{}/{}", fps.num().max(1), fps.den())
    }
}

fn chunk_path(dir: &Path, chunk: &Chunk) -> PathBuf {
    dir.join(format!("chunk-{:05}.mkv", chunk.index))
}

fn restore_strength(ctx: &NativeContext<'_>) -> f32 {
    ctx.plan
        .video
        .inference
        .as_ref()
        .map(|i| i.restore_strength)
        .unwrap_or(1.0)
}

fn session_request(ctx: &NativeContext<'_>, window: u32) -> SessionRequest {
    let mut config = serde_json::json!({
        "working_set": {
            "batch": ctx.working.batch,
            "block_swap": ctx.working.block_swap,
            "vae_tile": ctx.working.vae_tile,
            "offload": ctx.working.offload.as_str(),
        },
        "temporal_window": window,
        "task": if ctx.plan.video.interpolation.method == InterpolationMethod::Plugin {
            "interpolate+restore"
        } else {
            "restore"
        },
        "input": {
            "width": ctx.plan.video.source.square_width,
            "height": ctx.plan.video.source.square_height,
            "pix_fmt": "rgb24",
        },
    });
    // `SR_INFER_CONFIG_EXTRA` is merged over the generated config so a backend can
    // be given model paths, device indices or its own knobs without the engine
    // having to know about them. It is also how the test suite drives a backend
    // into a specific state (an injected out-of-memory fault, for instance)
    // without a test-only code path in the engine.
    if let Ok(extra) = std::env::var("SR_INFER_CONFIG_EXTRA") {
        if let Ok(serde_json::Value::Object(fields)) = serde_json::from_str(&extra) {
            if let Some(target) = config.as_object_mut() {
                for (key, value) in fields {
                    target.insert(key, value);
                }
            }
        }
    }
    SessionRequest {
        device_index: 0,
        model_path: None,
        model_name: None,
        config_json: Some(config.to_string()),
        vram_budget_bytes: ctx.plan.vram.ai_budget_mib.saturating_mul(1024 * 1024),
        host_memory_budget_bytes: 0,
    }
}

fn working_request(working: &WorkingSet) -> WorkingSetRequest {
    WorkingSetRequest {
        batch: working.batch,
        tile: working.vae_tile,
        offload: working.offload.as_str().to_string(),
        block_swap: working.block_swap,
        precision: String::new(),
    }
}

fn tile_for(working: &WorkingSet) -> Option<(u32, u32)> {
    if working.vae_tile > 0 {
        Some((working.vae_tile, working.vae_tile))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::profile::RestorationProfile;

    #[test]
    fn frame_rates_cross_the_boundary_as_rationals() {
        assert_eq!(rational_arg(Rational::new(24, 1).unwrap()), "24");
        assert_eq!(
            rational_arg(Rational::new(48000, 1001).unwrap()),
            "48000/1001"
        );
        // A degenerate rate must still produce something FFmpeg accepts.
        assert_eq!(rational_arg(Rational::new(0, 1).unwrap()), "1");
    }

    #[test]
    fn the_working_set_request_follows_the_ladder() {
        let profile = RestorationProfile::safe_16gb();
        let mut working = WorkingSet::initial(&profile.restoration);
        let first = working_request(&working);
        let change = working.degrade().expect("the ladder has a first rung");
        assert!(change.contains("block swap"), "{change}");
        let second = working_request(&working);
        assert!(second.block_swap > first.block_swap);
        assert_eq!(second.batch, first.batch);
        assert!(tile_for(&working).is_some());
    }

    #[test]
    fn a_chunk_name_is_stable_and_sortable() {
        let chunk = Chunk {
            index: 42,
            shot: 0,
            first_segment: 0,
            last_segment: 0,
            first_input: 0,
            last_input: 0,
            outputs: 0,
            drop_first: false,
            protected_boundaries: 0,
        };
        let path = chunk_path(Path::new("C:/tmp"), &chunk);
        assert!(path.to_string_lossy().ends_with("chunk-00042.mkv"));
    }

    #[test]
    fn chunk_encoding_round_trips_through_text() {
        for value in [
            ChunkEncoding::LosslessIntermediate,
            ChunkEncoding::DirectFinalCodec,
        ] {
            assert_eq!(ChunkEncoding::parse(value.as_str()), Some(value));
        }
        assert_eq!(ChunkEncoding::parse("nonsense"), None);
    }
}
