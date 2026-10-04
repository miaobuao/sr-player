# Architecture

## The one boundary that matters

```
sr-gui / sr-cli        presentation: pick a file, render events
        │
        ▼
sr-core                all media work: probe, timeline, decode, analysis,
        │              DSP, scheduling, state, child processes
        ▼
FFmpeg                 decode / filter / encode / mux
   └── optional plugin an accelerator behind a C ABI, never required
```

Nothing in `sr-core` knows what a window is. Nothing in `sr-gui` knows what a
codec is: it subscribes to an event bus and renders. That is what makes the
headless CLI, the GUI and any future front end share one implementation of the
hard parts.

## Why the engine is shaped this way

### Rational time, never floats

`Rational`/`Timestamp` carry an exact `pts` plus an explicit timebase. A 2 hour
film at `24000/1001` differs from `23.976` by ~7 ms — most of a frame — and that
is the class of error that turns into lip-sync drift in the last reel. Frame
rates, durations from ffprobe, and frame counts are all parsed into exact
rationals.

### Classify before touching a pixel

`media::classify` samples the runtime with `idet` at seven points and classifies
progressive / telecine / interlaced / mixed from the frames themselves. The
container's `field_order` flag is only used to raise a disagreement note,
because it lies regularly. The four outcomes drive four different filter chains,
and getting it wrong is irreversible: `decimate` on real interlaced video
destroys half the temporal information.

### Shots before interpolation

`media::scene` finds cuts natively (32-bin histogram + edge topology + luma MAD,
threshold adapted from the median/MAD of recent scores). A candidate cut is held
for one frame and confirmed only if the change persists, so a camera flash or a
lightning strike does not disable interpolation around it. Nothing is ever
synthesised across a cut.

### Measure, then decide, then maybe do nothing

`audio` implements ITU-R BS.1770 gating natively, cross-checked against FFmpeg's
`ebur128` (the two agree to 0.1 LU on the fixture). The metric that matters is
`LDR = programme − dialogue`. EBU R128 S4 is explicit: if LDR is already inside
~5 LU, do not adapt further. So the default outcome for a well-behaved film is
**no processing at all**, and the reason is written into the plan and the log.

When a correction is warranted it is a static, measurement-derived dialogue gain
gated to speech-active regions (120 ms attack / 800 ms release, 18 dB/s slew,
`+6 / −3 dB` clamps), plus up to `−4 dB` of subtraction from the 300 Hz–6 kHz
band. Both are applied by subtracting the band rather than resumming bands, so
"no duck" is bit-exact unity. Confidence scales the correction: a low-confidence
detector produces a gentle touch, not a confident mistake.

### Failure degrades, it does not exit

* The plan carries an **encoder chain**, not one encoder. `-encoders` advertises
  encoders that cannot open on the current GPU; the runner walks the chain.
* VRAM policy is `min(13 GiB, free − 2.5 GiB)` — measured against *free* memory,
  because the desktop and the driver are using the same card. OOM walks a ladder
  ordered by cost: throughput first (block swap), then quality (VAE tile), then
  temporal consistency (batch). Batches are restricted to valid `4n+1` values
  (`5 → 1`), because `3` is not a batch this class of model can accept.
* The worker-per-stage rule is documented in `infer`: one model resident at a
  time, stages sequenced, because two models do not fit on a 16 GB card.

### Publish atomically

`state::commit_file_atomic` fsyncs the artifact, then renames it over the
destination. A row never claims success before the bytes are durable, and a
stage's result and its `done` status are committed in one transaction, so a
resumed run cannot see a finished stage without its payload.

### Verify, do not assume

The `Qc` stage re-probes the output and compares it with the plan: duration
drift, resolution, frame rate, every audio track, subtitles, chapters,
attachments, true peak and integrated loudness. It is how the loudness defect in
this very build was caught (`loudnorm`'s JSON report claimed `+29.48 LUFS` for a
file `ebur128` and `volumedetect` both read as `-21.9 LUFS`); the final gain is
now computed from `ebur128`, one source of truth.

## The inference ABI

`include/sr_infer.h` defines three exported symbols
(`sr_infer_abi_version`, `sr_infer_capabilities`, `sr_infer_run`). A plugin can
be backed by Vulkan compute, DirectML, OpenVINO, MIGraphX, TensorRT or a CPU
kernel; the engine only asks what it can do and hands it frames.

`crates/sr-infer-plugin-example` is a real shared library implementing that ABI
with **no dependency on `sr-core`**, and `sr-core`'s tests load it through
`dlopen`/`LoadLibrary` and call it. The boundary is verified, not aspirational.

Engines are ranked and selected: plugin (model) > `minterpolate`
(motion-compensated, FFmpeg) > deterministic resample. The baseline always
exists, so the pipeline always has a correct answer.

## Stage map

| stage | what happens | checkpoint |
|---|---|---|
| Probe | ffprobe → typed manifest (streams, colour, SAR/DAR, chapters, attachments) | yes |
| Temporal | `idet` sampling → progressive/telecine/interlaced/mixed | yes |
| Scenes | native shot detection → shots with rational timestamps | yes |
| AudioAnalysis | `ebur128` + native BS.1770 + dialogue detection → LDR decision | yes |
| Plan | resolution, filters, encoder chain, VRAM budget, reasons and warnings | yes |
| Restore / Interpolate / Regrain | folded into the encode pass in the deterministic path, with honest notes | n/a |
| AudioProcess | dialogue rider + band duck → float WAV on disk (never in RAM) | chunk row |
| Encode / Mux | one FFmpeg pass: filter chain, encoder chain, loudness gain, limiter, `-map` everything | progress events |
| Qc | 11 checks comparing the output with the plan | yes |

## Memory

A two hour 5.1 feature is ~8 GB as float32, so nothing buffers a film: analysis
consumes blocks as they arrive, and the remaster streams into a temporary WAV on
disk that is deleted on success (or kept with `--keep-intermediates`).
