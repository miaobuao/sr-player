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

One refinement matters in practice: `idet`'s *multi-frame* detector is stateful,
so a single pathological pattern can colour a whole file. `testsrc2` encoded with
libx264 — a perfectly progressive clip — reports `Multi frame detection: TFF` for
every frame, while every other synthetic source comes back undetermined. So when
the container says progressive *and* the per-frame detector saw combing in less
than half the frames, the aggregate is overruled and the file is left alone. A
container that lies on genuinely interlaced content still loses: real combing is
visible per frame, so the per-frame signal agrees with the aggregate and there is
no veto.

### Shots before interpolation

`media::scene` finds cuts natively (32-bin histogram + edge topology + luma MAD,
threshold adapted from the median/MAD of recent scores). A candidate cut is held
for one frame and confirmed only if the change persists, so a camera flash or a
lightning strike does not disable interpolation around it. Nothing is ever
synthesised across a cut.

Crucially, the shot list is measured on the **decoded** stream — the same cadence
chain (`CadencePlan`) the encoder and the model use — so `Shot::start_frame`
means "the Nth frame the model will see". On a 3:2 telecine source the raw and
decoded numbering differ by 20%, and a shot list that is off by 20% protects
nothing.

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
* A model backend that answers `SR_ERR_OUT_OF_MEMORY` is not a failure: the
  executor tells the session about the smaller working set (or reports that the
  backend cannot apply it in place) and runs the *same segment* again. The
  reference plugin can inject that fault on demand, so the ladder is tested
  end to end rather than by inspection.
* A backend that fails outright **fails the job**. It is not silently replaced by
  a deterministic encode: the plan promised a frame rate and a level of detail
  that the fallback cannot produce, so the fallback's own QC stage would reject
  the result — with a message ("planned 48 fps, got 24") that hides the real
  reason. The recovery path is a re-run, which resumes from the committed chunks.
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

## GPU discovery

`gpu` probes **Vulkan first**. The loader is opened at runtime with `libloading`
(no SDK, no import library), the instance is created, and every physical device is
read for `vendorID`, `deviceID`, `deviceType`, driver strings and — through
`VK_EXT_memory_budget` — the per-heap budget and usage the driver will hand out.
Vendor tools (`nvidia-smi`, `rocm-smi`) are merged in afterwards to sharpen a row
with the one thing Vulkan cannot say: what *other* processes are holding.

The order used to be the other way round, which meant Intel was never seen, an AMD
card on Windows (where `rocm-smi` does not exist) was never seen, and the number
the planner actually needs was missing on two of three vendors.

Two details decide whether an unattended run OOMs:

* `heapUsage` is *this process's* usage, not the machine's. Taking it at face value
  reports 14.9 GiB free on a card whose desktop is already holding 3.4 GiB. The
  planner takes the **most pessimistic** of `budget - process_usage` and
  `total - system_usage`.
* An integrated GPU's "device-local" heap is system RAM. It is reported as shared,
  excluded from `free_mib`, and never chosen as the device a model runs on.

The backend's own device list wins over the probe when a backend is installed: it
is the thing that will actually allocate, and its numbering need not agree with
the probe's about which device is number zero.

## The inference ABI

`include/sr_infer.h` defines ABI **v2**: device enumeration (`sr_infer_devices`),
session lifecycle (`sr_infer_open` / `sr_infer_close`), capability query
(`sr_infer_query`), execution (`sr_infer_execute`), async fences
(`sr_infer_poll`) and structured errors (`sr_infer_last_error`). A job carries an
N-frame temporal window, an op (`restore` / `interpolate` / `scale`), a dtype and
layout, a tile size, a rational timestamp per frame, a chunk id and a seed. Every
struct starts with its own `struct_size`, so the ABI can grow without breaking
2.0 binaries.

Version 1 -- one call, two 8-bit frames in, one frame out -- is rejected at load
time with a message that says to recompile against the header. It was a
prototype: it could not express a temporal window, a device or a memory budget,
so a plugin could be *loaded* with no way to *run a model*.

A plugin can be backed by Vulkan compute, DirectML, OpenVINO, MIGraphX, TensorRT,
ncnn or a CPU kernel; the engine only asks what it can do and hands it frames.
`crates/sr-infer-plugin-example` is a reference shared library implementing that
ABI with **no dependency on `sr-core`** (it keeps its own copies of the structs, so
a drift between the header and the engine's bindings fails a test instead of
misreading memory).

`crates/sr-infer-gpu` is the same contract implemented on Vulkan with WGSL kernels
compiled by naga at runtime — no shader compiler, no SDK, and Vulkan only, because
a backend that calls itself Vulkan and quietly runs on DX12 would be the kind of
claim this project exists to stop making. It is a **test dependency** of `sr-core`:
the shipped engine has no GPU API dependency at all, and the engine's own test
suite loads the backend as a plugin, exactly as a user would.

Its quality is measured, not asserted:

| check | result |
|---|---|
| WGSL kernels vs an independent Rust implementation | bit-identical on every pixel |
| NVIDIA vs AMD driver, same input | bit-identical |
| rigid translation vs the hidden middle frame | interior PSNR infinite (exact); frame duplication: 13.8 dB |
| ABI contract: every output slot written | checked, after a version filled only the synthesised slots |

That last row is the instructive one. The backend's interpolation was correct, the
frame count was correct and the project's own checks passed — while every
pass-through frame in the file was blank, because the plugin wrote only the slots
it synthesised. Only looking at the pixels of the committed chunk caught it, which
is why `tests/gpu_execution.rs` decodes the intermediate and measures it instead of
trusting the log.

Engines are ranked and selected: plugin (model) > `minterpolate`
(motion-compensated, FFmpeg) > deterministic resample. The baseline always
exists, so the pipeline always has a correct answer.

### What executes the model

Selection is not execution. `pipeline::segments` turns the shot list into an
exact sequence of model calls and output slots; `pipeline::native` drives it:

```text
shots ──► segments ──► chunks ──► decode ─► session ─► chunk file ─► concat ─► mux
           (never      (resume                  (sr_infer_execute)
            across      unit)
            a cut)
```

A segment covers input frames `[first..=last]` and emits output slots
`m*first..=m*last`; consecutive segments overlap by one frame and drop the slot
the previous one already wrote, so the counts telescope to exactly `m*(n-1)+1`
regardless of how the work was split. A segment containing a shot boundary or a
dissolve guard is `Hold`: frames are repeated and **the model is not called at
all**. That is what makes "nothing is synthesised across a cut" a property of the
data flow rather than a promise to a filter.

`minterpolate` is never used to satisfy a request for model interpolation: if the
plan says `InterpolationMethod::Plugin` then the executor is
`VideoExecutor::NativeInference`, and the two are cross-checked by tests. If no
plugin is installed the plan downgrades *loudly* to frame duplication, with a
warning in the plan and `ffmpeg-single-pass` in the executor row -- it does not
quietly run a different algorithm.

## Stage map

| stage | what happens | checkpoint |
|---|---|---|
| Probe | ffprobe → typed manifest (streams, colour, SAR/DAR, chapters, attachments) | yes |
| Temporal | `idet` sampling → progressive/telecine/interlaced/mixed | yes |
| Scenes | native shot detection → shots with rational timestamps, measured on the **decoded** cadence | yes |
| AudioAnalysis | `ebur128` + native BS.1770 + dialogue detection → LDR decision | yes |
| Plan | executor, geometry, encoder chain, VRAM budget, working set, reasons and warnings | yes |
| Restore / Interpolate | a model session per job; per-segment calls; OOM walks the degrade ladder | per chunk |
| Regrain | grain measured per shot from the source, then applied per chunk | per chunk |
| AudioProcess | dialogue rider + band duck → float WAV on disk (never in RAM) | chunk row |
| Encode | native path: one encode per chunk + concat; deterministic path: one FFmpeg pass | per chunk |
| Mux | concatenated video + enhanced track + originals + subtitles + attachments + chapters | progress events |
| Qc | 9-11 checks comparing the output with the plan | yes |

| executor | Restore / Interpolate / Regrain |
|---|---|
| `ffmpeg-single-pass` | folded into the encode pass, with honest notes about what is *not* happening |
| `native-inference` | decode → `sr_infer_execute` per segment → per-chunk encode → concat → mux |

## Memory

A two hour 5.1 feature is ~8 GB as float32, so nothing buffers a film: analysis
consumes blocks as they arrive, and the remaster streams into a temporary WAV on
disk that is deleted on success (or kept with `--keep-intermediates`).
