# Architecture

## The one boundary that matters

```
sr-gui / sr-cli        presentation: pick a file, render events
        │
        ▼
sr-core                all media work: probe, timeline, decode, analysis,
        │              DSP, scheduling, state, child processes
        │
        ├─────────────► FFmpeg      decode / filter / encode / mux
        │
        └─────────────► sr-native   the one AI runtime (C++ over ncnn, Vulkan)
                                    RIFE 4.25 + Real-ESRGAN x4plus, in-process
```

There is exactly one inference runtime in the production tree, and it is not
written in this repository's Rust. That is a deliberate reversal: this project
previously carried its own ncnn checkpoint parser, graph evaluator, tensor
scheduler and WGSL kernels, which made it a second-rate reimplementation of a
runtime that already exists, is faster, and supports three vendors' Vulkan
drivers. All of it is gone. What remains here is the part that is actually this
project's work: media correctness, automatic decisions, chunking and recovery,
audio, QC, and the refusal to claim a model ran when it did not.

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
  because the desktop and the driver are using the same card. The only working-set
  knob the two networks this project runs actually have is the restoration **tile
  edge**, so that is the whole ladder: `auto → 512 → 384 → 256 → 192 → 128`. It is
  a short ladder because it is an honest one. A tile size cannot free memory inside
  an FFmpeg encoder, so the encoder path does not offer the ladder as a pretend
  fix — an encoder OOM walks the encoder chain like any other encoder failure.
* A model stage that fails outright **fails the job**. It is not silently replaced
  by a deterministic encode: the plan promised a frame rate and a level of detail
  that the fallback cannot produce, so the fallback's own QC stage would reject
  the result — with a message ("planned 48 fps, got 24") that hides the real
  reason. The recovery path is a re-run, which resumes from the committed chunks.
* **Nothing is substituted for a model.** A request for restoration or
  interpolation that the runtime cannot serve is refused before a pixel moves, with
  a message naming what is missing. The two things this replaced — resampling with
  Lanczos and calling it restoration, and running `minterpolate` and calling it
  model interpolation — both produced a log that said one thing and a picture that
  had received another, which is the failure mode this whole design is arranged
  against.
* One model at a time. Restoration and interpolation are sequenced rather than
  overlapped, because two networks plus their activation buffers do not fit on a
  16 GB card that is also driving a desktop.

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

When the native runtime lands, its own device list becomes the authority for which
device a model runs on: it is the thing that will actually allocate, and its
numbering need not agree with this probe's about which device is number zero. The
Rust probe stays for the *planner* and the UI, where it is the only source of a
memory budget before any model is open.

## The one AI runtime

`native/sr-native` is a C++ layer over a commit-pinned ncnn, and it is the only
thing in the tree that executes a neural network. Its ABI is deliberately shaped
like the product instead of like a framework:

```c
typedef struct sr_context  sr_context;    /* one device, one Vulkan instance   */
typedef struct sr_rife     sr_rife;       /* RIFE 4.25, ensemble off           */
typedef struct sr_restorer sr_restorer;   /* RealESRGAN_x4plus                 */

int  sr_device_count(void);
int  sr_device_info(int index, sr_device_info_t* out);
sr_context*  sr_context_create(int device_index);
sr_restorer* sr_restorer_create(sr_context*, const char* model_dir);
int          sr_restorer_process(sr_restorer*, const sr_image*, sr_image*, int scale, int tile);
sr_rife*     sr_rife_create(sr_context*, const char* model_dir);
int          sr_rife_process(sr_rife*, const sr_image* prev, const sr_image* next,
                             float timestep, sr_image* out);
```

There is no `sr_tensor`, no `sr_graph`, no `sr_operator` and no
`sr_execute_graph`, because this program is not a tensor framework. It runs two
networks, and the ABI says so.

Three consequences are worth stating, because each one removes a class of bug:

* **RIFE takes a pair and a timestep.** It cannot be handed a window, so it cannot
  be handed a pair that straddles a cut: the executor simply never constructs one.
  The cut-safety guarantee is a property of the call shape rather than a filter
  setting that a future edit could drop.
* **Frames never become files.** Decode → `sr_image` → ncnn → `sr_image` → encode,
  all in memory. No PNG round trip, no child process per frame, no temporary
  directory, so there is no file lifetime to get wrong and no image codec in the
  quality path.
* **The model does not decide the output raster.** A model scale is a resampling
  factor; landing on the target size is a separate, deterministic resize, and it
  is labelled as such in the plan rather than presented as recovered detail.

The models themselves are pinned by hash in a manifest and are never
redistributed with the source. RIFE is fixed at **4.25 with `ensemble = false`**
— not `rife-HD`, not `4.6`, and not the heavier variants — because a single
frozen model is what makes "zero per-title tuning" a testable claim instead of a
menu.

### What the executor promises the runtime

`pipeline::segments` turns the shot list into an exact sequence of model calls and
output slots, and `pipeline::native` is the thing that will drive it:

```text
shots ──► segments ──► chunks ──► decode ─► model ─► chunk file ─► concat ─► mux
           (never      (resume
            across      unit)
            a cut)
```

A segment covers input frames `[first..=last]` and emits output slots
`m*first..=m*last`; consecutive segments overlap by one frame and drop the slot the
previous one already wrote, so the counts telescope to exactly `m*(n-1)+1`
regardless of how the work was split. A segment containing a shot boundary or a
dissolve guard is `Hold`: frames are repeated and **the model is not called at
all**.

The planner and its tests exist today and are exercised independently of the
runtime. What does not exist yet is the executor's model stage: a `Synthesise`
segment currently fails loudly, saying that the plan asked for synthesised frames
and that there is no interpolator in this binary, rather than filling them in some
other way. That error is the seam where `sr_rife_process` goes.

## Stage map

| stage | what happens | checkpoint |
|---|---|---|
| Probe | ffprobe → typed manifest (streams, colour, SAR/DAR, chapters, attachments) | yes |
| Temporal | `idet` sampling → progressive/telecine/interlaced/mixed | yes |
| Scenes | native shot detection → shots with rational timestamps, measured on the **decoded** cadence | yes |
| AudioAnalysis | `ebur128` + native BS.1770 + dialogue detection → LDR decision | yes |
| Plan | executor, geometry, encoder chain, VRAM budget, working set, reasons and warnings | yes |
| Restore / Interpolate | *not built*: a request for either is refused before a pixel moves; the segment plan it will consume is already computed and tested | — |
| Regrain | grain measured per shot from the source, then applied per chunk | per chunk |
| AudioProcess | dialogue rider + band duck → float WAV on disk (never in RAM) | chunk row |
| Encode | chunked: one encode per chunk + concat; single-pass: one FFmpeg pass | per chunk |
| Mux | concatenated video + enhanced track + originals + subtitles + attachments + chapters | progress events |
| Qc | 9-11 checks comparing the output with the plan | yes |

| executor | Restore / Interpolate / Regrain |
|---|---|
| `ffmpeg-single-pass` | folded into the encode pass, with honest notes about what is *not* happening |
| `chunked` | decode → per-chunk encode with a per-shot chain → concat → mux; the model stage drops in between decode and encode |

## Memory

A two hour 5.1 feature is ~8 GB as float32, so nothing buffers a film: analysis
consumes blocks as they arrive, and the remaster streams into a temporary WAV on
disk that is deleted on success (or kept with `--keep-intermediates`).
