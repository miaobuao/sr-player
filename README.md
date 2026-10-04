# sr-player

Unattended film restoration: a portable Rust media engine plus a GPUI front end
that picks a file, runs the conversion, and shows progress and logs.

**There is no video playback in this application.** Playback stays with whatever
player you already like; `sr-gui` ends a job with *open folder* and *play with
the system player* buttons.

## What it actually does

```text
probe → classify cadence → detect shots → analyse audio → build plan
      → remaster audio → encode + mux (one FFmpeg pass) → quality control
```

Every stage checkpoints to SQLite, so an interrupted job resumes instead of
starting over, and every stage degrades instead of failing:

| failure | response |
|---|---|
| an encoder is advertised but will not open (AMD AMF on an NVIDIA box) | fall back to the next encoder in the chain |
| out of memory | walk the degrade ladder (block swap → VAE tile → offload → temporal batch) |
| the cadence classifier is unsure | deinterlace selectively, never drop frames |
| the dialogue detector is unsure | reduce the correction in proportion to confidence |
| loudness/dialogue ratio is already ≤ 5 LU | change nothing (EBU R128 S4) |
| a stage result is missing on resume | recompute just that stage |

## Layout

| crate | role |
|---|---|
| `crates/sr-core` | the media engine. Probing, rational timeline, cadence classification, shot detection, BS.1770 + dialogue analysis, native DSP, pipeline scheduler, SQLite state, inference ABI |
| `crates/sr-gui` | GPUI front end. File picker, run controls, stage ladder, progress, logs, media/plan/engine panels. Contains no media logic |
| `crates/sr-cli` | headless driver: `probe`, `analyze`, `convert`, `jobs`, `log`, `engines`, `profiles` |
| `crates/sr-infer-plugin-example` | a reference shared library implementing the `sr_infer` C ABI, used as a test instrument |
| `crates/sr-infer-gpu` | a Vulkan backend implementing that ABI: block-matching motion-compensated interpolation in WGSL |
| `include/sr_infer.h` | the plugin ABI as C |

Dependency direction is one-way: `sr-cli`/`sr-gui` → `sr-core` → FFmpeg. The
engine never depends on a UI, and the UI never touches media.

## Requirements

* **FFmpeg** (any recent build; `ffmpeg` and `ffprobe` on `PATH`, or set
  `SR_FFMPEG` / `SR_FFPROBE`). This is the only hard dependency.
* Rust 1.85+ (developed on 1.98, stable-msvc).
* **No Python. No PyTorch. No CUDA. No vendor SDK.** A model backend is an
  optional accelerator behind a C ABI, never a correctness dependency.
* **Vulkan is optional too.** It is used for two things: probing the machine's
  GPU (through the loader, at runtime, with no SDK) and running the GPU backend,
  which is a plugin you can simply not install. A machine without Vulkan runs the
  deterministic path and says so.

## The GPU backend

`crates/sr-infer-gpu` is a real backend, not a stub: WGSL compute kernels, driven
through the ABI, doing **block-matching motion-compensated interpolation**. For
every 8x8 block it searches the displacement that best matches the two frames,
warps both frames along half of it, and blends — preferring the nearer sample
where the two disagree strongly, because averaging an occlusion is what makes a
ghost.

```powershell
cargo build --release -p sr-infer-gpu
$env:SR_INFER_PLUGIN = "target\release\sr_infer_gpu.dll"
cargo run -p sr-cli -- convert "D:\films\movie.mkv" -o out.mkv
```

What it is *not*: a learned model. There are no weights, so it cannot invent
detail, and it does not restore — `sr_infer_query` advertises interpolation only,
which is why `--profile safe-16gb` will not claim a restoration that did not
happen. Its quality is measured rather than asserted: a rigid translation is
reconstructed **bit-exactly** (interior PSNR infinite, versus 13.8 dB for frame
duplication), the shader agrees with an independent Rust implementation
bit-for-bit, and on a machine with two GPUs both drivers produce identical
output.

## Quick start

```powershell
cargo build --release
cargo run -p sr-cli -- profiles
cargo run -p sr-cli -- analyze  "D:\films\movie.mkv"
cargo run -p sr-cli -- convert  "D:\films\movie.mkv" -o "D:\films\movie.restored.mkv"
cargo run -p sr-gui
```

Useful flags: `--profile deterministic|safe-16gb|preview`, `--dry-run`
(analyse and plan only), `--interpolate off|duplicate|minterpolate|plugin`,
`--chunk-encoding ffv1|direct`, `--no-audio`, `--quality N`, `--no-resume`,
`--keep-intermediates`.

## Two video executors

The plan names the executor before a pixel moves, and the log repeats it, because
these are not the same product:

| executor | what runs | when |
|---|---|---|
| `ffmpeg-single-pass` | one FFmpeg pass with a filter chain | no model plugin is installed, or the profile asks for no model work |
| `native-inference` | decode → **model session** → per-chunk encode → concat + mux | a plugin is installed and the profile asks for restoration or model interpolation |

`native-inference` is the path the project exists for. It is not an FFmpeg filter
graph with a nicer name:

* frames are decoded to interleaved RGB on a pipe, pushed through the plugin's
  session (`sr_infer_execute`), and the results are encoded;
* work is split into **shots** from the shot list, then into **chunks** of about
  1000 input frames. A frame pair that straddles a cut is never handed to the
  model — it is repeated instead — so "nothing is synthesised across a cut" is a
  property of the data flow, not a promise to a filter;
* every chunk is committed to the `chunks` table with its artifact before the
  next one starts, so a job that dies at 95% resumes at the chunk that failed;
* the frame count is exact (`multiplier × (n−1) + 1`) and the executor **fails**
  if what it wrote disagrees with the plan.

## Verification

```powershell
cargo test --workspace                     # 280 tests, 2 ignored (see Status below)
powershell -File testdata\make_fixture.ps1 # 8 s DVD-shaped fixture (the long-form check)
cargo run -p sr-cli -- analyze testdata\sample-dvd.mkv
cargo run -p sr-cli -- convert testdata\sample-dvd.mkv -o testdata\out.mkv
cargo run -p sr-cli -- probe   testdata\out.mkv
```

The corpus tests build their own fixtures with FFmpeg and assert what the *engine*
did with them, not what the code looks like it would do:

| file | what only it proves |
|---|---|
| `end_to_end.rs` | a video goes in and a converted one comes out; one `Finished` event; streams preserved; a dry run writes nothing |
| `native_execution.rs` | frames really went through the model: no call's range contained a cut, `2×(n−1)+1` frames, frames the source never had, an injected OOM answered by the degrade ladder |
| `resume.rs` | an interrupted job reuses its committed chunks, recomputes exactly the rest, and produces a frame-identical film |
| `gpu_execution.rs` | the engine drives the Vulkan backend over a real file and reports its capabilities truthfully |
| `restore.rs` | `SR_OP_RESTORE` executes through the ABI and every output slot is written |
| `cadence_corpus.rs` | a real 3:2 pulldown is classified telecine, inverse telecine gives back the frame count, the chain reaches 95 frames at 47.952 |
| `corpus_formats.rs` | anamorphic pixels are squared before the model sees them; true interlacing is deinterlaced and never decimated; PAL stays 25 fps |
| `corpus_timing.rs` | a variable frame rate keeps its runtime; a source starting at ten seconds comes out starting at zero |
| `grain_execution.rs` | per-shot grain reaches the encoder: the heavy shot returns to its source amplitude |
| `audio_channels.rs` | the rider's channel plan, per channel, on a 5.1 file |

The `testdata` fixture is deliberately awkward: 720×480 MPEG-2 at 29.97 with two
hard cuts, a 5.1 AC-3 track (dialogue in the centre channel) plus a second
stereo track, a subtitle track, chapters and a font attachment. A verified run
reports 3 shots / 2 cuts, and the output keeps the enhanced 5.1 track **plus
both original tracks**, the subtitle, the attachment and both chapters.

## Status: what works, and what is scaffold

Most of this project is machinery whose *effect* has not been demonstrated, and the
distinction matters more than any single feature. A capability can be fully
implemented, fully tested, and still produce no improvement, because what decides
picture quality is the weights — and there are none.

| capability | mechanism | what it does today | evidence |
|---|---|---|---|
| Plugin path runs decode → model → encode | yes | runs, per chunk, with resume | `native_execution.rs`, `resume.rs` |
| Inference ABI v2 | yes | sessions, devices, windows, tiling, OOM, fences | `gpu_execution.rs`, `restore.rs` |
| RIFE-class interpolation | **the graph, operators and checkpoint loader** | **nothing: no checkpoint ships** | `ifnet.rs`, `ifnet_gpu.rs` |
| Residual restoration | **the graph and the sub-pixel operator** | **nothing: the scaffold is an identity upscaler** | `restore.rs` |
| Learned backend reaches the engine | yes — `SR_OP_RESTORE` advertised | the engine will select it; the result is unchanged pictures | `gpu_execution.rs` |
| Interpolation that invents detail | no — the GPU backend is a block matcher | smooths motion, cannot invent | `motion.rs` |
| The final resize | no — Lanczos, labelled as such in the plan | resizes | `corpus_formats.rs` |
| Per-shot grain | **measured and applied per chunk** | **verified on the heavy shot only** (below) | `grain_execution.rs` |
| Resume after a clean interruption | yes | completed chunks are reused, output is frame-identical | `resume.rs` |
| Resume after a `kill -9` | not verified | the per-chunk commit should survive it; nothing proves it | — |
| Dialogue rider on 5.1 | yes — centre-channel semantics | acts on the centre, leaves the LFE bit-exact | unit tests in `audio::rider`; the end-to-end test is ignored |
| QC accepts a late-starting source | yes | normalises the offset instead of reporting a failure | `corpus_timing.rs` |
| Cadence: telecine, interlaced, PAL | yes | 60 → 48 → 95 frames at 47.952 on a real pulldown | `cadence_corpus.rs` |

The two rows in bold are where "implemented" and "works" come apart, and they are the
rows that decide whether this is a restoration system or a framework with a slot. The
graphs are real — a three-level coarse-to-fine interpolator, a residual restorer with
sub-pixel upsampling, both verified on the device against invariants that a wrong
implementation cannot satisfy — and neither does anything useful without weights.

**Nothing in the tree should be read as "the quality chain is done."** Two tests are
ignored, both with their reasons in the code: a one-frame flash is still detected as a
shot cut, and the audio channel-plan measurement contradicts its own evidence.

## Deliberate limitations

Stated rather than hidden:

* **No learned model ships.** The ABI carries RIFE-class interpolation and
  SeedVR2-class restoration, the executor that drives it is real, the Vulkan backend
  is real, and the network machinery — checkpoint format, operators, coarse-to-fine
  topology, sub-pixel upsampling, device execution — is built and tested. There are
  no weights. The GPU backend's *interpolation* is search-based motion compensation
  and its *restoration* is an identity upscaler until a checkpoint is supplied.
  Without a plugin the engine is deterministic: it resamples, corrects cadence,
  remasters audio and encodes, and invents nothing.
* **The model path transfers frames through host memory.** The ABI has a
  device-handle path (`SR_MEM_DEVICE`) for zero-copy, but the executor hands over
  host buffers today, so each call pays two uploads and a download. Fine for a
  preprocess-and-watch run; not how you would stream a two-hour feature at speed.
* **Lanczos still does the final resize.** A model that restores at source
  resolution is not a super-resolution model; the deterministic upscale to the
  target range is labelled as such in the plan, and "demote Lanczos to a fallback" is
  not something that can be done before there is something to fall forward to.
* **Re-grain is off by default, and one measurement contradicts its own result.**
  The per-shot estimator runs, its amplitudes reach the per-chunk encoder, and the
  heavy shot of the test fixture comes back at its source grain. The quiet half of
  the same run measured *lower* with re-grain than without, which adding noise cannot
  cause; that is recorded in `grain_execution.rs` as an open question rather than
  asserted away.
* **Only the per-chunk executor can vary a filter across a film.** FFmpeg's `noise`
  filter takes one constant, so the single-pass path applies one strength to
  everything and the plan says so.
* **Black-bar cropping is not implemented.** Bars are detected by nothing and
  cropped by nothing.
* **HDR sources bypass the SDR path** (with a warning) rather than being tone-mapped
  silently.
* **5.1 dialogue uses the centre channel**, not source separation. There is no
  DX/MX/FX model in this build; when the detector is unsure it says so.
* **PAL speed-down is never automatic** (25 → 24 fps would change duration, pitch,
  subtitles and chapters; a low-confidence automatic change is worse than none).
* **A one-frame flash is treated as a shot cut.** Nothing interpolates across a shot
  boundary, so this costs interpolation at every flash and multiplies the chunk count
  — a fade to white would fragment a take. The flash guard is not written.

