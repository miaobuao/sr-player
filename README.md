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
cargo test --workspace                     # 217 tests: 207 unit + 3 end-to-end + 1 model-execution + 1 gpu-execution + 5 gpu kernels
powershell -File testdata\make_fixture.ps1 # 8 s DVD-shaped fixture (the long-form check)
cargo run -p sr-cli -- analyze testdata\sample-dvd.mkv
cargo run -p sr-cli -- convert testdata\sample-dvd.mkv -o testdata\out.mkv
cargo run -p sr-cli -- probe   testdata\out.mkv
```

`crates/sr-core/tests/end_to_end.rs` builds its own fixture with FFmpeg and runs
the real pipeline over it, so `cargo test` alone proves the central claim: a
video goes in, a correctly converted one comes out, the job emits exactly one
`Finished` event, the output keeps its audio/subtitle/chapter streams, a dry run
writes nothing, and a missing input fails as a job rather than a panic. It skips
itself with a message when FFmpeg is not installed.

`crates/sr-core/tests/native_execution.rs` is the one that keeps the AI path
honest. It points the engine at the reference plugin and checks what is only true
when frames really went through it: the plugin logged a call (it can only do that
from inside `sr_infer_execute`), **no call's frame range contained a cut**, the
output has exactly `2×(n−1)+1` frames at twice the source rate, the output
contains distinct frames the source never had (so it is not a duplication
fallback), an injected out-of-memory fault is answered by the degrade ladder
instead of failing the job, and a second run reuses the committed chunks without
calling the model again.

The `testdata` fixture is deliberately awkward: 720×480 MPEG-2 at 29.97 with two
hard cuts, a 5.1 AC-3 track (dialogue in the centre channel) plus a second
stereo track, a subtitle track, chapters and a font attachment. A verified run
reports 3 shots / 2 cuts, and the output keeps the enhanced 5.1 track **plus
both original tracks**, the subtitle, the attachment and both chapters.

## Deliberate limitations

These are stated rather than hidden:

* **No learned model ships.** The plugin ABI (v2) can carry RIFE-class
  interpolation and SeedVR2-class restoration — sessions, devices, multi-frame
  windows, dtypes, tiling, memory budgets, out-of-memory feedback — and the
  executor that drives it is real, as is the Vulkan backend. What the GPU backend
  implements is *search-based* motion compensation, not a network: it finds flow
  instead of learning it, so it cannot invent detail and it does not restore.
  The in-tree reference plugin is a test instrument (blend + unsharp mask) whose
  job is to keep the ABI honest and the degrade ladder testable. Without a plugin
  the engine is deterministic: it resamples, corrects cadence, remasters audio and
  encodes, and invents nothing.
* **The model path transfers frames through host memory.** The ABI has a
  device-handle path (`SR_MEM_DEVICE`) for zero-copy, but the executor hands over
  host buffers today, so each call pays two uploads and a download. Fine for a
  preprocess-and-watch run; not how you would stream a two-hour feature at speed.
* **Lanczos still does the final resize.** A model that restores at source
  resolution is not a super-resolution model; the deterministic upscale to the
  target raster is labelled as such in the plan.
* **Re-grain is off by default.** Grain costs bitrate and fights the encoder;
  the per-shot grain estimator is not part of this build, so the knob exists but
  is not guessed at.
* **Black-bar cropping is not implemented.** Bars are still detected by nothing
  and cropped by nothing.
* **HDR sources bypass the SDR path** (with a warning) rather than being
  tone-mapped silently.
* **5.1 dialogue uses the centre channel**, not source separation. There is no
  DX/MX/FX model in this build; when the detector is unsure it says so.
* **PAL speed-down is never automatic** (25 → 24 fps would change duration,
  pitch, subtitles and chapters; low-confidence automatic changes are worse than
  no change).
* **Black-bar cropping is not implemented.** Bars are still detected by nothing
  and cropped by nothing.
* **HDR sources bypass the SDR path** (with a warning) rather than being
  tone-mapped silently.
* **5.1 dialogue uses the centre channel**, not source separation. There is no
  DX/MX/FX model in this build; when the detector is unsure it says so.
* **PAL speed-down is never automatic** (25 → 24 fps would change duration,
  pitch, subtitles and chapters; low-confidence automatic changes are worse than
  no change).
