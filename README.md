# sr-player

Unattended film restoration: a portable Rust media engine plus a GPUI front end
that picks a file, runs the conversion, and shows progress and logs.

**There is no video playback in this application.** Playback stays with whatever
player you already like; `sr-gui` ends a job with *open folder* and *play with
the system player* buttons.

> ### Read this first: the AI runtime is between two designs
>
> The self-built inference layer — a hand-written ncnn `.param`/`.bin` parser, a
> tensor graph evaluator, WGSL convolution/deconvolution/warp kernels, a generic
> `sr_infer` plugin ABI and its dynamic discovery — has been **deleted**. It was
> a second inference runtime that this project should never have owned: ncnn
> already is one, and the images it produced were never better than the FFmpeg
> path it was standing in front of.
>
> Its replacement is a single statically linked native runtime, `native/sr-native`
> (C++ over ncnn, Vulkan), exposing exactly two model tasks: RIFE 4.25
> interpolation and Real-ESRGAN x4plus restoration. **That runtime does not exist
> yet.** Until it does, a request for either task is **refused before a pixel
> moves** — it is not answered with `minterpolate`, and it is not answered with
> Lanczos. `RestorationProfile::deterministic()` and `preview` run the rest of the
> pipeline.

## What it actually does

```text
probe → classify cadence → detect shots → analyse audio → build plan
      → remaster audio → encode + mux → quality control
```

Every stage checkpoints to SQLite, so an interrupted job resumes instead of
starting over, and every stage degrades instead of failing:

| failure | response |
|---|---|
| an encoder is advertised but will not open (AMD AMF on an NVIDIA box) | fall back to the next encoder in the chain |
| out of memory in the encoder | treat it like any other encoder failure and walk the chain; a restoration tile size cannot free memory inside FFmpeg, so it is not offered as a pretend fix |
| the cadence classifier is unsure | deinterlace selectively, never drop frames |
| the dialogue detector is unsure | reduce the correction in proportion to confidence |
| loudness/dialogue ratio is already ≤ 5 LU | change nothing (EBU R128 S4) |
| a stage result is missing on resume | recompute just that stage |
| restoration or interpolation was requested but no runtime can run it | **refuse the job** and say what is missing |

## Layout

| path | role |
|---|---|
| `crates/sr-core` | the media engine. Probing, rational timeline, cadence classification, shot detection, BS.1770 + dialogue analysis, native DSP, pipeline scheduler, SQLite state |
| `crates/sr-gui` | GPUI front end. File picker, run controls, stage ladder, progress, logs, media/plan/runtime panels. Contains no media logic |
| `crates/sr-cli` | headless driver: `probe`, `analyze`, `convert`, `jobs`, `log`, `devices`, `profiles` |
| `native/sr-native` | *(next)* the one AI runtime: a product-shaped C ABI over a commit-pinned ncnn |
| `models/` | *(next)* downloaded checkpoints, never redistributed, hash-checked at load |

Dependency direction is one-way: `sr-cli`/`sr-gui` → `sr-core` → FFmpeg. The
engine never depends on a UI, and the UI never touches media.

## Requirements

* **FFmpeg** (any recent build; `ffmpeg` and `ffprobe` on `PATH`, or set
  `SR_FFMPEG` / `SR_FFPROBE`). This is the only external dependency the shipped
  binary has today.
* Rust 1.85+ (developed on 1.98, stable-msvc).
* **No Python. No PyTorch. No CUDA. No TensorRT. No ONNX Runtime. No wgpu.**
  When the native runtime lands it will bring exactly one thing with it: ncnn,
  statically linked.
* **Vulkan** is used for GPU probing (through the loader at runtime, no SDK) and,
  later, by ncnn. A machine without Vulkan runs the deterministic path and says so.

## The one AI runtime

The architecture is frozen, and this is the whole of it:

```text
FFmpeg decode ──► ncnn / Vulkan (in-process) ──► Rust re-grain ──► FFmpeg encode
                   ├── Real-ESRGAN x4plus   (restoration)
                   └── RIFE 4.25, ensemble off (interpolation)
```

There is no plugin discovery, no feature flag and no second backend — not even a
disabled one. The C ABI is shaped like the product rather than like a framework:

```c
int  sr_device_count(void);
int  sr_device_info(int index, sr_device_info_t* out);

sr_context*  sr_context_create(int device_index);
void         sr_context_destroy(sr_context*);

sr_restorer* sr_restorer_create(sr_context*, const char* model_dir);
int          sr_restorer_process(sr_restorer*, const sr_image* in, sr_image* out, int scale, int tile);
void         sr_restorer_destroy(sr_restorer*);

sr_rife*     sr_rife_create(sr_context*, const char* model_dir);
int          sr_rife_process(sr_rife*, const sr_image* prev, const sr_image* next, float timestep, sr_image* out);
void         sr_rife_destroy(sr_rife*);
```

No `sr_tensor`, no `sr_graph`, no `sr_execute_graph`. RIFE takes two frames and a
timestep; Real-ESRGAN takes a frame and a tile size. Frames move through memory,
never through a temporary PNG and never through a child process.

## Quick start

**The AI runtime is part of the build, so it has to be staged first.** A fresh
clone cannot `cargo build` until this has run once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File native\sr-native\setup-third-party.ps1
```

That downloads the pinned ncnn, generates a Vulkan import library from the system
loader, stages both sets of weights, and verifies every SHA256 against
`native/sr-native/pins.json`. `cargo build` then compiles and links the C++ itself.

There is deliberately no feature flag to skip it. A build that compiled the AI
stages out would still print "restoration" in its own logs while resampling with
Lanczos, which is the specific dishonesty this architecture exists to prevent.

```powershell
cargo build --release
cargo run -p sr-cli -- profiles
cargo run -p sr-cli -- devices
cargo run -p sr-cli -- analyze  "D:\films\movie.mkv"
cargo run -p sr-cli -- convert  "D:\films\movie.mkv" -o "D:\films\movie.restored.mkv"
cargo run -p sr-gui
```

`convert` uses the `safe-16gb` profile by default, which asks for both models —
so today it stops immediately with a message naming what is missing. Run it with
the configuration that exists:

```powershell
cargo run -p sr-cli -- convert "D:\films\movie.mkv" -o out.mkv --profile deterministic
```

Useful flags: `--profile deterministic|safe-16gb|preview`, `--dry-run`
(analyse and plan only), `--interpolate off|rife`, `--regrain <strength>`,
`--chunk-encoding ffv1|direct`, `--no-audio`, `--quality N`, `--no-resume`,
`--keep-intermediates`.

`--regrain` is the switch that selects the **chunked** executor, so it is also
what makes per-chunk resume reachable from the command line:

```powershell
cargo run -p sr-cli -- convert "D:\films\movie.mkv" -o out.mkv --profile deterministic --regrain 8
```

## Two video executors

The plan names the executor before a pixel moves, and the log repeats it, because
these are not the same product:

| executor | what runs | when |
|---|---|---|
| `ffmpeg-single-pass` | one FFmpeg pass with one filter chain | nothing needs to vary across the film |
| `chunked` | decode → per-chunk encode → concat + mux | a **measured per-shot re-grain** is in play, and later the model stage |

`chunked` is not an FFmpeg filter graph with a nicer name:

* frames are decoded to interleaved RGB on a pipe and each chunk is encoded by its
  own FFmpeg process with **its own filter chain**, which is the only way one shot
  can get a different grain strength from its neighbour — FFmpeg's `noise` filter
  takes one constant for a whole chain;
* work is split into **shots** from the shot list, then into **chunks** of about
  1000 input frames;
* every chunk is committed to the `chunks` table with its artifact before the next
  one starts, so a job that dies at 95% resumes at the chunk that failed;
* the frame count is exact (`multiplier × (n−1) + 1`) and the executor **fails**
  if what it wrote disagrees with the plan.

The segment planner also owns the rule the model stage will have to obey: **a
frame pair that straddles a cut is never handed to the model.** It is enforced by
never constructing that pair, not by asking a filter to notice a cut — which is a
property of the data flow rather than a promise. `segments.rs` tests it today.

## Verification

```powershell
cargo test --workspace
powershell -File testdata\make_fixture.ps1 # 8 s DVD-shaped fixture (the long-form check)
cargo run -p sr-cli -- analyze testdata\sample-dvd.mkv
cargo run -p sr-cli -- convert testdata\sample-dvd.mkv -o testdata\out.mkv --profile deterministic
```

The corpus tests build their own fixtures with FFmpeg and assert what the *engine*
did with them, not what the code looks like it would do:

| file | what only it proves |
|---|---|
| `end_to_end.rs` | a video goes in and a converted one comes out; one `Finished` event; streams preserved; a dry run writes nothing |
| `resume.rs` | an interrupted job reuses its committed chunks, recomputes exactly the rest, and produces a frame-identical film |
| `cadence_corpus.rs` | a real 3:2 pulldown is classified telecine, inverse telecine gives back the frame count, the chain reaches 95 frames at 47.952 |
| `corpus_formats.rs` | anamorphic pixels are squared; true interlacing is deinterlaced and never decimated; PAL stays 25 fps |
| `corpus_timing.rs` | a variable frame rate keeps its runtime; a source starting at ten seconds comes out starting at zero |
| `grain_execution.rs` | per-shot grain reaches the encoder: the heavy shot returns to its source amplitude |
| `audio_channels.rs` | the rider's channel plan, per channel, on a 5.1 file |

The `testdata` fixture is deliberately awkward: 720×480 MPEG-2 at 29.97 with two
hard cuts, a 5.1 AC-3 track (dialogue in the centre channel) plus a second
stereo track, a subtitle track, chapters and a font attachment. A verified run
reports 3 shots / 2 cuts, and the output keeps the enhanced 5.1 track **plus
both original tracks**, the subtitle, the attachment and both chapters.

## Status

**[`docs/status.md`](docs/status.md) is the current, honest assessment** — what is
verified with evidence, what is assumed, and what has never been attempted. Read it
before trusting anything below.

| capability | state | evidence |
|---|---|---|
| Cadence: telecine, interlaced, PAL | **works** — 60 → 48 → 95 frames at 47.952 on a real pulldown | `cadence_corpus.rs` |
| Per-chunk executor, resume, single-shot retry | **works** — frame-for-frame identical to an undisturbed run | `resume.rs`, `cli_resume.rs` |
| Per-shot grain measured and applied | **works on the heavy shot only** (below) | `grain_execution.rs` |
| Dialogue rider on 5.1 | acts on the centre, leaves the LFE bit-exact | `audio::rider` unit tests; the end-to-end test is `#[ignore]`d |
| QC accepts a late-starting source | normalises the offset instead of failing | `corpus_timing.rs` |
| Encoder chain fallback on a broken vendor encoder | falls through the chain | `runner.rs` |
| **RIFE 4.25 interpolation** | **works.** `ensemble = false`, on Vulkan, through the product | `rife_pair_test`; `docs/verification.md` |
| **Real-ESRGAN x4plus restoration** | **works**, in process, tiled | `restore_image_test`; `docs/verification.md` |
| Forced termination, late, with models | **works** — killed at 2 of 4 chunks, resumed reusing exactly those 2, byte-identical output | `cli_models.rs` |
| No Python / CUDA / TensorRT / PyTorch / ONNX | **verified against the binary**, not asserted | `docs/verification.md` |
| Vendor independence | **partial.** Runs on this machine's AMD integrated GPU; no Intel, no discrete AMD | `docs/verification.md` |
| Phase 4 audio: VAD, 5.1 rewrite, M/S stereo | **not built** | — |
| AI visual QC | **not built** | — |
| A full-length unattended film | **never attempted.** ~52× realtime | — |

**Nothing in this tree should be read as "the quality chain is done."** The models
run and are tested; whether the restoration stage is *worth running* is unresolved,
and the measurements say it may not be on clean sources — see `docs/status.md`.
One audio test is ignored, with its reason in the code: the audio channel-plan
measurement contradicts its own evidence.

## Deliberate limitations

Stated rather than hidden:

* **No model runs yet.** This is the current state of a deliberate rebuild, not a
  permanent property. Everything downstream of the model — per-shot re-grain,
  chunk checkpointing, the mux, QC — is built and tested and waiting for frames.
* **Only the chunked executor can vary a filter across a film.** FFmpeg's `noise`
  filter takes one constant, so the single-pass path applies one strength to
  everything and the plan says so.
* **Lanczos still does the final resize.** A restoration model enlarges by its own
  scale factor; landing on the *output* raster is a separate, deterministic resize,
  and it is labelled as such rather than presented as detail.
* **Re-grain is off by default, and one measurement contradicts its own result.**
  The per-shot estimator runs, its amplitudes reach the per-chunk encoder, and the
  heavy shot of the test fixture comes back at its source grain. The quiet half of
  the same run measured *lower* with re-grain than without, which adding noise
  cannot cause; that is recorded in `grain_execution.rs` as an open question rather
  than asserted away.
* **Black-bar cropping is not implemented.** Bars are detected by nothing and
  cropped by nothing.
* **HDR sources bypass the SDR path** (with a warning) rather than being tone-mapped
  silently.
* **5.1 dialogue uses the centre channel**, not source separation. There is no
  DX/MX/FX model in this build; when the detector is unsure it says so. The
  surround channels are also still lifted by an amount nobody has explained, which
  is why the end-to-end channel test is ignored rather than relaxed.
* **PAL speed-down is never automatic** (25 → 24 fps would change duration, pitch,
  subtitles and chapters; a low-confidence automatic change is worse than none).
