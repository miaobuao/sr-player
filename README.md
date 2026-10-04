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
| `crates/sr-infer-plugin-example` | a working shared library implementing the `sr_infer` C ABI, loaded by `sr-core`'s tests |
| `include/sr_infer.h` | the plugin ABI as C |

Dependency direction is one-way: `sr-cli`/`sr-gui` → `sr-core` → FFmpeg. The
engine never depends on a UI, and the UI never touches media.

## Requirements

* **FFmpeg** (any recent build; `ffmpeg` and `ffprobe` on `PATH`, or set
  `SR_FFMPEG` / `SR_FFPROBE`). This is the only hard dependency.
* Rust 1.85+ (developed on 1.98, stable-msvc).
* **No Python. No PyTorch. No CUDA. No vendor SDK.** A model backend is an
  optional accelerator behind a C ABI, never a correctness dependency.

## Quick start

```powershell
cargo build --release
cargo run -p sr-cli -- profiles
cargo run -p sr-cli -- analyze  "D:\films\movie.mkv"
cargo run -p sr-cli -- convert  "D:\films\movie.mkv" -o "D:\films\movie.restored.mkv"
cargo run -p sr-gui
```

Useful flags: `--profile deterministic|safe-16gb|preview`, `--dry-run`
(analyse and plan only), `--interpolate off|duplicate|minterpolate`,
`--no-audio`, `--quality N`, `--no-resume`, `--keep-intermediates`.

## Verification

```powershell
cargo test --workspace                     # 174 tests: 168 unit + 3 end-to-end + 3 plugin ABI
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

The `testdata` fixture is deliberately awkward: 720×480 MPEG-2 at 29.97 with two
hard cuts, a 5.1 AC-3 track (dialogue in the centre channel) plus a second
stereo track, a subtitle track, chapters and a font attachment. A verified run
reports 3 shots / 2 cuts, and the output keeps the enhanced 5.1 track **plus
both original tracks**, the subtitle, the attachment and both chapters.

## Deliberate limitations

These are stated rather than hidden:

* **No model backend ships.** Without a plugin the engine is deterministic: it
  resamples, corrects cadence, remasters audio and encodes — it never invents
  detail. `--interpolate minterpolate` gives real motion interpolation without a
  model, at a large speed cost.
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
