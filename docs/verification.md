# Verification record

What has actually been run, on what hardware, and what came out. Kept separate
from the design documents because those describe intent and this describes
results — a distinction that stops being obvious about three weeks in.

Reference machine throughout: RTX 5070 Ti, driver 617.14, 15227 MiB heap budget,
Vulkan 1.4.351. Second device is an AMD integrated part with 30746 MiB of *shared*
memory, which is why it is never offered to a model.

## Phase 1 — the native runtime, on real weights

`native/sr-native`, four tests, all passed. Reproduce with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File native\sr-native\setup-third-party.ps1
cmake -S native/sr-native -B native/sr-native/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/sr-native/build
```

| test | what it establishes |
|---|---|
| `device_test` | ncnn sees both devices through Vulkan; the integrated one is flagged as unified memory |
| `restore_image_test` | Real-ESRGAN x4plus runs in process; output differs from a bilinear 4x upscale by 30.28 levels, so a network ran; tiled and untiled agree to 5.16 levels |
| `warp_unit_test` | the vendored `Warp` layer against an independent bilinear implementation: identity, integer shifts on both axes, fractional flow — **bit-exact, worst error 0.0000** |
| `rife_pair_test` | RIFE 4.25 against exactly-computed ground truth |

`rife_pair_test` in detail:

```
constant image         -> constant, byte-exact (min 128 max 128 mean 128.00)
identical frames       -> 0.002 levels
2 px shift, t = 0.50   -> 0.303 levels   (a blend scores 0.277)
8 px shift, t = 0.50   -> 0.974 levels   (a blend scores 2.919)
4 px shift, t = 0.25   -> 1.125 levels   (a blend scores 3.773)
```

The first line is the one that took twelve rounds to earn. A constant image warped
by any flow is still that constant, so that case is insensitive to the flow field
and isolates the warp, the readback and the input range. It passed byte-exactly
while the flow cases were still failing, which is what proved the fault was in the
network's input range and nothing structural.

## Phase 2 — interpolation through the product

```powershell
sr-cli convert testdata\sample-dvd.mkv -o <fresh path>.mkv `
    --profile deterministic --interpolate rife --no-resume
```

```
[interpolate] RIFE 4.25 open on NVIDIA GeForce RTX 5070 Ti (device 0, 14.9 GiB),
              ensemble off, 2x film mode
[plan] 240 input frame(s) to 479 output frame(s) in 3 chunk(s), 218 segment(s)
```

On the artifact:

| property | value | expected |
|---|---|---|
| video frames | 479 | `2*(240-1)+1` = 479 |
| geometry | 1620x1080 | target canvas |
| cadence | ~59.94 fps | 2x 29.970 |
| duration | 8.005 s | source 8.01 s |
| streams | av1, flac 6ch, ac3 6ch, ac3 2ch, subrip, ttf attachment | all preserved |
| full decode | clean | no errors |
| adjacent frames | **0 of 11 identical** | a duplicating pass would show ~half |

That last row is the one that matters. A pass that repeated frames instead of
synthesising them would still produce 479 frames and the right duration; only
comparing consecutive frames distinguishes the two.

The plan states the cut guarantee in its own words: *"2 cut(s) in this file are
shot boundaries the interpolation is never allowed to cross"*.

## Phase 3 — restoration, and both models together

```powershell
sr-cli convert testdata\sample-dvd.mkv -o <fresh path>.mkv `
    --profile safe-16gb --interpolate rife --no-resume
```

```
[interpolate] RIFE 4.25 open on NVIDIA GeForce RTX 5070 Ti (device 0, 14.9 GiB), ensemble off, 2x film mode
[restore]     restoration open: 4x on NVIDIA GeForce RTX 5070 Ti (720x480 -> (2880, 1920) source raster 720x480)
[plan]        240 input frame(s) to 479 output frame(s) in 3 chunk(s), 218 segment(s)
```

| property | value | expected |
|---|---|---|
| video frames | 479 | `2*(240-1)+1` |
| geometry | 1620x1080 | target canvas |
| duration | 8.005 s | source 8.01 s |
| streams | av1, flac 6ch, ac3 6ch, ac3 2ch, subrip, ttf | all preserved |
| adjacent frames | **0 of 12 identical** | a duplicating pass would show ~half |
| wall clock | 749 s | 240 restores, 239 syntheses, 3 chunk encodes |

### Restoration is a network, not a resampler

The check that matters, and the one that would catch a regression to Lanczos: take
the same source frame, upscale it to 1620x1080 with Lanczos, and compare the
compressed size of that against the pipeline's output frame.

| frame | PNG bytes |
|---|---|
| Lanczos 1620x1080 | 37,191 |
| restored, after interpolation | 791,152 |

A 21x difference. A resampler cannot invent high-frequency detail, so it compresses
to almost nothing; a network that synthesises it does not. `restore_image_test`
measures the same property in isolation (30.28 levels from a bilinear upscale), and
this confirms it survives the whole product path.

### Efficiency, since acted on

Restoration produced 2880x1920 (4x of the 720x480 source) and the resize to the
1620x1080 canvas happened **after** RIFE, so the interpolator ran on 3.2x the pixels
the deliverable needs. That was fixed in two steps -- first moving the resize between
restoration and interpolation, then choosing the model's input size from the canvas
so it lands on the canvas directly:

    749.4 s  ->  617.9 s  ->  414.3 s

The first step was adopted on a measurement that turned out to be noise (see below);
the second on a 33% wall-clock reduction, with no quality claim attached, because the
proxy that would have supported one is not valid.

## Restoration QC, and a result that needs a product decision

Both fixtures below are built the same way: a known 1620x1080 original, downscaled
to 720x480 with Lanczos, then put through `--profile safe-16gb --interpolate off`.
Fidelity is PSNR against the original; "detail" is the mean PNG size of five frames,
which measures high-frequency content and **not** correctness.

### Synthetic fixture (testsrc2 blended with mandelbrot)

| | PSNR vs original | detail (mean PNG) |
|---|---|---|
| Lanczos upscale | 37.05 dB | 864,362 |
| the pipeline | 33.08 dB | 5,297,118 |

### Photographic fixture (a real 1620x1080 photograph)

| | PSNR vs original | detail (mean PNG) |
|---|---|---|
| original | — | 1,762,600 |
| Lanczos upscale | 33.69 dB | 1,174,080 |
| the pipeline | 28.48 dB | 4,971,358 |

### The model's input size, which is the decision this table settles

An earlier round handed the model `canvas / scale` instead of the source — 404x270
rather than 720x480 — because that lands exactly on the canvas and is 33% faster. It
was adopted on a wall-clock measurement with the quality claim explicitly withdrawn.
Measured properly, on two independent photographs:

| fixture | Lanczos | input = canvas/scale | input = source |
|---|---|---|---|
| photograph 1 | 33.69 dB | 28.48 dB | **30.16 dB** |
| photograph 2 | 39.03 dB | 35.66 dB | **38.45 dB** |

The reduction costs 1.7 to 2.8 dB and **has been reverted**. It happens before the
model, and a model cannot recover information thrown away before it saw the frame;
it responds by inventing more, which the detail measurement confirms (4,971,358 PNG
bytes against 4,770,727). The policy is now a function with a test, so changing it
again means deleting a number rather than editing an expression.

### What this says

**On both fixtures restoration scores several dB below a plain Lanczos upscale on
PSNR, while carrying several times the detail — including more detail than the
original itself.** 4,971,358 bytes against the photograph's own 1,762,600 is not
recovery of detail; it is detail that was not there to recover.

That is what a generative restorer does, and PSNR is known to punish exactly the
sharpening that makes such output look better to a person. So this is **not** a
verdict that the model is wrong. It is a statement that:

* **the two objectives are in tension and the product currently chooses one without
  saying so.** `safe_16gb` applies restoration unconditionally. On a clean,
  well-mastered source that has merely been downscaled, the model is being asked to
  restore degradation that is not present, and it obliges.
* **the sharpness proxy used in an earlier round was not valid.** It can rise sixfold
  while fidelity falls four decibels. That claim has been withdrawn from the commit
  it appeared in.
* **this is the input Phase 3's "automatic restoration/scale decisions" needs.** A
  decision about *whether* to restore, not only at what scale, requires either a
  degradation estimate or an acceptance that restoration is a stylistic choice.

Settling which of those is right needs the AI visual QC in the completion bar. The
numbers above are a specification for it, not a substitute.

## Defects found by doing the work above

### Fixed: a stored plan outlived the options that shaped it

The job identity was a hash of the input and output paths. Stage results — and
**the plan is a stage result** — are committed to the store and restored on a
resumed run, so changing an option and re-running the same output path silently
restored the plan built for the old options. `--interpolate rife` reported `off`,
and went on reporting `off`, because the first run under that output path had
committed a plan with interpolation disabled. That is how a correct implementation
looked broken for two rounds.

The explanation first written here — a cached `plan_json` column — was wrong.
`plan_json` is written and never read; the real mechanism is `load_or`'s stage
results. Worth recording because the wrong explanation was plausible enough to
have been left standing.

The options are now part of the identity, hashed alongside the paths. The cost is
that changing an option starts a new job and discards committed chunks, which is
the right trade: a chunk built under different settings is not a valid answer to
the new question.

### Fixed: the pre-run workload estimate

It read `429 frame(s) will be synthesised ... in 214 call(s)`. Two errors in one
line: the frame count summed `emitted()` over Synthesise segments, which is every
frame the segment *outputs* including the real ones, and the call count counted
segments rather than calls. At 2x a call produces one frame, so those two numbers
could not both be right — and that contradiction is the only reason it was ever
noticed. A pair of figures that merely looked plausible would still be there.

It now reads `214 frame(s) ... in 214 call(s)`: one call, one frame, and 214 of the
239 pairs in the file, the remaining 25 held by the two cuts and their guards.
A unit test recomputes both figures from the segment geometry and asserts the two
agree at 2x, which is what makes the contradiction impossible to reintroduce.
