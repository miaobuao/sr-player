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

### Open: the pre-run estimate is 80% wrong

`429 frame(s) will be synthesised` and `214 call(s)` against an actual 239
intermediates. The output is correct, so it is the planner's estimate rather than
the executor's arithmetic — but a figure that overstates the work by that much is
one nobody will trust.
