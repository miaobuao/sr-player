# sr-native

The one AI runtime. A C++ layer over a commit-pinned [ncnn](https://github.com/Tencent/ncnn),
exposing the C ABI in [`include/sr_native.h`](include/sr_native.h) to `sr-core`.

It runs exactly two networks:

| task | model | call |
|---|---|---|
| restoration | `RealESRGAN_x4plus` | `sr_restorer_process(in, out, scale, tile)` |
| interpolation | `RIFE 4.25`, `ensemble = false` | `sr_rife_process(prev, next, timestep, out)` |

## Status

**Built, and three of the four tests pass.** `native/sr-native` compiles against
the pinned ncnn and links; `device_test`, `restore_image_test` and
`warp_unit_test` pass on the reference machine with real weights.

`rife_pair_test` **fails**, and until it passes `sr-core` does not call any of
this: a request for restoration or interpolation is refused before a pixel moves
rather than answered with Lanczos or `minterpolate`. What is known about the
failure is in `pins.json` and in the test itself — the short version is that the
tensor entering the network has been verified exactly right, the one custom layer
is bit-exact against an independent implementation, the weights are verified
value-for-value against ncnn's own reader, and the network's flow still comes out
at ±150–430 px for two *identical* frames, where it must be zero.

## Setup

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File native\sr-native\setup-third-party.ps1
```

That builds `third_party/` and `models/` from `pins.json`, verifying every
SHA256. Then configure with CMake and Ninja from a shell where `vcvars64.bat` has
been called.

## Build prerequisites

ncnn's Vulkan backend needs the Vulkan **development** files, not just the
runtime: headers, a `vulkan-1` import library, and `glslangValidator` for its
compute shaders. Installing the LunarG SDK needs elevation, which the reference
machine does not grant, so this project takes the two halves separately:

* the **import library** is generated from `System32\vulkan-1.dll` with
  `dumpbin` and `lib` (265 exports), which needs no SDK at all;
* **glslang** comes inside ncnn's own prebuilt release, so no shader compiler is
  needed either.

CMake, Ninja and MSVC all ship with Visual Studio; they are simply not on `PATH`
by default. `vswhere` finds them.

## Pinning

Nothing here floats, and `pins.json` is the record.

* ncnn is pinned by **tag, commit and the SHA256 of the release archive** it was
  taken from — not `master`, ever. `pins.json` also records that ncnn `20250503`,
  the version the RIFE models were actually exported against, cannot submit to
  Vulkan on the reference machine at all, so nobody repeats that experiment.
* RIFE and Real-ESRGAN are **not** submodules. The repositories around them are
  wrappers we explicitly do not want (a VapourSynth plugin, a command-line tool),
  so only the files needed to run the two networks are vendored, under `src/`
  with a provenance header naming repository, blob SHA and license.
* Model weights are downloaded, never committed — `.gitignore` excludes `models/`.
  `pins.json` records where each came from and what it hashed to, so the
  directory is reproducible rather than merely present.

## How the models are stored

ncnn's published RIFE weights are **fp16**, and ncnn reads fp16 natively. The copy
this runtime loads is `ncnnoptimize`'s fp32 conversion, which is ncnn's own reader
rather than a hand-written one — a hand-written flat fp16 decoder was tried first
and was wrong by 9,450 values, which is worth knowing before anyone writes one
again. `pins.json` records the hash of the published file and of the converted
one, so either can be checked.

The published RIFE 4.25 artifact contains 288 fp16 words with exponent 31
(NaN/Inf), which is a defect in the artifact and not in the download. Zeroing
exactly those words changes the output by nothing measurable, so they are not
what `rife_pair_test` is failing on, and the file is kept byte-faithful.

## Why no general tensor API

Because "generic" is what went wrong last time. An ABI that can express any
graph also lets a caller believe a model ran when a filter did. Two functions
with fixed shapes cannot express that ambiguity: either `sr_rife_process`
produced a frame or it returned an error.

## Deliberate non-goals

* No external executable is ever spawned, and no frame is ever written to disk.
  Decode → `sr_image` → ncnn → `sr_image` → encode, in memory.
* No Python, no PyTorch, no CUDA, no TensorRT, no ONNX Runtime, no DirectML.
  Vulkan is the only compute API, and ncnn is the only runtime.
* No model menu. One restoration model and one interpolation model, frozen,
  because the product promise is unattended operation with no per-title tuning.
