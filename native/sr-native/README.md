# sr-native

The one AI runtime. A C++ layer over a commit-pinned [ncnn](https://github.com/Tencent/ncnn),
exposing the C ABI in [`include/sr_native.h`](include/sr_native.h) to `sr-core`.

It runs exactly two networks:

| task | model | call |
|---|---|---|
| restoration | `RealESRGAN_x4plus` | `sr_restorer_process(in, out, scale, tile)` |
| interpolation | `RIFE 4.25`, `ensemble = false` | `sr_rife_process(prev, next, timestep, out)` |

## Status

**Not built yet.** The header is the frozen contract; the implementation, the
pinned ncnn checkout and the model manifests are the next phase. Until they
exist, `sr-core` refuses a request for either model task rather than substituting
Lanczos or `minterpolate` for it.

## Build prerequisites

ncnn's Vulkan backend needs the Vulkan **development** files, not just the
runtime: headers, a `vulkan-1` import library, and `glslangValidator` to compile
its compute shaders to SPIR-V. On Windows that means the LunarG Vulkan SDK
(`winget install KhronosGroup.VulkanSDK`). A machine that can *run* Vulkan
(`vulkan-1.dll` in `System32`) is not necessarily a machine that can *build* it.

CMake, Ninja and MSVC all ship with Visual Studio; they are simply not on `PATH`
by default. `vswhere` finds them.

## Pinning

Nothing here floats.

* `third_party/ncnn` is a git submodule fixed at a release tag. Not `master`,
  ever — a runtime that changes under a release is not a runtime you can test.
* RIFE and Real-ESRGAN are **not** submodules. We need a couple of files each,
  and the repositories around them are wrappers (a VapourSynth plugin, a
  command-line tool) that this project explicitly does not want. The files are
  vendored under `src/` with a provenance header naming repository, commit,
  file and license; `third_party/LICENSES/` carries the texts.
* Model weights are downloaded, never committed. `.gitignore` excludes `models/`
  and `/.sr-models/`. Each model directory carries a `manifest.json` with a
  `sha256`, and `sr_*_create` verifies it — a mismatch is a refusal, not a
  warning.

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
