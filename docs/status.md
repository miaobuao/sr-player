# Where this actually stands

One page, written at the end of the work, so that whoever continues does not have to
reconstruct it from thirty rounds of commit messages.

The objective is **not achieved**. Three of its four phases are done and one has not
been started. This file separates what has evidence behind it from what does not.

---

## Verified, with the evidence recorded

| claim | evidence |
|---|---|
| One AI runtime, no fallback, no flag | `native/sr-native` only; `crates/sr-core/build.rs` **fails the build** when it is absent rather than compiling the AI stages out |
| ncnn pinned | tag `20260526`, commit, and the SHA256 of the release archive, in `pins.json` |
| The runtime works on real weights | `device_test`, `restore_image_test`, `warp_unit_test`, `rife_pair_test` all pass |
| RIFE is the only interpolator | `Synthesise` segments call it; nothing else can produce frames |
| A pair across a cut cannot reach RIFE | the ABI takes a **pair**, not a window — structural, not a filter setting |
| Interpolation is correct | 479 output frames from 240 inputs, exactly `2*(n-1)+1`; 0 of 12 adjacent frames byte-identical |
| Restoration is in-process | no temporary file, no child process; a Lanczos upscale of the same frame compresses to 37 KB against the pipeline's 791 KB |
| Forced termination resumes | killed after 2 of 4 chunks, resumed reusing exactly those 2, output byte-identical to an uninterrupted run |
| No forbidden runtime dependency | imports are system DLLs plus the MSVC runtime; 49 modules loaded at runtime, none of them CUDA/TensorRT/PyTorch/ONNX/Python |
| Licences and provenance | `LICENSE-MIT`, `LICENSE-APACHE`, `third_party/LICENSES/`, provenance headers with real blob SHAs |
| Reproducible from a pin | `setup-third-party.ps1` rebuilds `third_party/` and `models/`, verifying every hash; run and confirmed |

`docs/verification.md` carries the numbers. `cargo test --workspace` is green.

---

## Not done

**Phase 4 is essentially untouched.** Per-shot regrain after interpolation, Silero
VAD through sherpa-ncnn replacing the hand-written speech detector, the 5.1 routing
rewrite (dialogue gain on the centre path, masking-band ducking on the relevant
background channels, LFE preserved), and conservative M/S stereo are all unbuilt.
The audio path is the one that has had least attention.

**The completion bar has never been approached.**

* no full-length film has been run. Every fixture is 8–12 seconds. Throughput is
  roughly **52× realtime**, so a two-hour film is on the order of four days of
  compute — throughput is itself an unsolved problem, not a footnote;
* **no AI visual QC exists.** It is named in the completion bar and three separate
  measurements now point at it as the only thing that could settle them;
* resume is verified at four chunks, not at film scale.

**Vendor independence is unverified.** `b38ed87` shows both models running
correctly on this machine's AMD integrated GPU with numerics agreeing to a few
hundredths of a level, and the model path contains no vendor-specific code
(`vendor_id` is reported and never branched on). But that device is *integrated*
with shared memory, **no Intel GPU was tested at all**, and the Rust side of the
boundary was not exercised on AMD. The clause asks for representative discrete AMD
and Intel GPUs. It is not satisfied.

---

## Assumed, and not verified

These are the places where the code is believed correct because nothing has
contradicted it, which is not the same thing.

1. **That restoration is worth doing.** Measured against known originals on two
   photographs, the pipeline scores **1.7–2.8 dB below a plain Lanczos upscale** on
   PSNR while carrying 2.8× the detail of the original — detail that was not there
   to recover. A generative restorer does this by design and PSNR punishes it, so
   this is not a defect. But `safe_16gb` applies restoration **unconditionally**, so
   a clean source that has merely been downscaled is asked to restore degradation
   that is not present. Whether that is right is a product decision that has not
   been made; the product currently makes it by default.
2. **That behaviour holds beyond short clips.** Nothing has been run long enough to
   expose accumulation, scratch-space growth, or drift.
3. **That the tile ladder has ever been exercised.** It is written and unit-tested,
   but no run has hit an allocation failure on the reference machine.
4. **That `sr_restorer_*` is model-agnostic in practice.** It is at the ABI level,
   and only Real-ESRGAN has ever been behind it.

---

## Where I would start, in order

1. **Build the AI visual QC.** It is in the completion bar, it is the only thing
   that can settle the restoration question above, and three measurements are
   already waiting on it. Everything else here is cheaper once it exists.
2. **Decide whether restoration is conditional**, and encode the decision. The
   evidence for the question is in `docs/verification.md`.
3. **Phase 4's audio work**, which is the largest untouched area and the one with
   the least verification of any kind.
4. **A long run**, to find what only a long run finds.

---

## Things worth knowing that cost time to learn

* **RIFE operates on 0..1, not 0..255.** `RIFE::process_v4` multiplies its input by
  255, and the network's own preprocessing immediately divides it back. Feeding it
   the multiplied value drives the flownet 255× past its range. This took twelve
  rounds because each half of the scaling was tested alone: 0..255 gave a garbage
  flow but a plausible-looking frame, and 0..1 with the output read back as 0..255
  gave a *black* frame, which "confirmed" the wrong answer. The fix came from reading
  1.4 KB of GLSL that had been vendored as compiled SPIR-V since round one.
* **Stage results are cached under the job id**, and the plan is a stage result. A
  changed option used to silently restore the old plan. The job identity now hashes
  the options.
* **Measure more than one sample.** Three times in this work a conclusion was drawn
  from a single frame or a single photograph and was wrong. The last one reversed an
  optimisation that had already been committed.
* **Check that the test could have failed.** Four tests passed while proving nothing
  — a whole-file hash comparison over two missing files, a reuse assertion that
  holds when everything is recomputed, a fixture that was one chunk when the test
  needed several, and a statistic that was zero because no case ran.
