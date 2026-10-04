# Third-party licences

Everything this product redistributes, and the terms it is redistributed under.
`native/sr-native/pins.json` is the machine-readable record — repository, commit,
blob SHA and archive hash for each. This directory is the human-readable half: the
licence texts themselves, so a redistributor has them in the tree rather than having
to fetch them from repositories that may move or disappear.

## What ships, and under what terms

| component | where it lives | licence | text |
|---|---|---|---|
| ncnn | linked into `sr-native` as a static library | BSD-3-Clause | [`ncnn.txt`](ncnn.txt) |
| RIFE (VapourSynth fork) | `src/rife/` — the `Warp` layer, three shaders | MIT | [`RIFE.txt`](RIFE.txt) |
| RIFE (nihui) | model weights for the 4.6 control only | MIT | [`rife-nihui.txt`](rife-nihui.txt) |
| Real-ESRGAN | model weights for `realesrgan-x4plus` | **BSD-3-Clause** | [`Real-ESRGAN.txt`](Real-ESRGAN.txt) |

Real-ESRGAN is BSD-3-Clause, not MIT. `pins.json` said MIT until this directory was
created and the actual text was read; the correction is in that file too. It matters
because BSD-3-Clause carries a non-endorsement clause that MIT does not, and a
redistributor relying on the wrong one would be relying on a term nobody granted.

## What is vendored, and what is not

Only `src/rife/` is third-party **source**. It is the `Warp` layer the RIFE models
require — an ncnn custom layer that exists nowhere else — plus the SPIR-V shaders it
compiles. Each file carries a provenance header naming its repository, blob SHA and
licence, and `warp.cpp` records the one local change made to it.

The model weights are **not** vendored and are not in this repository. They are
downloaded by `setup-third-party.ps1`, hash-checked against `pins.json`, and
`.gitignore` excludes `models/`. Their licences are recorded here because a build
that downloads them still redistributes them in everything it produces.

ncnn is **not** vendored as source either. It is taken as a pinned prebuilt archive,
verified by SHA256, and linked statically. That makes its licence text a shipping
requirement rather than a courtesy: a static link puts ncnn's code inside this
program's binary.

## The product's own licence

`LICENSE-MIT` and `LICENSE-APACHE` at the repository root, matching the
`MIT OR Apache-2.0` declaration in `Cargo.toml`.
