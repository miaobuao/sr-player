// Links `sr-core` against the one AI runtime.
//!
//! There is no fallback here and no feature flag. `sr-native` is part of the
//! product, so a machine that cannot build it cannot build the product — which is
//! the honest arrangement, and the reason this script fails loudly with
//! instructions rather than quietly compiling the AI stages out. A build that
//! silently omitted them would produce a binary that still says "restoration" in
//! its own logs while resampling with Lanczos, which is the failure this whole
//! redesign exists to remove.
//!
//! Rust cannot compile the C++ itself, so this drives the batch file next door,
//! which calls vcvars64 and CMake. It re-runs whenever any source under
//! `native/sr-native` changes.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let native = manifest
        .parent()
        .and_then(Path::parent)
        .expect("crates/sr-core has a grandparent")
        .join("native")
        .join("sr-native");

    if !native.join("CMakeLists.txt").exists() {
        panic!(
            "native/sr-native is missing from {}.\n\
             The AI runtime is part of this product, not an optional extra.",
            native.display()
        );
    }

    let ncnn = native.join("third_party").join("ncnn");
    if !ncnn.join("include").join("ncnn").join("net.h").exists() {
        panic!(
            "third_party/ncnn is not staged.\n\n\
             Run, from the repository root:\n    \
             powershell -NoProfile -ExecutionPolicy Bypass -File native\\sr-native\\setup-third-party.ps1\n\n\
             That downloads the pinned ncnn, generates the Vulkan import library and\n\
             stages the model weights, verifying every hash against pins.json."
        );
    }

    build_native(&native);

    // Where the libraries landed. ncnn's own CMake package already knows its
    // transitive dependencies, but the static library is linked here by hand, so
    // every archive it needs has to be named.
    println!("cargo:rustc-link-search=native={}", native.join("build").display());
    println!("cargo:rustc-link-search=native={}", ncnn.join("lib").display());
    println!(
        "cargo:rustc-link-search=native={}",
        native.join("third_party").join("vulkan").display()
    );

    let ncnn_archives = [
        "ncnn",
        "glslang",
        "SPIRV",
        "MachineIndependent",
        "GenericCodeGen",
        "OSDependent",
        "glslang-default-resource-limits",
    ];
    let missing: Vec<&str> = ncnn_archives
        .iter()
        .copied()
        .filter(|name| !ncnn.join("lib").join(format!("{name}.lib")).exists())
        .collect();
    if !missing.is_empty() {
        panic!(
            "third_party/ncnn/lib is missing {missing:?}.\n\
             Re-run native/sr-native/setup-third-party.ps1."
        );
    }

    println!("cargo:rustc-link-lib=static=sr-native");
    for archive in ncnn_archives {
        println!("cargo:rustc-link-lib=static={archive}");
    }
    println!("cargo:rustc-link-lib=vulkan-1");
    // ncnn's static library reaches for these directly.
    println!("cargo:rustc-link-lib=ws2_32");
    println!("cargo:rustc-link-lib=dbghelp");

    println!("cargo:rerun-if-changed=build.rs");
    for dir in ["src", "include", "tests"] {
        watch_tree(&native.join(dir));
    }
    println!("cargo:rerun-if-changed={}", native.join("CMakeLists.txt").display());
}

/// Re-runs this script when any file under `dir` changes.
fn watch_tree(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            watch_tree(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Runs the batch file next door, and only when something actually needs rebuilding.
///
/// The check is on the library's timestamp against every source, so an ordinary
/// `cargo build` with nothing changed does not pay for a CMake configure.
fn build_native(native: &Path) {
    let library = native.join("build").join("sr-native.lib");
    if let Some(newest) = newest_source(native) {
        if let (Ok(lib_time), Ok(src_time)) = (fs::metadata(&library), fs::metadata(&newest)) {
            if let (Ok(lib_time), Ok(src_time)) = (lib_time.modified(), src_time.modified()) {
                if lib_time >= src_time {
                    return;
                }
            }
        }
    }

    let script = native.join("build-native.bat");
    println!("cargo:warning=building native/sr-native (this runs CMake and MSVC)");
    let output = Command::new("cmd")
        .arg("/c")
        .arg(&script)
        .output()
        .unwrap_or_else(|e| panic!("could not run {}: {e}", script.display()));

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "native/sr-native failed to build.\n\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
    }
}

fn newest_source(native: &Path) -> Option<PathBuf> {
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let mut consider = |path: PathBuf| {
        if let Ok(modified) = fs::metadata(&path).and_then(|m| m.modified()) {
            if newest.as_ref().map(|(t, _)| modified > *t).unwrap_or(true) {
                newest = Some((modified, path));
            }
        }
    };
    for dir in ["src", "include", "tests"] {
        collect(&native.join(dir), &mut consider);
    }
    consider(native.join("CMakeLists.txt"));
    newest.map(|(_, path)| path)
}

fn collect(dir: &Path, consider: &mut impl FnMut(PathBuf)) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, consider);
        } else {
            consider(path);
        }
    }
}
