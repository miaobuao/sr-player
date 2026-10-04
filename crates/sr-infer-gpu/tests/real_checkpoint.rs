//! A real checkpoint, parsed by this crate's loader.
//!
//! The models are downloadable — `models/rife-HD/*.param` and `.bin` are in the
//! `rife-ncnn-vulkan` repository, and the `.param` files are small text — so the
//! question "can this backend load a real network" has an answer that is not a guess.
//! Point `SR_INFER_MODEL_DIR` at a directory of `.param` files and this parses every
//! one, prints the layer kinds it does not implement, and fails only if the parser
//! cannot read the file at all.
//!
//! The distinction matters: parsing is what this crate claims to do, and the gap in
//! operators is a list to work through rather than a wall. A test that demanded the
//! whole network would say nothing until the last operator landed; this one says
//! exactly what is missing, every run.
//!
//! Skips itself when the variable is unset, so the suite runs without the models.

use sr_infer_gpu::ifnet::parse_param;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[test]
fn a_real_checkpoint_parses_and_reports_what_is_missing() {
    let Ok(dir) = std::env::var("SR_INFER_MODEL_DIR") else {
        eprintln!("SKIPPED: set SR_INFER_MODEL_DIR to a directory of .param files");
        return;
    };
    let mut checked = 0usize;
    for entry in std::fs::read_dir(&dir).expect("read the model directory") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("param") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read the param file");
        let graph = match parse_param(&text) {
            Ok(graph) => graph,
            Err(err) => panic!("{} could not be parsed: {err}", path.display()),
        };
        let layer_count = graph.layers.len();
        let weight_count = graph.weight_count;
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        for layer in &graph.layers {
            *kinds.entry(layer.kind.clone()).or_insert(0) += 1;
        }
        let model = &graph;
        // The library's own list, not a copy of it here: a test that keeps its own
        // The library's own list, not a copy of it here: a test that keeps its own idea
        // of what is supported goes stale the moment the library grows, and then
        // reports a gap that no longer exists. It did exactly that on the run which
        // added three operators.
        let mut missing: BTreeMap<String, usize> = BTreeMap::new();
        for layer in &graph.layers {
            if !sr_infer_gpu::ifnet::SUPPORTED_KINDS.contains(&layer.kind.as_str()) {
                *missing.entry(layer.kind.clone()).or_insert(0) += 1;
            }
        }
        eprintln!(
            "{}: {} layers, {} weights",
            path.file_name().unwrap().to_string_lossy(),
            layer_count, weight_count
        );
        eprintln!("  kinds: {kinds:?}");
        eprintln!("  not implemented here: {missing:?}");
        checked += 1;
    }
    assert!(checked > 0, "no .param files found in {dir}");
}