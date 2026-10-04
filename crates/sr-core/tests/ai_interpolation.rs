//! RIFE through the Rust API.
//!
//! `rife_pair_test` proves the C++ runtime works and `ai`'s unit tests prove it can
//! be enumerated, but neither calls `Rife::interpolate` — the path that actually
//! crosses the FFI with three frame buffers and a model handle. This file is that
//! call, and nothing else: if it passes, the boundary carries pixels.
//!
//! The ground truth is computed here rather than compared against a reference
//! implementation, so a wrong answer cannot hide behind a wrong expectation. Two
//! frames of a static pattern, the second shifted by an exact number of pixels,
//! describe pure horizontal translation; the frame at timestep `t` is the pattern
//! shifted by `shift * t`.

use sr_core::ai::{self, FrameView};

const W: i32 = 640;
const H: i32 = 384;
/// Wide enough that the network's padding cannot reach the measured region.
const BORDER: i32 = 32;

const PI: f64 = std::f64::consts::PI;

fn pattern(x: i32, y: i32) -> u8 {
    // A period of 64 horizontally: long enough that the network's 1/8 working
    // scale still resolves it, and curved enough over an 8-pixel shift that
    // averaging the two frames is measurably *not* the midpoint.
    let horizontal = 60.0 * (2.0 * PI * x as f64 / 64.0).sin();
    let vertical = 40.0 * (2.0 * PI * y as f64 / 89.0).sin();
    (128.0 + horizontal + vertical).clamp(0.0, 255.0) as u8
}

fn frame(shift: i32) -> Vec<u8> {
    let mut pixels = vec![0u8; (W * H * 3) as usize];
    for y in 0..H {
        for x in 0..W {
            let sx = (x - shift).clamp(0, W - 1);
            let v = pattern(sx, y);
            let i = ((y * W + x) * 3) as usize;
            pixels[i] = v;
            pixels[i + 1] = v;
            pixels[i + 2] = v;
        }
    }
    pixels
}

/// Mean absolute error over the interior, excluding a border the padding affects.
fn mae(a: &[u8], b: &[u8]) -> f64 {
    let mut total = 0.0;
    let mut count = 0u64;
    for y in BORDER..H - BORDER {
        for x in BORDER..W - BORDER {
            for c in 0..3 {
                let i = ((y * W + x) * 3 + c) as usize;
                total += (a[i] as f64 - b[i] as f64).abs();
                count += 1;
            }
        }
    }
    total / count as f64
}

#[test]
fn rife_synthesises_a_frame_through_the_rust_api() {
    if !ai::models_installed() {
        eprintln!(
            "SKIPPED: no model weights. Run native/sr-native/setup-third-party.ps1, \
             or point SR_MODELS_DIR at an installed models directory."
        );
        return;
    }

    let runtime = ai::Runtime::open_preferred().expect("a device that can host a model");
    println!("device: {} ({})", runtime.device().name, runtime.device().device_type.as_str());
    let mut rife = runtime
        .open_rife(&ai::rife_model_dir())
        .expect("the RIFE model loads");

    // --- the trivial case, which no correct interpolator can fail -------------
    {
        let same = frame(0);
        let mut out = vec![0u8; same.len()];
        let mut view = FrameView::new(&mut out, W, H);
        rife.interpolate(&same, &same, 0.5, &mut view)
            .expect("a pair of identical frames");
        let error = mae(&out, &same);
        println!("identical frames -> {error:.3} levels");
        assert!(
            error < 1.0,
            "a pair of identical frames must interpolate to that frame, got {error:.3} levels"
        );
    }

    // --- the case that proves the timestep reaches the network ----------------
    {
        let a = frame(0);
        let b = frame(8);
        let truth = frame(4); // 8 pixels advanced by t = 0.5
        let mut out = vec![0u8; a.len()];
        let mut view = FrameView::new(&mut out, W, H);
        rife.interpolate(&a, &b, 0.5, &mut view)
            .expect("an 8-pixel pair");

        let error = mae(&out, &truth);
        let blend = {
            let mut blend = vec![0u8; a.len()];
            for i in 0..a.len() {
                blend[i] = ((a[i] as u16 + b[i] as u16) / 2) as u8;
            }
            mae(&blend, &truth)
        };
        println!("8 px shift, t = 0.50 -> {error:.3} levels (a blend scores {blend:.3})");
        assert!(error < 2.0, "the synthesised frame is {error:.3} levels off");
        assert!(
            error < blend * 0.6,
            "the result must beat averaging the inputs ({error:.3} against {blend:.3}), \
             or no network ran"
        );
    }

    // --- the ends of the timestep range are refused ---------------------------
    {
        let a = frame(0);
        let b = frame(8);
        let mut out = vec![0u8; a.len()];
        let mut view = FrameView::new(&mut out, W, H);
        assert!(
            rife.interpolate(&a, &b, 0.0, &mut view).is_err(),
            "timestep 0 is not a synthesis request"
        );
        assert!(
            rife.interpolate(&a, &b, 1.0, &mut view).is_err(),
            "timestep 1 is not a synthesis request"
        );
    }

    // --- a pair that is too small is rejected, not read past ------------------
    {
        let short = vec![0u8; 16];
        let mut out = vec![0u8; (W * H * 3) as usize];
        let mut view = FrameView::new(&mut out, W, H);
        assert!(
            rife.interpolate(&short, &short, 0.5, &mut view).is_err(),
            "a truncated frame must be refused rather than read past its end"
        );
    }
}
