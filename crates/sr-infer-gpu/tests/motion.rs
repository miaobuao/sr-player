//! Does the GPU backend actually produce motion-compensated interpolation?
//!
//! Three questions, three answers, none of them "it compiled":
//!
//! 1. **Does the shader compute what the Rust reference computes?** The same
//!    algorithm is implemented twice — once in WGSL, once in plain Rust — and the
//!    two must agree. A wrong index in a shader still produces something that
//!    looks like a frame, so "plausible" is not evidence.
//! 2. **Is the result closer to the truth than frame duplication?** A moving
//!    pattern is rendered at three positions; the middle one is the ground truth,
//!    and the interpolator only ever sees the two outer frames. PSNR against that
//!    hidden frame is what separates interpolation from guessing.
//! 3. **Do two vendors' drivers agree?** On a machine with both an NVIDIA and an
//!    AMD adapter the same kernels run on each; a driver-specific bug is exactly
//!    what a single-vendor test cannot see.

use sr_infer_gpu::engine::Engine;
use sr_infer_gpu::kernels::reference::{self, Frame};

/// An aperiodic value, so a block's SAD has one clear minimum.
///
/// A periodic pattern matches again every period, and a uniform one matches
/// everywhere: either would make a `t`-position measurement meaningless.
fn texture(x: i64, y: i64) -> u8 {
    let mut h = (x as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((y as u32).wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    (((h >> 24) & 0xff) / 2 + 60) as u8
}

/// The whole frame translated rigidly by `offset` pixels.
///
/// Every pixel moves, so a PSNR against the hidden middle frame measures the
/// motion compensation rather than the fraction of the frame that happens to be
/// static — which is what made a first version of this test report a meaningless
/// 0.9 dB of improvement.
fn shifted_frame(width: usize, height: usize, offset: i64) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height as i64 {
        for x in 0..width as i64 {
            let value = texture(x - offset, y);
            let index = ((y as usize) * width + x as usize) * 3;
            data[index] = value;
            data[index + 1] = value;
            data[index + 2] = value;
        }
    }
    data
}

/// A bright square moving over a textured background, for the kernel comparisons.
fn scene(width: usize, height: usize, offset: i32, texture: bool) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) * 3;
            // A gradient plus a fine checker gives the search something to lock
            // onto; a flat background would make every displacement equally good.
            let base = if texture {
                let checker = if ((x / 4) + (y / 4)) % 2 == 0 { 40 } else { 8 };
                (30 + (x * 90 / width) + checker) as u8
            } else {
                40
            };
            data[index] = base;
            data[index + 1] = base;
            data[index + 2] = base;
        }
    }
    for y in 0..16usize {
        for x in 0..16usize {
            let px = (x as i32 + 24 + offset).clamp(0, width as i32 - 1) as usize;
            let py = (20 + y).min(height - 1);
            let index = (py * width + px) * 3;
            data[index] = 235;
            data[index + 1] = 235;
            data[index + 2] = 235;
        }
    }
    data
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse <= 1e-12 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// PSNR over everything except a `margin`-wide border.
///
/// Content that translated out of the frame cannot be reconstructed by any
/// method, and those strips would otherwise dominate the number: the border is
/// the test's geometry, not the algorithm's quality.
fn psnr_region(a: &[u8], b: &[u8], width: usize, height: usize, margin: usize) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut squared = 0.0f64;
    let mut count = 0usize;
    for y in margin..height.saturating_sub(margin) {
        for x in margin..width.saturating_sub(margin) {
            for c in 0..3 {
                let index = (y * width + x) * 3 + c;
                let d = a[index] as f64 - b[index] as f64;
                squared += d * d;
                count += 1;
            }
        }
    }
    let mse = squared / count.max(1) as f64;
    if mse <= 1e-12 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

#[test]
fn the_shader_matches_the_rust_reference() {
    let adapters = Engine::adapters();
    if adapters.is_empty() {
        eprintln!("SKIPPED: no Vulkan adapter");
        return;
    }
    let (width, height) = (128usize, 96usize);
    let a = scene(width, height, 0, true);
    let b = scene(width, height, 6, true);
    let frame_a = Frame::from_rgb8(width, height, &a);
    let frame_b = Frame::from_rgb8(width, height, &b);
    let expected = reference::interpolate(&frame_a, &frame_b, 0.5);

    for engine in adapters {
        let name = engine.info.name.clone();
        let mut session = match Engine::open(engine) {
            Ok(session) => session,
            Err(err) => {
                eprintln!("SKIPPED {name}: {err}");
                continue;
            }
        };
        let actual = session
            .interpolate_pair(&a, &b, width as u32, height as u32, 0.5)
            .unwrap_or_else(|err| panic!("{name}: {err}"));

        let mut worst = 0i32;
        let mut differing = 0usize;
        for (x, y) in actual.iter().zip(&expected) {
            let delta = (*x as i32 - *y as i32).abs();
            if delta > 0 {
                differing += 1;
            }
            worst = worst.max(delta);
        }
        eprintln!(
            "{name}: {differing} of {} samples differ, worst delta {worst}",
            actual.len()
        );
        // Bit-exactness is not required across floating point implementations, but
        // a single ULP is not what a wrong index looks like: that is off by whole
        // pixels and shows up as a delta in the tens.
        assert!(
            worst <= 2,
            "{name}: the shader and the reference disagree by up to {worst} of 255, which is not \
             rounding"
        );
    }
}

#[test]
fn interpolation_beats_frame_duplication_against_ground_truth() {
    let adapters = Engine::adapters();
    if adapters.is_empty() {
        eprintln!("SKIPPED: no Vulkan adapter");
        return;
    }
    let (width, height) = (160usize, 120usize);
    // The frame translates 8 px between the two frames the interpolator sees, so
    // the hidden middle frame sits 4 px from each.
    let first = shifted_frame(width, height, 0);
    let last = shifted_frame(width, height, 8);
    let truth = shifted_frame(width, height, 4);

    let engine = adapters.into_iter().next().expect("checked above");
    let name = engine.info.name.clone();
    let mut session = Engine::open(engine).expect("open the first adapter");
    let interpolated = session
        .interpolate_pair(&first, &last, width as u32, height as u32, 0.5)
        .expect("interpolate");

    // What the engine does today when no model is installed: repeat a frame.
    let duplicated = first.clone();

    // A 16-pixel margin covers the search radius and the bilinear footprint: the
    // pixels beyond it genuinely moved out of frame and no interpolator can know
    // what was there.
    let margin = 16;
    let interpolated_psnr = psnr_region(&interpolated, &truth, width, height, margin);
    let duplicated_psnr = psnr_region(&duplicated, &truth, width, height, margin);
    eprintln!(
        "{name}: interior interpolated {interpolated_psnr:.2} dB vs duplicated {duplicated_psnr:.2} dB \
         (whole frame: {:.2} vs {:.2})",
        psnr(&interpolated, &truth),
        psnr(&duplicated, &truth)
    );
    assert!(
        interpolated_psnr > duplicated_psnr + 15.0,
        "a rigid translation must be reconstructed far better than by repeating a frame: \
         {interpolated_psnr:.2} dB vs {duplicated_psnr:.2} dB"
    );
    assert!(
        interpolated_psnr > 40.0,
        "a rigid translation should come back almost exactly, got {interpolated_psnr:.2} dB"
    );
}

#[test]
fn every_vendor_driver_agrees() {
    let adapters = Engine::adapters();
    if adapters.len() < 2 {
        eprintln!("SKIPPED: only {} Vulkan adapter(s)", adapters.len());
        return;
    }
    let (width, height) = (96usize, 64usize);
    let a = scene(width, height, 0, true);
    let b = scene(width, height, 5, true);

    let mut results = Vec::new();
    for engine in adapters {
        let name = engine.info.name.clone();
        let vendor = engine.info.vendor_id;
        let mut session = match Engine::open(engine) {
            Ok(session) => session,
            Err(err) => {
                eprintln!("SKIPPED {name}: {err}");
                continue;
            }
        };
        let frame = session
            .interpolate_pair(&a, &b, width as u32, height as u32, 0.5)
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        eprintln!("{name} (vendor 0x{vendor:04x}) produced a frame");
        results.push((name, vendor, frame));
    }
    assert!(
        results.len() >= 2,
        "needed two usable adapters to compare vendors"
    );

    let (first_name, _, first) = &results[0];
    for (name, _, frame) in &results[1..] {
        let mut worst = 0i32;
        for (x, y) in first.iter().zip(frame) {
            worst = worst.max((*x as i32 - *y as i32).abs());
        }
        eprintln!("{first_name} vs {name}: worst delta {worst}");
        // Integer arithmetic and sampling are identical; only the float rounding
        // inside the shader can differ, and only in the last bit or two.
        assert!(
            worst <= 2,
            "{first_name} and {name} disagree by up to {worst} of 255 on identical input"
        );
    }
}

#[test]
fn a_job_through_the_abi_fills_every_output_slot() {
    // The engine hands over `(n-1)*m+1` output images and reads them all back, so
    // a plugin that fills only the synthesised slots leaves the pass-throughs as
    // whatever the caller had there. The end-to-end symptom is subtle: the
    // output has the right frame count and the interpolated frames are correct,
    // while every original frame in it is blank.
    use sr_infer_gpu::abi::*;
    if Engine::adapters().is_empty() {
        eprintln!("SKIPPED: no Vulkan adapter");
        return;
    }
    let (width, height) = (64usize, 48usize);
    let a = scene(width, height, 0, true);
    let b = scene(width, height, 6, true);

    let mut session: *mut sr_infer_gpu::Session = std::ptr::null_mut();
    let desc = SrInferSessionDesc {
        struct_size: std::mem::size_of::<SrInferSessionDesc>() as u32,
        device_index: 0,
        flags: 0,
        reserved: 0,
        model_path: std::ptr::null(),
        model_name: std::ptr::null(),
        config_json: std::ptr::null(),
        vram_budget_bytes: 0,
        host_memory_budget_bytes: 0,
    };
    let code = unsafe { sr_infer_gpu::sr_infer_open(&desc, &mut session) };
    assert_eq!(code, SR_OK, "open failed");

    let image = |data: &mut Vec<u8>| SrInferImage {
        struct_size: std::mem::size_of::<SrInferImage>() as u32,
        memory: 1,
        dtype: SR_DTYPE_U8,
        layout: SR_LAYOUT_INTERLEAVED,
        color: SR_COLOR_RGB,
        range: 0,
        bit_depth: 8,
        planes: 3,
        width: width as u32,
        height: height as u32,
        stride: 0,
        plane_stride: [0; 4],
        data: data.as_mut_ptr() as *mut std::ffi::c_void,
        device_handle: 0,
        pts_num: 0,
        pts_den: 1,
    };

    let mut input_a = a.clone();
    let mut input_b = b.clone();
    let inputs = [image(&mut input_a), image(&mut input_b)];
    // 2 inputs at 2x produce 3 slots; the middle one is synthesised.
    let mut slot_0 = vec![0u8; width * height * 3];
    let mut slot_1 = vec![0u8; width * height * 3];
    let mut slot_2 = vec![0u8; width * height * 3];
    let mut outputs = [image(&mut slot_0), image(&mut slot_1), image(&mut slot_2)];

    let job = SrInferJob {
        struct_size: std::mem::size_of::<SrInferJob>() as u32,
        op: SR_OP_INTERPOLATE,
        flags: 0,
        input_count: 2,
        output_count: 3,
        inputs: inputs.as_ptr(),
        outputs: outputs.as_mut_ptr(),
        multiplier: 2,
        strength: 1.0,
        tile_width: 0,
        tile_height: 0,
        tile_pad: 0,
        reserved: 0,
        seed: 0,
        chunk_id: 0,
        fence: 0,
    };
    let mut result = SrInferResult {
        struct_size: std::mem::size_of::<SrInferResult>() as u32,
        outputs_written: 0,
        tiles: 0,
        reserved: 0,
        vram_used_bytes: 0,
        vram_budget_bytes: 0,
        fence: 0,
        message: std::ptr::null(),
    };
    let code = unsafe { sr_infer_gpu::sr_infer_execute(session, &job, &mut result) };
    unsafe { sr_infer_gpu::sr_infer_close(session) };
    assert_eq!(code, SR_OK, "execute failed");
    assert_eq!(result.outputs_written, 1, "one frame was synthesised");

    assert_eq!(
        slot_0, a,
        "output slot 0 must be the first input frame, not whatever was in the buffer"
    );
    assert_eq!(slot_2, b, "output slot 2 must be the second input frame");
    assert_ne!(slot_1, a, "the middle slot must be synthesised");
    assert_ne!(slot_1, b, "the middle slot must be synthesised");
}

#[test]
fn the_capability_report_does_not_claim_restoration() {
    // The engine decides what to run from these numbers, so a backend that
    // advertises `SR_OP_RESTORE` and cannot restore is how a pipeline ends up
    // reporting a restoration that never happened.
    use sr_infer_gpu::abi::*;
    if Engine::adapters().is_empty() {
        eprintln!("SKIPPED: no Vulkan adapter");
        return;
    }
    let mut caps = SrInferCaps {
        struct_size: std::mem::size_of::<SrInferCaps>() as u32,
        abi_version: 0,
        ops: 0,
        dtypes: 0,
        flags: 0,
        device_count: 0,
        max_batch: 1,
        min_temporal_window: 2,
        max_temporal_window: 2,
        tile_min: 0,
        tile_max: 0,
        upscale_num: 1,
        upscale_den: 1,
        max_pixels: 0,
        vram_bytes: 0,
        backend: std::ptr::null(),
        precision: std::ptr::null(),
        vendor: std::ptr::null(),
        model: std::ptr::null(),
        model_version: std::ptr::null(),
    };
    let code = unsafe { sr_infer_gpu::sr_infer_query(&mut caps) };
    assert_eq!(code, SR_OK);
    assert_eq!(caps.abi_version, SR_INFER_ABI_VERSION);
    assert!(caps.ops & SR_OP_INTERPOLATE != 0);
    assert_eq!(
        caps.ops & SR_OP_RESTORE,
        0,
        "restoration is not implemented and must not be advertised"
    );
    assert_eq!(caps.dtypes, SR_DTYPE_U8);
    assert!(caps.device_count >= 1);
    let backend = unsafe { std::ffi::CStr::from_ptr(caps.backend) };
    assert_eq!(backend.to_str().unwrap(), "vulkan");
}
