//! The restoration path through the ABI.
//!
//! `SR_OP_RESTORE` was advertised by nothing, so the engine could not select the GPU
//! backend for a restoration job and the plan's resize stayed with Lanczos. This
//! drives the op through the same ABI the engine uses — query, open, execute — and
//! checks the invariant the scaffold guarantees: with an identity upsampler the graph
//! must return the frame it was given, exactly.
//!
//! That is a weak claim about picture quality and a strong one about execution: the
//! convolution, the residual blocks, the long skip, the sub-pixel rearrangement and
//! the device dispatch all have to happen, in order, on the right buffers, for the
//! bytes to come back unchanged.
//!
//! Skips itself when there is no Vulkan adapter.

use sr_infer_gpu::abi::*;

fn capabilities() -> Option<SrInferCaps> {
    let mut caps = SrInferCaps {
        struct_size: std::mem::size_of::<SrInferCaps>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    let code = unsafe { sr_infer_gpu::sr_infer_query(&mut caps) };
    if code != SR_OK {
        eprintln!("SKIPPED: the plugin could not report its capabilities ({code})");
        return None;
    }
    if caps.device_count == 0 {
        eprintln!("SKIPPED: no Vulkan adapter");
        return None;
    }
    Some(caps)
}

struct Session {
    raw: *mut sr_infer_gpu::Session,
}

impl Session {
    fn open() -> Option<Self> {
        capabilities()?;
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
        let mut raw: *mut sr_infer_gpu::Session = std::ptr::null_mut();
        let code = unsafe { sr_infer_gpu::sr_infer_open(&desc, &mut raw) };
        if code != SR_OK || raw.is_null() {
            eprintln!("SKIPPED: the session could not be opened ({code})");
            return None;
        }
        Some(Session { raw })
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            sr_infer_gpu::sr_infer_close(self.raw);
        }
    }
}

fn image(data: &mut Vec<u8>, width: u32, height: u32) -> SrInferImage {
    SrInferImage {
        struct_size: std::mem::size_of::<SrInferImage>() as u32,
        memory: 1,
        dtype: SR_DTYPE_U8,
        layout: SR_LAYOUT_INTERLEAVED,
        color: SR_COLOR_RGB,
        range: 0,
        bit_depth: 8,
        planes: 3,
        width,
        height,
        stride: 0,
        plane_stride: [0; 4],
        data: data.as_mut_ptr() as *mut std::ffi::c_void,
        device_handle: 0,
        pts_num: 0,
        pts_den: 1,
    }
}

/// The session's last error, read through the ABI the way a caller would.
fn last_error(session: *mut sr_infer_gpu::Session) -> String {
    let mut buffer = vec![0i8; 512];
    let code = unsafe {
        sr_infer_gpu::sr_infer_last_error(session, buffer.as_mut_ptr(), buffer.len() as u32)
    };
    if code != SR_OK {
        return format!("(no error text, code {code})");
    }
    unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// A frame with structure, so a rearrangement that moves the wrong pixel is visible.
fn frame(width: u32, height: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            data.push(((x * 7 + y * 13) % 256) as u8);
            data.push(((x * 3 + y * 29) % 256) as u8);
            data.push(((x * 17 + y * 5) % 256) as u8);
        }
    }
    data
}

#[test]
fn the_capability_report_offers_restoration() {
    let Some(caps) = capabilities() else {
        return;
    };
    assert_eq!(caps.abi_version, SR_INFER_ABI_VERSION);
    assert!(
        caps.ops & SR_OP_RESTORE != 0,
        "the backend now runs the restoration graph and must say so; ops was {}",
        caps.ops
    );
    assert!(caps.ops & SR_OP_INTERPOLATE != 0);
    // It still does not claim to upscale: the plan owns the resize, and a backend
    // that doubled the picture while the plan also resized would scale twice.
    assert_eq!((caps.upscale_num, caps.upscale_den), (1, 1));
}

#[test]
fn a_restoration_job_returns_every_frame_it_was_given() {
    let Some(session) = Session::open() else {
        return;
    };
    let (width, height) = (32u32, 24u32);
    let bytes = (width * height * 3) as usize;

    let source = frame(width, height);
    let mut inputs_storage: Vec<Vec<u8>> = (0..3).map(|_| source.clone()).collect();
    let mut inputs: Vec<SrInferImage> = inputs_storage
        .iter_mut()
        .map(|data| image(data, width, height))
        .collect();
    let mut outputs_storage: Vec<Vec<u8>> = (0..3).map(|_| vec![0u8; bytes]).collect();
    let mut outputs: Vec<SrInferImage> = outputs_storage
        .iter_mut()
        .map(|data| image(data, width, height))
        .collect();

    let job = SrInferJob {
        struct_size: std::mem::size_of::<SrInferJob>() as u32,
        op: SR_OP_RESTORE,
        flags: 0,
        input_count: inputs.len() as u32,
        output_count: outputs.len() as u32,
        inputs: inputs.as_ptr(),
        outputs: outputs.as_mut_ptr(),
        multiplier: 1,
        strength: 1.0,
        tile_width: 0,
        tile_height: 0,
        tile_pad: 0,
        reserved: 0,
        seed: 0,
        chunk_id: 7,
        fence: 0,
    };
    let mut result = SrInferResult {
        struct_size: std::mem::size_of::<SrInferResult>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    let code = unsafe { sr_infer_gpu::sr_infer_execute(session.raw, &job, &mut result) };
    assert_eq!(
        code,
        SR_OK,
        "executing a restoration job failed: {}",
        last_error(session.raw)
    );
    assert_eq!(result.outputs_written, 3);

    // Every output slot must hold the frame it was given. The invariant is exact
    // because every learned weight in the scaffold is zero: the picture passes
    // through two identity taps and a rearrangement that has to put it back.
    for (index, written) in outputs_storage.iter().enumerate() {
        let differing = written
            .iter()
            .zip(source.iter())
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(
            differing, 0,
            "output {index} differs from its input in {differing} of {bytes} bytes; a \
             restoration graph with identity weights must return the frame unchanged"
        );
    }
}

#[test]
fn a_restoration_job_that_is_not_one_to_one_is_refused() {
    let Some(session) = Session::open() else {
        return;
    };
    let (width, height) = (16u32, 16u32);
    let mut data = frame(width, height);
    let mut inputs = [image(&mut data, width, height)];
    let job = SrInferJob {
        struct_size: std::mem::size_of::<SrInferJob>() as u32,
        op: SR_OP_RESTORE,
        flags: 0,
        input_count: 1,
        output_count: 0,
        inputs: inputs.as_mut_ptr(),
        outputs: std::ptr::null_mut(),
        multiplier: 1,
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
        ..unsafe { std::mem::zeroed() }
    };
    assert_eq!(
        unsafe { sr_infer_gpu::sr_infer_execute(session.raw, &job, &mut result) },
        SR_ERR_INVALID_ARGUMENT,
        "a restoration job with no outputs must be refused, not run"
    );
}
