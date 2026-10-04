//! # Example inference plugin
//!
//! This crate exists to keep the inference ABI honest: it is a *real* shared
//! library implementing the `sr_infer` C ABI, with no dependency on `sr-core` at
//! all. The engine loads it in its tests and calls it through `dlopen`, which
//! proves the plugin boundary actually works instead of merely being documented.
//!
//! It implements the two operations a real backend must provide:
//!
//! * `SR_OP_BLEND_INTERPOLATE` — the classic two-frame blend, `out = (a + b) / 2`
//! * `SR_OP_SCALE` — nearest-neighbour resample by `scale_num / scale_den`
//!
//! A production plugin replaces these with a model runtime (Vulkan compute,
//! DirectML, OpenVINO, MIGraphX, ...). Nothing else about the ABI changes.
//!
//! Build it into the application's plugin directory:
//!
//! ```text
//! cargo build -p sr-infer-plugin-example --release
//! copy target/release/sr_infer_plugin_example.dll sr_infer.dll
//! ```

use std::os::raw::{c_char, c_int};

pub const SR_INFER_ABI_VERSION: u32 = 1;

pub const SR_OK: i32 = 0;
pub const SR_ERR_UNSUPPORTED: i32 = -1;
pub const SR_ERR_INVALID_ARGUMENT: i32 = -2;

pub const SR_OP_SCALE: u32 = 1;
pub const SR_OP_BLEND_INTERPOLATE: u32 = 2;

pub const SR_PIXEL_GRAY8: u32 = 3;
pub const SR_PIXEL_RGB8: u32 = 1;
pub const SR_PIXEL_RGBA8: u32 = 2;

#[repr(C)]
pub struct SrInferCapabilities {
    pub abi_version: u32,
    pub scale: u32,
    pub interpolate: u32,
    pub restore: u32,
    pub max_pixels: u64,
    pub backend: *const c_char,
    pub precision: *const c_char,
    pub vendor: *const c_char,
}

#[repr(C)]
pub struct SrInferFrame {
    pub data: *mut u8,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub pixel_format: u32,
}

#[repr(C)]
pub struct SrInferRequest {
    pub op: u32,
    pub input_count: u32,
    pub a: SrInferFrame,
    pub b: SrInferFrame,
    pub out: SrInferFrame,
    pub scale_num: u32,
    pub scale_den: u32,
}

#[repr(C)]
pub struct SrInferResponse {
    pub frames_written: u32,
    pub vram_used_bytes: u64,
    pub message: *const c_char,
}

static BACKEND: &[u8] = b"cpu\0";
static PRECISION: &[u8] = b"fp32\0";

fn bytes_per_pixel(pixel_format: u32) -> Option<usize> {
    match pixel_format {
        SR_PIXEL_GRAY8 => Some(1),
        SR_PIXEL_RGB8 => Some(3),
        SR_PIXEL_RGBA8 => Some(4),
        _ => None,
    }
}

fn row_stride(frame: &SrInferFrame, bpp: usize) -> usize {
    if frame.stride == 0 {
        frame.width as usize * bpp
    } else {
        frame.stride as usize
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sr_infer_abi_version() -> u32 {
    SR_INFER_ABI_VERSION
}

/// # Safety
/// `out` must be a valid, writable pointer to `SrInferCapabilities`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sr_infer_capabilities(out: *mut SrInferCapabilities) -> c_int {
    if out.is_null() {
        return SR_ERR_INVALID_ARGUMENT;
    }
    let caps = SrInferCapabilities {
        abi_version: SR_INFER_ABI_VERSION,
        scale: 1,
        interpolate: 1,
        // Deliberately 0: this example must never claim it can restore film.
        restore: 0,
        max_pixels: 0,
        backend: BACKEND.as_ptr() as *const c_char,
        precision: PRECISION.as_ptr() as *const c_char,
        vendor: std::ptr::null(),
    };
    unsafe { *out = caps };
    SR_OK
}

/// # Safety
/// All frame pointers inside `request` must reference at least
/// `height * stride` readable (and for `out`, writable) bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sr_infer_run(
    request: *const SrInferRequest,
    response: *mut SrInferResponse,
) -> c_int {
    if request.is_null() {
        return SR_ERR_INVALID_ARGUMENT;
    }
    let request = unsafe { &*request };
    if !response.is_null() {
        unsafe {
            (*response).frames_written = 0;
            (*response).vram_used_bytes = 0;
            (*response).message = std::ptr::null();
        }
    }
    let code = match request.op {
        SR_OP_BLEND_INTERPOLATE => unsafe { blend(request) },
        SR_OP_SCALE => unsafe { scale(request) },
        _ => SR_ERR_UNSUPPORTED,
    };
    if code == SR_OK && !response.is_null() {
        unsafe { (*response).frames_written = 1 };
    }
    code
}

unsafe fn blend(request: &SrInferRequest) -> c_int {
    let (a, b, out) = (&request.a, &request.b, &request.out);
    if a.data.is_null() || b.data.is_null() || out.data.is_null() {
        return SR_ERR_INVALID_ARGUMENT;
    }
    if a.width != b.width || a.height != b.height || a.width != out.width || a.height != out.height
    {
        return SR_ERR_INVALID_ARGUMENT;
    }
    if a.pixel_format != b.pixel_format || a.pixel_format != out.pixel_format {
        return SR_ERR_INVALID_ARGUMENT;
    }
    let bpp = match bytes_per_pixel(a.pixel_format) {
        Some(v) => v,
        None => return SR_ERR_UNSUPPORTED,
    };
    let rows = a.height as usize;
    let row_bytes = a.width as usize * bpp;
    let (sa, sb, so) = (
        row_stride(a, bpp),
        row_stride(b, bpp),
        row_stride(out, bpp),
    );
    for row in 0..rows {
        let src_a = unsafe { a.data.add(row * sa) };
        let src_b = unsafe { b.data.add(row * sb) };
        let dst = unsafe { out.data.add(row * so) };
        for index in 0..row_bytes {
            let av = unsafe { *src_a.add(index) } as u16;
            let bv = unsafe { *src_b.add(index) } as u16;
            unsafe { *dst.add(index) = ((av + bv) / 2) as u8 };
        }
    }
    SR_OK
}

unsafe fn scale(request: &SrInferRequest) -> c_int {
    let (input, out) = (&request.a, &request.out);
    if input.data.is_null() || out.data.is_null() {
        return SR_ERR_INVALID_ARGUMENT;
    }
    if request.scale_num == 0 || request.scale_den == 0 {
        return SR_ERR_INVALID_ARGUMENT;
    }
    if input.pixel_format != out.pixel_format {
        return SR_ERR_INVALID_ARGUMENT;
    }
    let bpp = match bytes_per_pixel(input.pixel_format) {
        Some(v) => v,
        None => return SR_ERR_UNSUPPORTED,
    };
    let expected_w = input.width as u64 * request.scale_num as u64 / request.scale_den as u64;
    let expected_h = input.height as u64 * request.scale_num as u64 / request.scale_den as u64;
    if out.width as u64 != expected_w || out.height as u64 != expected_h {
        return SR_ERR_INVALID_ARGUMENT;
    }
    let (si, so) = (row_stride(input, bpp), row_stride(out, bpp));
    for y in 0..out.height as usize {
        let src_y = y * input.height as usize / out.height as usize;
        let src_row = unsafe { input.data.add(src_y * si) };
        let dst_row = unsafe { out.data.add(y * so) };
        for x in 0..out.width as usize {
            let src_x = x * input.width as usize / out.width as usize;
            for channel in 0..bpp {
                let value = unsafe { *src_row.add(src_x * bpp + channel) };
                unsafe { *dst_row.add(x * bpp + channel) = value };
            }
        }
    }
    SR_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(data: &mut [u8], width: u32, height: u32) -> SrInferFrame {
        SrInferFrame {
            data: data.as_mut_ptr(),
            width,
            height,
            stride: 0,
            pixel_format: SR_PIXEL_RGB8,
        }
    }

    #[test]
    fn capabilities_declare_a_vendor_neutral_cpu_backend() {
        let mut caps = SrInferCapabilities {
            abi_version: 0,
            scale: 0,
            interpolate: 0,
            restore: 0,
            max_pixels: 0,
            backend: std::ptr::null(),
            precision: std::ptr::null(),
            vendor: std::ptr::null(),
        };
        assert_eq!(
            unsafe { sr_infer_capabilities(&mut caps) },
            SR_OK
        );
        assert_eq!(caps.abi_version, SR_INFER_ABI_VERSION);
        assert_eq!(caps.restore, 0);
        assert!(caps.vendor.is_null());
    }

    #[test]
    fn blend_interpolates_between_two_frames() {
        let mut a = vec![0u8, 0, 0, 200, 200, 200];
        let mut b = vec![100u8, 100, 100, 0, 0, 0];
        let mut out = vec![0u8; 6];
        let request = SrInferRequest {
            op: SR_OP_BLEND_INTERPOLATE,
            input_count: 2,
            a: frame(&mut a, 2, 1),
            b: frame(&mut b, 2, 1),
            out: frame(&mut out, 2, 1),
            scale_num: 1,
            scale_den: 1,
        };
        let mut response = SrInferResponse {
            frames_written: 0,
            vram_used_bytes: 0,
            message: std::ptr::null(),
        };
        assert_eq!(unsafe { sr_infer_run(&request, &mut response) }, SR_OK);
        assert_eq!(response.frames_written, 1);
        assert_eq!(out, vec![50, 50, 50, 100, 100, 100]);
    }

    #[test]
    fn unknown_operations_are_rejected() {
        let mut buf = vec![0u8; 3];
        let request = SrInferRequest {
            op: 4242,
            input_count: 1,
            a: frame(&mut buf, 1, 1),
            b: SrInferFrame {
                data: std::ptr::null_mut(),
                width: 0,
                height: 0,
                stride: 0,
                pixel_format: 0,
            },
            out: SrInferFrame {
                data: std::ptr::null_mut(),
                width: 0,
                height: 0,
                stride: 0,
                pixel_format: 0,
            },
            scale_num: 1,
            scale_den: 1,
        };
        assert_eq!(
            unsafe { sr_infer_run(&request, std::ptr::null_mut()) },
            SR_ERR_UNSUPPORTED
        );
    }
}
