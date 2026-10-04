//! The ABI structs, copied from `include/sr_infer.h`.
//!
//! Deliberately *not* imported from `sr-core`. A plugin that shares the engine's
//! type definitions would keep compiling after the two drifted apart, and the
//! drift would show up as misread memory in the middle of a feature. Keeping an
//! independent copy means a mismatch fails a test instead.

use std::ffi::c_char;

pub const SR_INFER_ABI_VERSION: u32 = 2;

pub const SR_OK: i32 = 0;
pub const SR_ERR_UNSUPPORTED: i32 = -1;
pub const SR_ERR_INVALID_ARGUMENT: i32 = -2;
pub const SR_ERR_RUNTIME: i32 = -3;
pub const SR_ERR_OUT_OF_MEMORY: i32 = -4;
pub const SR_ERR_CANCELLED: i32 = -5;
pub const SR_ERR_DEVICE_LOST: i32 = -6;
pub const SR_ERR_NO_DEVICE: i32 = -7;

pub const SR_OP_RESTORE: u32 = 1 << 0;
pub const SR_OP_INTERPOLATE: u32 = 1 << 1;
pub const SR_OP_SCALE: u32 = 1 << 2;

pub const SR_DTYPE_U8: u32 = 1 << 0;

pub const SR_LAYOUT_INTERLEAVED: u32 = 1;
pub const SR_COLOR_RGB: u32 = 1;

pub const SR_CAP_TILING: u32 = 1 << 0;
pub const SR_CAP_DEVICE_MEMORY: u32 = 1 << 1;
pub const SR_CAP_ASYNC: u32 = 1 << 2;
pub const SR_CAP_TEMPORAL: u32 = 1 << 3;

#[repr(C)]
pub struct SrInferDevice {
    pub struct_size: u32,
    pub index: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub reserved: u32,
    pub name: *const c_char,
    pub driver: *const c_char,
    pub vram_bytes: u64,
    pub vram_budget_bytes: u64,
    pub vram_used_bytes: u64,
}

#[repr(C)]
pub struct SrInferCaps {
    pub struct_size: u32,
    pub abi_version: u32,
    pub ops: u32,
    pub dtypes: u32,
    pub flags: u32,
    pub device_count: u32,
    pub max_batch: u32,
    pub min_temporal_window: u32,
    pub max_temporal_window: u32,
    pub tile_min: u32,
    pub tile_max: u32,
    pub upscale_num: u32,
    pub upscale_den: u32,
    pub max_pixels: u64,
    pub vram_bytes: u64,
    pub backend: *const c_char,
    pub precision: *const c_char,
    pub vendor: *const c_char,
    pub model: *const c_char,
    pub model_version: *const c_char,
}

#[repr(C)]
pub struct SrInferImage {
    pub struct_size: u32,
    pub memory: u32,
    pub dtype: u32,
    pub layout: u32,
    pub color: u32,
    pub range: u32,
    pub bit_depth: u32,
    pub planes: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub plane_stride: [u32; 4],
    pub data: *mut std::ffi::c_void,
    pub device_handle: u64,
    pub pts_num: i64,
    pub pts_den: i64,
}

#[repr(C)]
pub struct SrInferJob {
    pub struct_size: u32,
    pub op: u32,
    pub flags: u32,
    pub input_count: u32,
    pub output_count: u32,
    pub inputs: *const SrInferImage,
    pub outputs: *mut SrInferImage,
    pub multiplier: u32,
    pub strength: f32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tile_pad: u32,
    pub reserved: u32,
    pub seed: u64,
    pub chunk_id: u64,
    pub fence: u64,
}

#[repr(C)]
pub struct SrInferResult {
    pub struct_size: u32,
    pub outputs_written: u32,
    pub tiles: u32,
    pub reserved: u32,
    pub vram_used_bytes: u64,
    pub vram_budget_bytes: u64,
    pub fence: u64,
    pub message: *const c_char,
}

#[repr(C)]
pub struct SrInferSessionDesc {
    pub struct_size: u32,
    pub device_index: u32,
    pub flags: u32,
    pub reserved: u32,
    pub model_path: *const c_char,
    pub model_name: *const c_char,
    pub config_json: *const c_char,
    pub vram_budget_bytes: u64,
    pub host_memory_budget_bytes: u64,
}

/// Bytes an image occupies, honouring a caller-supplied stride.
pub fn image_bytes(image: &SrInferImage) -> usize {
    let row = if image.stride == 0 {
        image.width as usize * 3
    } else {
        image.stride as usize
    };
    row * image.height as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_the_engine() {
        // These numbers are the ABI wire format. `sr-core` asserts the same
        // numbers in `infer::abi`, from its own definitions, so a change to one
        // side fails both.
        let sizes = [
            ("SrInferImage", std::mem::size_of::<SrInferImage>()),
            ("SrInferJob", std::mem::size_of::<SrInferJob>()),
            ("SrInferResult", std::mem::size_of::<SrInferResult>()),
            ("SrInferSessionDesc", std::mem::size_of::<SrInferSessionDesc>()),
            ("SrInferDevice", std::mem::size_of::<SrInferDevice>()),
            ("SrInferCaps", std::mem::size_of::<SrInferCaps>()),
        ];
        for (name, size) in sizes {
            eprintln!("{name} = {size}");
        }
        assert_eq!(std::mem::size_of::<SrInferImage>(), 96);
        assert_eq!(std::mem::size_of::<SrInferJob>(), 88);
        assert_eq!(std::mem::size_of::<SrInferResult>(), 48);
        assert_eq!(std::mem::size_of::<SrInferSessionDesc>(), 56);
        assert_eq!(std::mem::size_of::<SrInferDevice>(), 64);
        assert_eq!(std::mem::size_of::<SrInferCaps>(), 112);
    }

    #[test]
    fn a_packed_image_is_width_times_height_times_three() {
        let image = SrInferImage {
            struct_size: std::mem::size_of::<SrInferImage>() as u32,
            memory: 1,
            dtype: SR_DTYPE_U8,
            layout: SR_LAYOUT_INTERLEAVED,
            color: SR_COLOR_RGB,
            range: 0,
            bit_depth: 8,
            planes: 3,
            width: 320,
            height: 240,
            stride: 0,
            plane_stride: [0; 4],
            data: std::ptr::null_mut(),
            device_handle: 0,
            pts_num: 0,
            pts_den: 1,
        };
        assert_eq!(image_bytes(&image), 320 * 240 * 3);
    }
}
