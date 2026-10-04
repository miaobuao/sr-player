//! Raw declarations for `native/sr-native/include/sr_native.h`.
//!
//! Kept in one file with the header quoted above it, so a change on either side is
//! a readable diff rather than a hunt. There is no bindgen step: the ABI is four
//! functions plus two constructors, and a generator would be more machinery than
//! the thing it generates.
//!
//! Every function here is `unsafe`. Nothing outside `ai::` should call them.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int};

// enums, SR_OK .. SR_ERR_INTERNAL
pub const SR_OK: c_int = 0;
pub const SR_ERR_INVALID_ARGUMENT: c_int = 1;
pub const SR_ERR_NO_DEVICE: c_int = 2;
pub const SR_ERR_MODEL_MISSING: c_int = 3;
pub const SR_ERR_OUT_OF_MEMORY: c_int = 4;
pub const SR_ERR_DEVICE_LOST: c_int = 5;
pub const SR_ERR_INTERNAL: c_int = 6;

pub const SR_DEVICE_OTHER: i32 = 0;
pub const SR_DEVICE_INTEGRATED: i32 = 1;
pub const SR_DEVICE_DISCRETE: i32 = 2;
pub const SR_DEVICE_VIRTUAL: i32 = 3;
pub const SR_DEVICE_CPU: i32 = 4;

/// `sr_image` — a borrowed view of one interleaved RGB8 frame.
#[repr(C)]
#[derive(Debug)]
pub struct sr_image {
    pub data: *mut u8,
    pub width: i32,
    pub height: i32,
    /// Bytes between the start of one row and the next; may exceed `width * 3`.
    pub stride: i32,
    /// Must be 3.
    pub channels: i32,
}

/// `sr_device_info_t`.
///
/// The Rust and C layouts have to agree exactly, and there is no build step that
/// checks it — but the C side does: `sr_device_info` rejects the call unless
/// `struct_size` equals its own `sizeof`, so a mismatch is a loud error rather
/// than a misread field.
#[repr(C)]
pub struct sr_device_info_t {
    pub struct_size: u32,
    pub index: i32,
    pub name: [c_char; 256],
    pub driver_version: [c_char; 64],
    pub api_version: [c_char; 32],
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: i32,
    pub total_mib: u64,
    pub budget_mib: u64,
    pub unified_memory: i32,
}

impl Default for sr_device_info_t {
    fn default() -> Self {
        // Safety: all-zero is a valid initial state for this struct, and the only
        // field that must be non-zero before the call is `struct_size`.
        let mut value: sr_device_info_t = unsafe { std::mem::zeroed() };
        value.struct_size = std::mem::size_of::<sr_device_info_t>() as u32;
        value
    }
}

/// Opaque handles. The C side owns the memory; these are only ever pointers.
pub enum sr_context {}
pub enum sr_rife {}
pub enum sr_restorer {}

extern "C" {
    pub fn sr_device_count() -> c_int;
    pub fn sr_device_info(index: i32, out: *mut sr_device_info_t) -> c_int;

    pub fn sr_context_create(device_index: i32) -> *mut sr_context;
    pub fn sr_context_destroy(ctx: *mut sr_context);
    pub fn sr_last_error(ctx: *mut sr_context) -> *const c_char;

    pub fn sr_restorer_create(ctx: *mut sr_context, model_dir: *const c_char) -> *mut sr_restorer;
    pub fn sr_restorer_destroy(restorer: *mut sr_restorer);
    pub fn sr_restorer_output_size(
        restorer: *const sr_restorer,
        in_width: i32,
        in_height: i32,
        scale: i32,
        out_width: *mut i32,
        out_height: *mut i32,
    ) -> i32;
    pub fn sr_restorer_process(
        restorer: *mut sr_restorer,
        input: *const sr_image,
        output: *mut sr_image,
        scale: i32,
        tile: i32,
    ) -> c_int;

    pub fn sr_rife_create(ctx: *mut sr_context, model_dir: *const c_char) -> *mut sr_rife;
    pub fn sr_rife_destroy(rife: *mut sr_rife);
    pub fn sr_rife_process(
        rife: *mut sr_rife,
        previous: *const sr_image,
        next: *const sr_image,
        timestep: f32,
        output: *mut sr_image,
    ) -> c_int;
}
