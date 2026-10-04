//! Device management: adapters, pipelines, and the buffers a job runs through.
//!
//! Everything here is Vulkan. `wgpu` can also run on DX12 and Metal, and this
//! crate deliberately does not ask for them: a backend that calls itself Vulkan
//! and quietly runs on something else is the kind of claim this project exists to
//! stop making. If a machine has no Vulkan device, this backend reports no device
//! and the engine falls back — visibly.

use crate::kernels::{Params, BLOCK, SEARCH_RADIUS, SHADER};
use std::collections::HashMap;
use std::sync::Mutex;

/// Adapter enumeration is serialised.
///
/// Creating two Vulkan instances at once from different threads makes the loader
/// return an empty device list on this machine — which the engine would read as
/// "this plugin has no device" and quietly fall back to FFmpeg. A mutex is a
/// cheap price for not having that failure mode, and the engine only enumerates
/// at probe time anyway.
static ENUMERATION: Mutex<()> = Mutex::new(());

pub struct AdapterInfo {
    pub index: u32,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub driver: String,
    pub vram_bytes: u64,
    pub max_buffer_bytes: u64,
}

pub struct Engine {
    adapter: wgpu::Adapter,
    pub info: AdapterInfo,
}

fn device_type_code(kind: wgpu::DeviceType) -> u32 {
    // Matches SR_DEVICE_* in include/sr_infer.h.
    match kind {
        wgpu::DeviceType::DiscreteGpu => 3,
        wgpu::DeviceType::IntegratedGpu => 2,
        wgpu::DeviceType::VirtualGpu => 4,
        wgpu::DeviceType::Cpu => 1,
        wgpu::DeviceType::Other => 1,
    }
}

impl Engine {
    /// Every Vulkan adapter on the machine, in enumeration order.
    pub fn adapters() -> Vec<Engine> {
        let _serialised = ENUMERATION.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
            .into_iter()
            .enumerate()
            .map(|(index, adapter)| {
                let info = adapter.get_info();
                let limits = adapter.limits();
                Engine {
                    info: AdapterInfo {
                        index: index as u32,
                        name: info.name.clone(),
                        vendor_id: info.vendor,
                        device_id: info.device,
                        device_type: device_type_code(info.device_type),
                        driver: if info.driver_info.trim().is_empty() {
                            info.driver.clone()
                        } else {
                            info.driver_info.clone()
                        },
                        vram_bytes: 0,
                        max_buffer_bytes: limits.max_buffer_size,
                    },
                    adapter,
                }
            })
            .collect()
    }

    pub fn open(engine: Engine) -> Result<Session, String> {
        let limits = engine.adapter.limits();
        let (device, queue) = pollster::block_on(engine.adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("sr-infer-gpu"),
                required_features: wgpu::Features::empty(),
                // A 1080p frame pair needs ~75 MB of storage buffer; the downlevel
                // defaults cap a binding at 128 MB, which is enough, but asking
                // for the adapter's own limits avoids a surprise at 4K.
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                    max_buffer_size: limits.max_buffer_size,
                    ..wgpu::Limits::downlevel_defaults()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            },
        ))
        .map_err(|err| format!("request_device failed: {err}"))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        // Building the pipelines here rather than on the first frame is what
        // makes `sr_infer_open` the place where a broken shader is discovered.
        let search = pipeline("search");
        let warp = pipeline("warp");

        Ok(Session {
            device,
            queue,
            search,
            warp,
            workspaces: HashMap::new(),
        })
    }
}

/// One (width, height) worth of buffers, reused across frames.
struct Workspace {
    luma_a: wgpu::Buffer,
    luma_b: wgpu::Buffer,
    rgb_a: wgpu::Buffer,
    rgb_b: wgpu::Buffer,
    rgb_out: wgpu::Buffer,
    flow: wgpu::Buffer,
    params: wgpu::Buffer,
    readback: wgpu::Buffer,
    blocks_x: u32,
    blocks_y: u32,
    pixels: usize,
}

pub struct Session {
    device: wgpu::Device,
    queue: wgpu::Queue,
    search: wgpu::ComputePipeline,
    warp: wgpu::ComputePipeline,
    workspaces: HashMap<(u32, u32), Workspace>,
}

impl Session {
    /// Interpolates one frame between `a` and `b` at position `t`.
    ///
    /// `a` and `b` are interleaved RGB8, `width * height * 3` bytes each.
    pub fn interpolate_pair(
        &mut self,
        a: &[u8],
        b: &[u8],
        width: u32,
        height: u32,
        t: f32,
    ) -> Result<Vec<u8>, String> {
        let expected = width as usize * height as usize * 3;
        if a.len() != expected || b.len() != expected {
            return Err(format!(
                "frame size mismatch: {width}x{height} needs {expected} bytes, got {} and {}",
                a.len(),
                b.len()
            ));
        }
        if !self.workspaces.contains_key(&(width, height)) {
            let workspace = self.create_workspace(width, height)?;
            self.workspaces.insert((width, height), workspace);
        }
        let workspace = self
            .workspaces
            .get(&(width, height))
            .expect("just inserted");

        let pixels = workspace.pixels;
        let luma_a = to_luma(a, pixels);
        let luma_b = to_luma(b, pixels);
        let (rgb_a, rgb_b) = (to_f32(a), to_f32(b));
        let params = Params {
            width,
            height,
            blocks_x: workspace.blocks_x,
            blocks_y: workspace.blocks_y,
            t_num: t,
            t_den: 1.0,
            search_radius: SEARCH_RADIUS,
            block: BLOCK,
        };
        self.queue
            .write_buffer(&workspace.params, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&workspace.luma_a, 0, bytemuck::cast_slice(&luma_a));
        self.queue
            .write_buffer(&workspace.luma_b, 0, bytemuck::cast_slice(&luma_b));
        self.queue
            .write_buffer(&workspace.rgb_a, 0, bytemuck::cast_slice(&rgb_a));
        self.queue
            .write_buffer(&workspace.rgb_b, 0, bytemuck::cast_slice(&rgb_b));

        let layouts = (
            self.search.get_bind_group_layout(0),
            self.warp.get_bind_group_layout(0),
        );
        // Each pipeline declares only the bindings its entry point uses, so the
        // two groups are genuinely different: the search pass never sees the RGB
        // planes, and the warp pass only reads the flow field.
        let search_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("search"),
            layout: &layouts.0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: workspace.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: workspace.luma_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: workspace.luma_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: workspace.flow.as_entire_binding(),
                },
            ],
        });
        let warp_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("warp"),
            layout: &layouts.1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: workspace.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: workspace.flow.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: workspace.rgb_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: workspace.rgb_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: workspace.rgb_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("interpolate"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("search"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.search);
            pass.set_bind_group(0, &search_group, &[]);
            pass.dispatch_workgroups(
                workspace.blocks_x.div_ceil(8),
                workspace.blocks_y.div_ceil(8),
                1,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warp"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.warp);
            pass.set_bind_group(0, &warp_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        let byte_len = (pixels * 3 * 4) as u64;
        encoder.copy_buffer_to_buffer(&workspace.rgb_out, 0, &workspace.readback, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = workspace.readback.slice(..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|err| format!("device poll failed: {err}"))?;
        receiver
            .recv()
            .map_err(|err| format!("readback channel closed: {err}"))?
            .map_err(|err| format!("readback mapping failed: {err}"))?;

        let view = slice
            .get_mapped_range()
            .map_err(|err| format!("mapped range failed: {err}"))?;
        let floats: &[f32] =
            bytemuck::cast_slice(&view[..pixels * 3 * std::mem::size_of::<f32>()]);
        let mut out = vec![0u8; pixels * 3];
        for (index, value) in floats.iter().enumerate() {
            out[index] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        drop(view);
        workspace.readback.unmap();
        Ok(out)
    }

    fn create_workspace(&self, width: u32, height: u32) -> Result<Workspace, String> {
        let pixels = width as usize * height as usize;
        let floats = (pixels * 3 * std::mem::size_of::<f32>()) as u64;
        let luma_bytes = (pixels * std::mem::size_of::<f32>()) as u64;
        let blocks_x = width.div_ceil(BLOCK);
        let blocks_y = height.div_ceil(BLOCK);
        let flow_bytes = (blocks_x as u64 * blocks_y as u64) * 8;

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        // A 4K frame pair is 200 MB of float RGB; refusing early with a number is
        // better than an out-of-memory abort in the middle of a feature.
        let limit = self.device.limits().max_storage_buffer_binding_size as u64;
        if floats > limit || luma_bytes > limit || flow_bytes > limit {
            return Err(format!(
                "{width}x{height} needs a {:.1} MiB binding for one plane, above this device's \
                 {:.1} MiB limit",
                floats as f64 / 1_048_576.0,
                limit as f64 / 1_048_576.0
            ));
        }
        Ok(Workspace {
            luma_a: make("luma_a", luma_bytes, storage),
            luma_b: make("luma_b", luma_bytes, storage),
            rgb_a: make("rgb_a", floats, storage),
            rgb_b: make("rgb_b", floats, storage),
            rgb_out: make("rgb_out", floats, storage | wgpu::BufferUsages::COPY_SRC),
            flow: make("flow", flow_bytes, storage),
            params: make(
                "params",
                std::mem::size_of::<Params>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            readback: make(
                "readback",
                floats,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            blocks_x,
            blocks_y,
            pixels,
        })
    }
}

fn to_luma(rgb: &[u8], pixels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(pixels);
    for index in 0..pixels {
        let r = rgb[index * 3] as f32 / 255.0;
        let g = rgb[index * 3 + 1] as f32 / 255.0;
        let b = rgb[index * 3 + 2] as f32 / 255.0;
        out.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    out
}

fn to_f32(rgb: &[u8]) -> Vec<f32> {
    rgb.iter().map(|byte| *byte as f32 / 255.0).collect()
}
