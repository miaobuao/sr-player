//! Running a parsed model on the GPU.
//!
//! The CPU reference in [`crate::ifnet`] is what makes the arithmetic checkable
//! on a machine with no GPU, and it is not what a film will be restored with: at
//! 1080p a single convolution over a handful of channels is tens of millions of
//! multiply-adds per frame, and there are frames to spare.
//!
//! This module dispatches the same operators on the device, driven by the same
//! parsed graph, so the two can be compared on identical inputs — which is the
//! only reason a CPU reference is worth having.
//!
//! The shader is compiled by naga at runtime, so there is no shader compiler or
//! Vulkan SDK in the build. Vulkan is the only backend requested: the ABI
//! advertises a Vulkan backend, and quietly running somewhere else would make the
//! capability report a lie.

use crate::ifnet::{self, Model, ModelError, Planar};
use std::collections::HashMap;

/// The operator kernels.
///
/// Every buffer uses the same interleaved layout as the CPU reference —
/// `data[(y * width + x) * channels + channel]` — so a disagreement between the
/// two is a disagreement about arithmetic rather than about memory order. Getting
/// that wrong once already: a channel is not a contiguous range in this layout,
/// which is why concatenation is a shader and not a buffer copy.
pub const SHADER: &str = r#"
struct Params {
    width: u32,
    height: u32,
    in_ch: u32,
    out_ch: u32,
    divisor: f32,
    // The input's size, which is the output's for every operator except `interp`.
    // The shader's copy has the same eight 4-byte fields, so the two layouts are
    // 32 bytes and cannot drift.
    src_width: f32,
    src_height: f32,
    pad0: f32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
// For `add` this is the second operand rather than a weight table. One binding,
// two names: an auto-generated layout includes *every* binding the module
// declares, so a fifth binding that only `add` uses would have to be supplied by
// the convolution's bind group too, and wgpu rejects it if it is not.
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn input_at(x: i32, y: i32, channel: u32) -> f32 {
    if (x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.height)) {
        return 0.0;
    }
    return input[(u32(y) * params.width + u32(x)) * params.in_ch + channel];
}

// 3x3 convolution, padding 1, weights laid out out_ch x in_ch x 3 x 3 then bias.
@compute @workgroup_size(8, 8, 1)
fn conv3x3(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let per_out = params.in_ch * 9u;
    for (var oc = 0u; oc < params.out_ch; oc = oc + 1u) {
        var sum = aux[params.out_ch * per_out + oc];
        for (var ic = 0u; ic < params.in_ch; ic = ic + 1u) {
            let base = oc * per_out + ic * 9u;
            for (var ky = 0; ky < 3; ky = ky + 1) {
                for (var kx = 0; kx < 3; kx = kx + 1) {
                    let sx = i32(id.x) + kx - 1;
                    let sy = i32(id.y) + ky - 1;
                    let value = input_at(sx, sy, ic);
                    sum = sum + value * aux[base + u32(ky * 3 + kx)];
                }
            }
        }
        output[(id.y * params.width + id.x) * params.out_ch + oc] = sum;
    }
}

@compute @workgroup_size(8, 8, 1)
fn prelu(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let at = (id.y * params.width + id.x) * params.in_ch;
    for (var c = 0u; c < params.in_ch; c = c + 1u) {
        let value = input[at + c];
        if (value < 0.0) {
            output[at + c] = value * aux[c];
        } else {
            output[at + c] = value;
        }
    }
}

// Backward warp: output(p) = input(p + flow(p) / divisor), bilinear.
@compute @workgroup_size(8, 8, 1)
fn warp(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let at = (id.y * params.width + id.x) * 2u;
    let sx = f32(id.x) + aux[at] / params.divisor;
    let sy = f32(id.y) + aux[at + 1u] / params.divisor;
    let x0 = floor(sx);
    let y0 = floor(sy);
    let fx = sx - x0;
    let fy = sy - y0;
    let ix = i32(x0);
    let iy = i32(y0);
    let out_at = (id.y * params.width + id.x) * params.out_ch;
    for (var c = 0u; c < params.out_ch; c = c + 1u) {
        let top = input_at(ix, iy, c) * (1.0 - fx) + input_at(ix + 1, iy, c) * fx;
        let bottom = input_at(ix, iy + 1, c) * (1.0 - fx) + input_at(ix + 1, iy + 1, c) * fx;
        output[out_at + c] = top * (1.0 - fy) + bottom * fy;
    }
}

// Element-wise add, with a single-channel right operand broadcast over the planes
// (which is how a flow-derived mask is combined with a frame).
@compute @workgroup_size(64, 1, 1)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = params.width * params.height * params.out_ch;
    if (id.x >= count) {
        return;
    }
    if (params.in_ch == params.out_ch) {
        output[id.x] = input[id.x] + aux[id.x];
    } else {
        // The layout is interleaved, so the pixel a channel belongs to is
        // `index / channels`, not `index % plane`: with three channels the second
        // form wraps every pixel and reads the wrong plane for most of them.
        let pixel = id.x / params.out_ch;
        output[id.x] = input[id.x] + aux[pixel];
    }
}

// Concatenation along channels, two operands at a time.
//
// This cannot be a buffer copy: the layout is interleaved, so one channel of a
// frame is not a contiguous range and `copy_buffer_to_buffer` would move quarter
// slices of the wrong pixels. A pair at a time is enough — the graph runner folds
// longer concatenations into pairs — and it keeps the binding count fixed.
@compute @workgroup_size(64, 1, 1)
fn concat(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = params.width * params.height * params.out_ch;
    if (id.x >= count) {
        return;
    }
    let first_ch = params.in_ch;
    let second_ch = params.out_ch - first_ch;
    let pixel = id.x / params.out_ch;
    let channel = id.x % params.out_ch;
    if (channel < first_ch) {
        output[id.x] = input[pixel * first_ch + channel];
    } else {
        output[id.x] = aux[pixel * second_ch + (channel - first_ch)];
    }
}

// Bilinear resize, matching the reference's pixel-centre convention exactly: an
// exact 2x downsample has to sample the same points, or the pyramid's levels
// disagree with the reference by a fraction of a pixel that grows with depth.
fn src_at(x: i32, y: i32, channel: u32) -> f32 {
    return input[(u32(y) * u32(params.src_width) + u32(x)) * params.in_ch + channel];
}

@compute @workgroup_size(64, 1, 1)
fn interp(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = params.width * params.height * params.in_ch;
    if (id.x >= count) {
        return;
    }
    let pixel = id.x / params.in_ch;
    let channel = id.x % params.in_ch;
    let x = pixel % params.width;
    let y = pixel / params.width;
    let scale_x = params.src_width / f32(params.width);
    let scale_y = params.src_height / f32(params.height);
    let sx = max((f32(x) + 0.5) * scale_x - 0.5, 0.0);
    let sy = max((f32(y) + 0.5) * scale_y - 0.5, 0.0);
    let x0 = floor(sx);
    let y0 = floor(sy);
    let fx = sx - x0;
    let fy = sy - y0;
    let last_x = i32(params.src_width) - 1;
    let last_y = i32(params.src_height) - 1;
    let ix0 = min(i32(x0), last_x);
    let iy0 = min(i32(y0), last_y);
    let ix1 = min(i32(x0) + 1, last_x);
    let iy1 = min(i32(y0) + 1, last_y);
    let top = src_at(ix0, iy0, channel) * (1.0 - fx) + src_at(ix1, iy0, channel) * fx;
    let bottom = src_at(ix0, iy1, channel) * (1.0 - fx) + src_at(ix1, iy1, channel) * fx;
    output[id.x] = top * (1.0 - fy) + bottom * fy;
}
// A scalar multiply: `output = input * factor`. `divisor` carries the factor and
// `pad0` the operation code, which is what ncnn's `BinaryOp` reduces to when it
// has one bottom and a scalar. This is the "upsampled flow times two" step, and
// leaving it out of a coarse-to-fine pyramid halves every refined displacement.
@compute @workgroup_size(64, 1, 1)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = params.width * params.height * params.in_ch;
    if (id.x >= count) {
        return;
    }
    let left = input[id.x];
    let right = params.divisor;
    let op = u32(params.src_width);
    if (op == 2u) {
        output[id.x] = left * right;
    } else if (op == 0u) {
        output[id.x] = left + right;
    } else if (op == 1u) {
        output[id.x] = left - right;
    } else if (op == 3u) {
        if (right == 0.0) {
            output[id.x] = 0.0;
        } else {
            output[id.x] = left / right;
        }
    } else {
        output[id.x] = left;
    }
}
// Depth to space: `channels * scale^2` planes become `channels` planes at `scale`
// times the size. The inverse of a sub-pixel convolution, and how a restoration
// network upsamples without any filtering to get subtly wrong.
@compute @workgroup_size(64, 1, 1)
fn depth_to_space(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_ch = params.out_ch;
    let count = params.width * params.height * out_ch;
    if (id.x >= count) {
        return;
    }
    let scale = u32(params.src_width);
    let per_output = scale * scale;
    let pixel = id.x / out_ch;
    let channel = id.x % out_ch;
    let x = pixel % params.width;
    let y = pixel / params.width;
    let source_x = x / scale;
    let source_y = y / scale;
    let dx = x % scale;
    let dy = y % scale;
    let source_channel = channel * per_output + dy * scale + dx;
    let source_width = params.width / scale;
    let source_at = (source_y * source_width + source_x) * (out_ch * per_output) + source_channel;
    output[id.x] = input[source_at];
}
"#;

/// One dispatch's uniform block. The layout must match `Params` in the shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    in_ch: u32,
    out_ch: u32,
    divisor: f32,
    // The operand's size, which is the output's for every operator except
    // `interp`. Eight 4-byte fields in the shader's copy too, so the two layouts
    // are 32 bytes and cannot drift.
    src_width: f32,
    src_height: f32,
    pad0: f32,
}

impl Params {
    fn new(width: usize, height: usize, in_ch: usize, out_ch: usize, divisor: f32) -> Self {
        Params {
            width: width as u32,
            height: height as u32,
            in_ch: in_ch as u32,
            out_ch: out_ch as u32,
            divisor,
            src_width: width as f32,
            src_height: height as f32,
            pad0: 0.0,
        }
    }

    /// The operand's size, for the operators where it differs from the output's.
    fn with_source(mut self, width: usize, height: usize) -> Self {
        self.src_width = width as f32;
        self.src_height = height as f32;
        self
    }
}

/// A device with the operator pipelines built on it.
pub struct GpuOps {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    conv: wgpu::ComputePipeline,
    prelu: wgpu::ComputePipeline,
    warp: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    concat: wgpu::ComputePipeline,
    interp: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    shuffle: wgpu::ComputePipeline,
}

impl GpuOps {
    /// Opens the first Vulkan adapter and builds the pipelines.
    pub fn open() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
            .into_iter()
            .next()
            .ok_or_else(|| "no Vulkan adapter".to_string())?;
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("sr-infer-gpu-ifnet"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_buffer_size: limits.max_buffer_size,
                ..wgpu::Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .map_err(|err| format!("request_device failed: {err}"))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ifnet"),
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
        // Built here so a broken shader fails at open rather than mid-film.
        let conv = pipeline("conv3x3");
        let prelu = pipeline("prelu");
        let warp = pipeline("warp");
        let add = pipeline("add");
        let concat = pipeline("concat");
        let interp = pipeline("interp");
        let scale = pipeline("scale");
        let shuffle = pipeline("depth_to_space");
        Ok(GpuOps {
            device,
            queue,
            conv,
            prelu,
            warp,
            add,
            concat,
            interp,
            scale,
            shuffle,
        })
    }

    fn storage(&self, values: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("storage"),
                contents: bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// A zeroed storage buffer.
    ///
    /// The contents are initialised rather than left undefined, which matters more
    /// than the allocation it costs: a buffer the shader reads where no layer wrote
    /// is a nondeterministic result, and nondeterminism in a restoration pipeline
    /// is the worst kind of bug to find later. It was found here by a test that
    /// passed when run alone and failed in the full suite, which is exactly the
    /// signature of reading uninitialised device memory.
    fn empty(&self, floats: usize) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let zeros = vec![0.0f32; floats.max(1)];
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("output"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Runs the graph on the device and reads the result back.
    ///
    /// `a` and `b` are the graph's two `Input` layers in order. They are not
    /// required to have the same channel count: a graph may take a frame and a
    /// flow, and a check that assumed two frames would refuse a valid model.
    pub fn forward(&self, model: &Model, a: &Planar, b: &Planar) -> Result<Planar, ModelError> {
        if a.width != b.width || a.height != b.height {
            return Err(ModelError::Shape(format!(
                "the two inputs are {0}x{1} and {2}x{3}",
                a.width, a.height, b.width, b.height
            )));
        }
        let mut blobs: HashMap<String, wgpu::Buffer> = HashMap::new();
        let mut shapes: HashMap<String, (usize, usize, usize)> = HashMap::new();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut inputs_seen = 0usize;
        let mut cursor = 0usize;

        for layer in &model.graph.layers {
            let bottom = |index: usize| -> Result<(&wgpu::Buffer, (usize, usize, usize)), ModelError> {
                let name = layer.bottoms.get(index).ok_or_else(|| {
                    ModelError::Shape(format!("layer `{}` has no bottom {index}", layer.name))
                })?;
                let buffer = blobs.get(name).ok_or_else(|| {
                    ModelError::Shape(format!(
                        "layer `{}` reads blob `{name}`, which no earlier layer produced",
                        layer.name
                    ))
                })?;
                Ok((buffer, shapes[name]))
            };
            let top = layer.tops.first().cloned().unwrap_or_default();
            match layer.kind.as_str() {
                "Input" => {
                    let frame = if inputs_seen == 0 { a } else { b };
                    inputs_seen += 1;
                    blobs.insert(top.clone(), self.storage(&frame.data));
                    shapes.insert(
                        top,
                        (frame.width, frame.height, frame.channels),
                    );
                }
                "Split" => {
                    let (buffer, shape) = bottom(0)?;
                    // The same buffer under several names: a split copies nothing.
                    let buffer = buffer.clone();
                    for name in &layer.tops {
                        blobs.insert(name.clone(), buffer.clone());
                        shapes.insert(name.clone(), shape);
                    }
                }
                "Concat" => {
                    let mut frames = Vec::new();
                    for index in 0..layer.bottoms.len() {
                        let (buffer, shape) = bottom(index)?;
                        frames.push((buffer.clone(), shape));
                    }
                    let (width, height, _) = frames
                        .first()
                        .map(|(_, shape)| *shape)
                        .ok_or_else(|| ModelError::Shape("empty concat".into()))?;
                    for (_, shape) in &frames {
                        if shape.0 != width || shape.1 != height {
                            return Err(ModelError::Shape("concat of different sizes".into()));
                        }
                    }
                    // Folded pairwise, because the kernel takes two operands and a
                    // fixed binding count is worth more than a general one here.
                    let mut accumulated = frames[0].clone();
                    for (buffer, shape) in frames.iter().skip(1) {
                        let channels = accumulated.1 .2 + shape.2;
                        let output = self.empty(width * height * channels);
                        let params = self.uniform(Params::new(
                            width,
                            height,
                            accumulated.1 .2,
                            channels,
                            1.0,
                        ));
                        self.dispatch_pair(
                            &mut encoder,
                            &self.concat,
                            "concat",
                            &accumulated.0,
                            buffer,
                            &output,
                            &params,
                            width * height * channels,
                        );
                        accumulated = (output, (width, height, channels));
                    }
                    blobs.insert(top.clone(), accumulated.0);
                    shapes.insert(top, accumulated.1);
                }
                "Convolution" => {
                    let (input, (width, height, in_ch)) = bottom(0)?;
                    let input = input.clone();
                    let out_ch = layer
                        .num_output()
                        .ok_or_else(|| ModelError::Shape(format!("`{}` has no num_output", layer.name)))?;
                    if layer.kernel() != 3 {
                        return Err(ModelError::Shape(format!(
                            "`{}` is not 3x3",
                            layer.name
                        )));
                    }
                    let need = out_ch * in_ch * 9 + out_ch;
                    let weights = model.weights.get(cursor..cursor + need).ok_or_else(|| {
                        ModelError::Weights(format!("`{}` runs past the end", layer.name))
                    })?;
                    cursor += need;
                    let output = self.dispatch_conv(
                        &mut encoder,
                        &input,
                        weights,
                        Params::new(width, height, in_ch, out_ch, 1.0),
                    );
                    blobs.insert(top.clone(), output);
                    shapes.insert(top, (width, height, out_ch));
                }
                "PReLU" => {
                    let (input, (width, height, channels)) = bottom(0)?;
                    let input = input.clone();
                    let slopes = model.weights.get(cursor..cursor + channels).ok_or_else(|| {
                        ModelError::Weights(format!("`{}` runs past the end", layer.name))
                    })?;
                    cursor += channels;
                    let output = self.empty(width * height * channels);
                    let weights = self.storage(slopes);
                    let params = self.uniform(Params::new(width, height, channels, channels, 1.0));
                    let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("prelu"),
                        layout: &self.prelu.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: weights.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params.as_entire_binding(),
                            },
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("prelu"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.prelu);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(
                        (width as u32).div_ceil(8),
                        (height as u32).div_ceil(8),
                        1,
                    );
                    drop(pass);
                    blobs.insert(top.clone(), output);
                    shapes.insert(top, (width, height, channels));
                }
                "Warp" => {
                    let (input, (width, height, channels)) = bottom(0)?;
                    let (flow, flow_shape) = bottom(1)?;
                    if flow_shape.2 != 2 {
                        return Err(ModelError::Shape(format!(
                            "`{}` needs a 2-channel flow, got {}",
                            layer.name, flow_shape.2
                        )));
                    }
                    let (input, flow) = (input.clone(), flow.clone());
                    let divisor = layer.option(0).map(|v| v as f32).unwrap_or(2.0);
                    let output = self.empty(width * height * channels);
                    let params = self.uniform(Params::new(width, height, channels, channels, divisor));
                    let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("warp"),
                        layout: &self.warp.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: flow.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params.as_entire_binding(),
                            },
                        ],
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("warp"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.warp);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(
                        (width as u32).div_ceil(8),
                        (height as u32).div_ceil(8),
                        1,
                    );
                    drop(pass);
                    blobs.insert(top.clone(), output);
                    shapes.insert(top, (width, height, channels));
                }
                "Add" => {
                    let (input, (width, height, channels)) = bottom(0)?;
                    let (other, other_shape) = bottom(1)?;
                    let (input, other) = (input.clone(), other.clone());
                    let output = self.empty(width * height * channels);
                    let params = self.uniform(Params::new(
                        width,
                        height,
                        other_shape.2,
                        channels,
                        1.0,
                    ));
                    self.dispatch_pair(
                        &mut encoder,
                        &self.add,
                        "add",
                        &input,
                        &other,
                        &output,
                        &params,
                        width * height * channels,
                    );
                    blobs.insert(top.clone(), output);
                    shapes.insert(top, (width, height, channels));
                }
                "BinaryOp" => {
                    // One bottom and a scalar, which is the only form the pyramid
                    // uses: the flow scale. A two-bottom element-wise operation
                    // would need the pair dispatch and is refused by name.
                    if layer.bottoms.len() != 1 {
                        return Err(ModelError::Shape(format!(
                            "`{}`: only a one-bottom BinaryOp is implemented on the device",
                            layer.name
                        )));
                    }
                    let (input, (width, height, channels)) = bottom(0)?;
                    let input = input.clone();
                    let factor = layer.float_option(1).unwrap_or(1.0);
                    let op = layer.option(0).unwrap_or(0) as f32;
                    let output = self.empty(width * height * channels);
                    // `scale` reads the factor from `divisor` and the op code from
                    // `src_width`, both free for a kernel that never resizes.
                    let mut params = Params::new(width, height, channels, channels, factor);
                    params.src_width = op;
                    let params = self.uniform(params);
                    self.dispatch_single(
                        &mut encoder,
                        &self.scale,
                        "scale",
                        &input,
                        &output,
                        &params,
                        width * height * channels,
                    );
                    blobs.insert(top.clone(), output);
                    shapes.insert(top, (width, height, channels));
                }
                "DepthToSpace" => {
                    let (input, (src_width, src_height, in_ch)) = bottom(0)?;
                    let input = input.clone();
                    let scale = layer.option(0).unwrap_or(1).max(1) as usize;
                    if in_ch % (scale * scale) != 0 {
                        return Err(ModelError::Shape(format!(
                            "`{}`: {} channels is not a multiple of the {} a scale of {scale} \
                             needs",
                            layer.name,
                            in_ch,
                            scale * scale
                        )));
                    }
                    let channels = in_ch / (scale * scale);
                    let (width, height) = (src_width * scale, src_height * scale);
                    let output = self.empty(width * height * channels);
                    // `scale` travels in `src_width`, which the shuffle kernel reads
                    // and nothing else needs at this point in the graph.
                    let mut params = Params::new(width, height, channels, channels, 1.0);
                    params.src_width = scale as f32;
                    let params = self.uniform(params);
                    self.dispatch_single(
                        &mut encoder,
                        &self.shuffle,
                        "depth_to_space",
                        &input,
                        &output,
                        &params,
                        width * height * channels,
                    );
                    blobs.insert(top.clone(), output);
                    shapes.insert(top, (width, height, channels));
                }
                "Interp" => {
                    let (input, (src_width, src_height, channels)) = bottom(0)?;
                    let input = input.clone();
                    let width = layer.option(0).unwrap_or(src_width as i32).max(1) as usize;
                    let height = layer.option(1).unwrap_or(src_height as i32).max(1) as usize;
                    let output = self.empty(width * height * channels);
                    let params = self
                        .uniform(
                            Params::new(width, height, channels, channels, 1.0)
                                .with_source(src_width, src_height),
                        );
                    self.dispatch_single(
                        &mut encoder,
                        &self.interp,
                        "interp",
                        &input,
                        &output,
                        &params,
                        width * height * channels,
                    );
                    blobs.insert(top.clone(), output);
                    shapes.insert(top, (width, height, channels));
                }
                other => {
                    return Err(ModelError::Shape(format!(
                        "layer `{other}` ({}) is not implemented on the device",
                        layer.name
                    )))
                }
            }
        }

        let last = model
            .graph
            .layers
            .last()
            .ok_or_else(|| ModelError::Shape("the graph is empty".into()))?;
        let name = last
            .tops
            .last()
            .ok_or_else(|| ModelError::Shape("the last layer has no output".into()))?
            .clone();
        let (width, height, channels) = shapes[&name];
        let floats = width * height * channels;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (floats * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&blobs[&name], 0, &readback, 0, (floats * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|err| ModelError::Shape(format!("waiting for the device: {err}")))?;
        receiver
            .recv()
            .map_err(|err| ModelError::Shape(format!("readback: {err}")))?
            .map_err(|err| ModelError::Shape(format!("mapping the readback failed: {err}")))?;
        let data = slice
            .get_mapped_range()
            .map_err(|err| ModelError::Shape(format!("mapped range: {err}")))?;
        let values: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        Ok(Planar {
            width,
            height,
            channels,
            data: values,
        })
    }

    fn uniform(&self, params: Params) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// One dispatch of a kernel that takes a single storage input.
    ///
    /// `interp` needs its own because wgpu derives the bind group layout *per entry
    /// point*: a kernel that never mentions the second binding has a three-binding
    /// layout, and supplying four entries is a validation error rather than a
    /// harmless extra.
    fn dispatch_single(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        params: &wgpu::Buffer,
        count: usize,
    ) {
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        drop(pass);
    }

    /// One dispatch of a kernel that takes two storage inputs: the binding order
    /// is the same for `add` and `concat`, so they share this.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_pair(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
        first: &wgpu::Buffer,
        second: &wgpu::Buffer,
        output: &wgpu::Buffer,
        params: &wgpu::Buffer,
        count: usize,
    ) {
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: first.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: second.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        drop(pass);
    }

    fn dispatch_conv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        weights: &[f32],
        params: Params,
    ) -> wgpu::Buffer {
        let (width, height) = (params.width, params.height);
        let floats = width as usize * height as usize * params.out_ch as usize;
        let output = self.empty(floats);
        let weights = self.storage(weights);
        let params = self.uniform(params);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv"),
            layout: &self.conv.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.conv);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        drop(pass);
        output
    }
}

/// The graph a real RIFE checkpoint describes, checked for the layers this module
/// can execute.
pub fn unsupported_on_device(model: &Model) -> Vec<String> {
    model
        .graph
        .layers
        .iter()
        .filter(|layer| {
            !matches!(
                layer.kind.as_str(),
                "Input" | "Split" | "Concat" | "Add" | "PReLU" | "Convolution" | "Warp" | "Interp" | "BinaryOp" | "DepthToSpace"
            )
        })
        .map(|layer| format!("{} ({})", layer.kind, layer.name))
        .collect()
}

/// Keeps the CPU reference honest about being the fallback.
pub fn reference_matches(model: &Model, a: &Planar, b: &Planar) -> Result<f32, String> {
    let gpu = GpuOps::open()?;
    let device = gpu
        .forward(model, a, b)
        .map_err(|err| format!("gpu: {err}"))?;
    let cpu = model
        .forward(a, b)
        .map_err(|err| format!("cpu: {err}"))?;
    if device.data.len() != cpu.data.len() {
        return Err(format!(
            "the device produced {} values and the reference {}",
            device.data.len(),
            cpu.data.len()
        ));
    }
    Ok(device
        .data
        .iter()
        .zip(cpu.data.iter())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max))
}

/// Re-exported so callers can name the CPU type without importing both modules.
pub use ifnet::Planar as CpuPlanar;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ifnet::{parse_param, blend};

    /// Skips rather than fails: the rest of the suite must run on a machine with
    /// no Vulkan adapter, and a silent skip would be worse than a named one.
    fn ops_or_skip() -> Option<GpuOps> {
        match GpuOps::open() {
            Ok(ops) => Some(ops),
            Err(err) => {
                eprintln!("SKIPPED: {err}");
                None
            }
        }
    }

    fn model(param_body: &str, weights: Vec<f32>) -> Model {
        let text = format!(
            "7767517\n{}\n{param_body}",
            param_body.lines().count()
        );
        Model {
            graph: parse_param(&text).expect("parse"),
            weights,
        }
    }

    /// The resize must agree with the reference in both directions. A pyramid
    /// upsamples and downsamples repeatedly, so a half-pixel disagreement here
    /// compounds with depth rather than cancelling.
    #[test]
    fn the_device_resize_matches_the_reference_in_both_directions() {
        let Some(ops) = ops_or_skip() else {
            return;
        };
        let graph = model(
            "Input            a        0 1 a\n\
             Interp           down     1 1 a small 0=7 1=5\n\
             Interp           up       1 1 small big 0=21 1=15\n",
            Vec::new(),
        );
        let input = gradient(21, 15, 3, 0.3);
        let gpu = ops.forward(&graph, &input, &input).expect("forward");
        let cpu = graph.forward(&input, &input).expect("forward");
        assert_eq!((gpu.width, gpu.height, gpu.channels), (21, 15, 3));
        let worst = gpu
            .data
            .iter()
            .zip(cpu.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-4,
            "the device's resize must match the reference; worst difference {worst}"
        );
    }

    /// The whole three-level pyramid on the device, with the same invariant as the
    /// single-stage graph: no flow anywhere means the average of the two frames.
    ///
    /// This is the multi-scale wiring checked end to end on the device — six
    /// resizes, two full-resolution warps, the flow refinement and the fusion.
    #[test]
    fn a_zero_flow_pyramid_on_the_device_is_a_blend() {
        let Some(ops) = ops_or_skip() else {
            return;
        };
        let (width, height, channels) = (24, 16, 3);
        let graph = crate::ifnet::pyramid(width, height, channels, 4);
        assert!(
            unsupported_on_device(&graph).is_empty(),
            "the device must run the whole pyramid: {:?}",
            unsupported_on_device(&graph)
        );
        let a = gradient(width, height, channels, 0.2);
        let b = gradient(width, height, channels, 0.7);
        let gpu = ops.forward(&graph, &a, &b).expect("forward on the device");
        assert_eq!((gpu.width, gpu.height, gpu.channels), (width, height, channels));
        let expected = crate::ifnet::blend(&a, &b, 0.5);
        let worst = gpu
            .data
            .iter()
            .zip(expected.data.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "a pyramid with no flow must be a blend on the device too; worst difference {worst}"
        );
    }

    /// The restoration scaffold on the device, with the same invariant the CPU
    /// reference is held to: an identity upsampler must make the whole network an
    /// exact nearest-neighbour upscale.
    ///
    /// This is the device path for `SR_OP_RESTORE`'s operator set — convolution,
    /// PReLU, residual addition, and the sub-pixel rearrangement — end to end.
    #[test]
    fn a_restoration_scaffold_on_the_device_is_an_exact_nearest_upscale() {
        let Some(ops) = ops_or_skip() else {
            return;
        };
        let (width, height, channels, scale) = (6, 4, 3, 2);
        let graph = crate::ifnet::residual_sr(width, height, channels, 4, scale, 2);
        assert!(
            unsupported_on_device(&graph).is_empty(),
            "the device must run the whole restoration graph: {:?}",
            unsupported_on_device(&graph)
        );
        let input = gradient(width, height, channels, 0.3);
        let gpu = ops.forward(&graph, &input, &input).expect("forward");
        assert_eq!(
            (gpu.width, gpu.height, gpu.channels),
            (width * scale, height * scale, channels)
        );
        let expected = crate::ifnet::nearest_upscale(&input, scale);
        let worst = gpu
            .data
            .iter()
            .zip(expected.data.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "the device's restoration graph must be a nearest upscale with an \
             identity upsampler; worst difference {worst}"
        );
    }

    fn gradient(width: usize, height: usize, channels: usize, base: f32) -> Planar {
        let mut frame = Planar::new(width, height, channels);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    frame.set(
                        channel,
                        x,
                        y,
                        base + (x * 3 + y * 5 + channel * 7) as f32 * 0.001,
                    );
                }
            }
        }
        frame
    }

    /// The device must agree with the reference on the operator whose weight
    /// layout is the compatibility surface with a real checkpoint.
    #[test]
    fn the_device_convolution_matches_the_reference() {
        let Some(ops) = ops_or_skip() else {
            return;
        };
        // 3 input channels, 4 output channels, deterministic weights.
        let out_ch = 4;
        let in_ch = 3;
        let mut weights = vec![0.0f32; out_ch * in_ch * 9 + out_ch];
        for (index, value) in weights.iter_mut().enumerate() {
            *value = ((index * 37) % 19) as f32 * 0.01 - 0.09;
        }
        let graph = model(
            "Input            a        0 1 a\n\
             Convolution      conv     1 1 a out 0=4 1=3 7=3\n",
            weights,
        );
        let input = gradient(11, 7, in_ch, 0.2);
        let gpu = ops
            .forward(&graph, &input, &input)
            .expect("forward on the device");
        let cpu = graph.forward(&input, &input).expect("forward on the cpu");
        assert_eq!((gpu.width, gpu.height, gpu.channels), (11, 7, 4));
        let worst = gpu
            .data
            .iter()
            .zip(cpu.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-4,
            "the device and the reference must agree; worst difference {worst}"
        );
    }

    /// The architectural invariant, on the device this time.
    ///
    /// A zero flow field has to make the whole graph a plain average. That is the
    /// property a wrong warp, a wrong sign or a mis-ordered concat destroys, and
    /// it is checkable without a checkpoint.
    ///
    /// This is the test that found the layout bug: it failed with a difference of
    /// 0.25550002, which is the 0.25 between the two frames plus one pixel of
    /// gradient, so the answer was one warped frame rather than their average. The
    /// cause was concatenation being a buffer copy — a channel is not a contiguous
    /// range in an interleaved layout, so the copy moved the wrong bytes and the
    /// fusion's second operand was never the second frame.
    #[test]
    fn a_rife_shaped_graph_on_the_device_is_a_blend_when_the_flow_is_zero() {
        let Some(ops) = ops_or_skip() else {
            return;
        };
        let graph = model(
            "Input            input0   0 1 a\n\
             Input            input1   0 1 b\n\
             Convolution      enc0     1 1 a enc 0=4 1=3 7=3\n\
             PReLU            enc0r    1 1 enc encr 0=4\n\
             Convolution      flow     1 1 encr flow 0=2 1=3 7=4\n\
             Split            fork     1 2 flow flow_a flow_b\n\
             Warp             warpa    2 1 a flow_a warpa 0=2\n\
             Warp             warpb    2 1 b flow_b warpb 0=2\n\
             Concat           cat      2 1 warpa warpb joined\n\
             Convolution      fus      1 1 joined out 0=3 1=3 7=6\n",
            {
                let mut weights: Vec<f32> = Vec::new();
                weights.extend(std::iter::repeat(0.0).take(4 * 3 * 9 + 4));
                weights.extend(std::iter::repeat(0.0).take(4));
                weights.extend(std::iter::repeat(0.0).take(2 * 4 * 9 + 2));
                let mut fusion = vec![0.0f32; 3 * 6 * 9 + 3];
                for output in 0..3 {
                    for input in [output, output + 3] {
                        fusion[output * 6 * 9 + input * 9 + 4] = 0.5;
                    }
                }
                weights.extend(fusion);
                weights
            },
        );
        let a = gradient(9, 5, 3, 0.2);
        let b = gradient(9, 5, 3, 0.7);
        let gpu = ops.forward(&graph, &a, &b).expect("forward on the device");
        let expected = blend(&a, &b, 0.5);
        let worst = gpu
            .data
            .iter()
            .zip(expected.data.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "a zero flow field must make the device's graph a blend; worst difference {worst}"
        );
    }

    /// A constant flow field must move content, and in the right direction. On the
    /// device, with the sign carried through the graph rather than applied by the
    /// test.
    #[test]
    fn the_device_warp_moves_content_the_way_the_flow_says() {
        let Some(ops) = ops_or_skip() else {
            return;
        };
        // A single bright pixel at (2, 2), and a flow of +4 in x everywhere, so
        // the warp must sample it at x - 2 with the default divisor of 2.
        let mut a = Planar::new(9, 5, 1);
        a.set(0, 2, 2, 1.0);
        let graph = model(
            "Input            src      0 1 src\n\
             Input            fl       0 1 fl\n\
             Warp             warpa    2 1 src fl out 0=2\n",
            Vec::new(),
        );
        let mut flow = Planar::new(9, 5, 2);
        for y in 0..5 {
            for x in 0..9 {
                flow.set(0, x, y, -4.0);
            }
        }
        let gpu = ops.forward(&graph, &a, &flow).expect("forward");
        let column = |x: usize| -> f32 { (0..5).map(|y| gpu.at(0, x, y)).sum() };
        assert!(
            column(4) > 0.5,
            "a flow of -4 at divisor 2 must bring the pixel from x=2 to x=4, got {:?}",
            (0..9).map(column).collect::<Vec<_>>()
        );
        assert!(column(2) < 0.01, "and it must have left x=2");
    }

    /// A layer the device cannot run must be named, not approximated.
    ///
    /// The list is deliberately shorter than the CPU reference's: the device
    /// implements fewer layers, and a model that runs on one and not the other has
    /// to say so rather than produce a frame with a stage missing.
    #[test]
    fn a_layer_the_device_cannot_run_is_reported_by_name() {
        let graph = model(
            "Input            a        0 1 a\n\
             BatchNorm        bn       1 1 a out 0=3\n",
            Vec::new(),
        );
        assert_eq!(
            unsupported_on_device(&graph),
            vec!["BatchNorm (bn)".to_string()]
        );

        // A supported kind with an unsupported arity is a different case: it is not
        // in the static list, and the runner has to refuse it by name when it gets
        // there rather than treat the second operand as absent.
        let two_bottom = model(
            "Input            a        0 1 a\n\
             Input            b        0 1 b\n\
             BinaryOp         op       2 1 a b out 0=0\n",
            Vec::new(),
        );
        assert!(unsupported_on_device(&two_bottom).is_empty());
        // Skipping rather than substituting a dummy device: the previous version
        // guessed a `Planar` when there was no adapter and then failed the
        // assertion, which made this test pass alone and fail in the full suite.
        // No device means nothing to assert about what the device does.
        let Some(ops) = ops_or_skip() else {
            return;
        };
        let error = ops
            .forward(
                &two_bottom,
                &gradient(4, 4, 1, 0.1),
                &gradient(4, 4, 1, 0.2),
            )
            .expect_err("a two-bottom BinaryOp is not implemented on the device");
        assert!(
            error.to_string().contains("one-bottom BinaryOp"),
            "the error must say which form is supported: {error}"
        );

        // And the pyramid must not be on the static list, or the multi-scale device
        // test would be measuring nothing.
        let pyramid = crate::ifnet::pyramid(16, 16, 3, 4);
        assert!(
            unsupported_on_device(&pyramid).is_empty(),
            "{:?}",
            unsupported_on_device(&pyramid)
        );
    }
}
