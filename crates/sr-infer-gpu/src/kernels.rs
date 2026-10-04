//! The WGSL kernels, and the CPU reference they are checked against.
//!
//! Motion-compensated interpolation, in three passes:
//!
//! 1. `search` — one invocation per 8x8 block. For a block at `p` it finds the
//!    displacement `d` that minimises the sum of absolute differences between
//!    `a[p - d/2]` and `b[p + d/2]`. Searching *symmetrically* around the
//!    interpolated position is what makes the result an interpolation rather than
//!    a warp of one endpoint: the motion is shared by both frames.
//! 2. `smooth` — a 3x3 mean on the vector field. Block search produces a step
//!    function; without this the output shows block-shaped seams even when every
//!    vector is individually correct.
//! 3. `warp` — per pixel, sample `a` backwards and `b` forwards along the
//!    interpolated vector and blend. Where the two samples disagree strongly the
//!    nearer one wins, which is what stops a moving edge from turning into a
//!    ghost: averaged edges are exactly the artefact this pass exists to avoid.
//!
//! These are the same three steps the CPU reference in this file performs, and
//! `tests/motion.rs` requires the two to agree.

pub const BLOCK: u32 = 8;
pub const SEARCH_RADIUS: i32 = 8;

/// Parameters shared by every pass. `repr(C)` and 16-byte aligned, because that
/// is what a uniform buffer expects.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub width: u32,
    pub height: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    /// Numerator/denominator of the position between the two frames.
    pub t_num: f32,
    pub t_den: f32,
    pub search_radius: i32,
    pub block: u32,
}

pub const SHADER: &str = r#"
struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    t_num: f32,
    t_den: f32,
    search_radius: i32,
    block: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> luma_a: array<f32>;
@group(0) @binding(2) var<storage, read> luma_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> flow: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> rgb_a: array<f32>;
@group(0) @binding(5) var<storage, read> rgb_b: array<f32>;
@group(0) @binding(6) var<storage, read_write> rgb_out: array<f32>;

fn luma_a_at(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    return luma_a[u32(cy) * params.width + u32(cx)];
}

fn luma_b_at(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    return luma_b[u32(cy) * params.width + u32(cx)];
}

// WGSL has no way to pass a storage buffer into a function, so the samplers are
// per-buffer rather than generic over one.

fn texel_a(px: i32, py: i32, channel: u32) -> f32 {
    let cx = clamp(px, 0, i32(params.width) - 1);
    let cy = clamp(py, 0, i32(params.height) - 1);
    return rgb_a[u32(cy) * params.width * 3u + u32(cx) * 3u + channel];
}

fn texel_b(px: i32, py: i32, channel: u32) -> f32 {
    let cx = clamp(px, 0, i32(params.width) - 1);
    let cy = clamp(py, 0, i32(params.height) - 1);
    return rgb_b[u32(cy) * params.width * 3u + u32(cx) * 3u + channel];
}

fn bilinear_a(x: f32, y: f32, channel: u32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let top = mix(texel_a(x0, y0, channel), texel_a(x0 + 1, y0, channel), fx);
    let bottom = mix(texel_a(x0, y0 + 1, channel), texel_a(x0 + 1, y0 + 1, channel), fx);
    return mix(top, bottom, fy);
}

fn bilinear_b(x: f32, y: f32, channel: u32) -> f32 {
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let top = mix(texel_b(x0, y0, channel), texel_b(x0 + 1, y0, channel), fx);
    let bottom = mix(texel_b(x0, y0 + 1, channel), texel_b(x0 + 1, y0 + 1, channel), fx);
    return mix(top, bottom, fy);
}

// ---- pass 1: block matching ------------------------------------------------

@compute @workgroup_size(8, 8)
fn search(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.blocks_x || id.y >= params.blocks_y) { return; }
    let block = i32(params.block);
    let radius = params.search_radius;
    let origin_x = i32(id.x) * block;
    let origin_y = i32(id.y) * block;

    var best_cost = 1e30;
    var best = vec2<i32>(0, 0);

    // Symmetric search: a is sampled at -d/2 and b at +d/2, so `d` is the whole
    // inter-frame displacement and each frame carries half of it. The halving is
    // integer division, which is why the CPU reference uses `dx / 2` too: the
    // search is at whole-pixel granularity on each side.
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            var cost = 0.0;
            for (var by = 0; by < block; by = by + 1) {
                for (var bx = 0; bx < block; bx = bx + 1) {
                    let px = origin_x + bx;
                    let py = origin_y + by;
                    let av = luma_a_at(px - dx / 2, py - dy / 2);
                    let bv = luma_b_at(px + dx - dx / 2, py + dy - dy / 2);
                    cost = cost + abs(av - bv);
                }
            }
            // Prefer the smaller displacement when costs tie, so a flat region
            // does not invent motion out of numerical noise.
            let tie = f32(abs(dx) + abs(dy)) * 1e-6;
            if (cost + tie < best_cost) {
                best_cost = cost + tie;
                best = vec2<i32>(dx, dy);
            }
        }
    }
    flow[id.y * params.blocks_x + id.x] = vec2<f32>(f32(best.x), f32(best.y));
}

// ---- pass 2: warp and blend ------------------------------------------------

/// The 3x3 mean of the raw block field at the block containing (x, y).
///
/// Block matching produces a step function; interpolating a step function gives
/// block-shaped seams even when every vector is individually correct. Averaging
/// here rather than in a separate pass keeps the field in one buffer (a separate
/// smoothing pass would read and write the same storage buffer in one dispatch,
/// which is a race, not a smoothing).
fn flow_at(x: f32, y: f32) -> vec2<f32> {
    let block = f32(params.block);
    let bx = clamp(i32(floor(x / block)), 0, i32(params.blocks_x) - 1);
    let by = clamp(i32(floor(y / block)), 0, i32(params.blocks_y) - 1);
    var sum = vec2<f32>(0.0, 0.0);
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let nx = bx + dx;
            let ny = by + dy;
            if (nx < 0 || ny < 0 || nx >= i32(params.blocks_x) || ny >= i32(params.blocks_y)) {
                continue;
            }
            sum = sum + flow[u32(ny) * params.blocks_x + u32(nx)];
            count = count + 1.0;
        }
    }
    return sum / count;
}

@compute @workgroup_size(8, 8)
fn warp(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) { return; }
    let x = f32(id.x);
    let y = f32(id.y);
    let t = params.t_num / params.t_den;
    let d = flow_at(x, y);

    let index = (id.y * params.width + id.x) * 3u;
    for (var c = 0u; c < 3u; c = c + 1u) {
        let av = bilinear_a(x - d.x * t, y - d.y * t, c);
        let bv = bilinear_b(x + d.x * (1.0 - t), y + d.y * (1.0 - t), c);
        // A large disagreement means one of the two samples is occluded.
        // Averaging there is what produces a ghost, so the nearer sample wins.
        let blended = mix(av, bv, t);
        rgb_out[index + c] = select(blended, select(bv, av, t < 0.5), abs(av - bv) > 0.125);
    }
}
"#;

// ---- the CPU reference -----------------------------------------------------

/// The same algorithm, in Rust, used to check the kernels.
///
/// Two independent implementations of one algorithm is the only way to know that
/// a shader computes what its author thinks it computes: a GPU result that looks
/// plausible is not evidence, and a wrong WGSL expression (an integer division
/// where a float was meant, a transposed index) produces something that still
/// looks like a frame.
pub mod reference {
    use super::{BLOCK, SEARCH_RADIUS};

    /// `luma` is `width * height`, `rgb` is `width * height * 3`, row major.
    pub struct Frame {
        pub width: usize,
        pub height: usize,
        pub luma: Vec<f32>,
        pub rgb: Vec<f32>,
    }

    impl Frame {
        pub fn from_rgb8(width: usize, height: usize, data: &[u8]) -> Self {
            let mut luma = vec![0.0f32; width * height];
            let mut rgb = vec![0.0f32; width * height * 3];
            for i in 0..width * height {
                let r = data[i * 3] as f32 / 255.0;
                let g = data[i * 3 + 1] as f32 / 255.0;
                let b = data[i * 3 + 2] as f32 / 255.0;
                rgb[i * 3] = r;
                rgb[i * 3 + 1] = g;
                rgb[i * 3 + 2] = b;
                luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
            }
            Frame {
                width,
                height,
                luma,
                rgb,
            }
        }
    }

    fn luma_at(frame: &Frame, x: i32, y: i32) -> f32 {
        let cx = x.clamp(0, frame.width as i32 - 1) as usize;
        let cy = y.clamp(0, frame.height as i32 - 1) as usize;
        frame.luma[cy * frame.width + cx]
    }

    fn rgb_at(frame: &Frame, x: f32, y: f32, channel: usize) -> f32 {
        let w = frame.width as i32;
        let h = frame.height as i32;
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let at = |px: i32, py: i32| -> f32 {
            let cx = px.clamp(0, w - 1) as usize;
            let cy = py.clamp(0, h - 1) as usize;
            frame.rgb[(cy * frame.width + cx) * 3 + channel]
        };
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    pub fn search(a: &Frame, b: &Frame, radius: i32) -> (usize, usize, Vec<[f32; 2]>) {
        let blocks_x = a.width.div_ceil(BLOCK as usize);
        let blocks_y = a.height.div_ceil(BLOCK as usize);
        let mut flow = vec![[0.0f32; 2]; blocks_x * blocks_y];
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let origin_x = (bx * BLOCK as usize) as i32;
                let origin_y = (by * BLOCK as usize) as i32;
                let mut best_cost = f32::MAX;
                let mut best = [0i32; 2];
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let mut cost = 0.0f32;
                        for y in 0..BLOCK as i32 {
                            for x in 0..BLOCK as i32 {
                                let px = origin_x + x;
                                let py = origin_y + y;
                                let av = luma_at(a, px - dx / 2, py - dy / 2);
                                let bv = luma_at(b, px + dx - dx / 2, py + dy - dy / 2);
                                cost += (av - bv).abs();
                            }
                        }
                        let tie = (dx.abs() + dy.abs()) as f32 * 1e-6;
                        if cost + tie < best_cost {
                            best_cost = cost + tie;
                            best = [dx, dy];
                        }
                    }
                }
                flow[by * blocks_x + bx] = [best[0] as f32, best[1] as f32];
            }
        }
        (blocks_x, blocks_y, flow)
    }

    pub fn smooth(blocks_x: usize, blocks_y: usize, flow: &mut [[f32; 2]]) {
        let source = flow.to_vec();
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut sum = [0.0f32; 2];
                let mut count = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let x = bx as i32 + dx;
                        let y = by as i32 + dy;
                        if x < 0 || y < 0 || x >= blocks_x as i32 || y >= blocks_y as i32 {
                            continue;
                        }
                        let v = source[y as usize * blocks_x + x as usize];
                        sum[0] += v[0];
                        sum[1] += v[1];
                        count += 1.0;
                    }
                }
                flow[by * blocks_x + bx] = [sum[0] / count, sum[1] / count];
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn warp(
        a: &Frame,
        b: &Frame,
        blocks_x: usize,
        blocks_y: usize,
        flow: &[[f32; 2]],
        t: f32,
    ) -> Vec<u8> {
        let mut out = vec![0u8; a.width * a.height * 3];
        let block = BLOCK as f32;
        for y in 0..a.height {
            for x in 0..a.width {
                let bx = ((x as f32 / block).floor() as i32)
                    .clamp(0, blocks_x as i32 - 1) as usize;
                let by = ((y as f32 / block).floor() as i32)
                    .clamp(0, blocks_y as i32 - 1) as usize;
                let d = flow[by * blocks_x + bx];
                let (fx, fy) = (x as f32, y as f32);
                let ax = fx - d[0] * t;
                let ay = fy - d[1] * t;
                let bxx = fx + d[0] * (1.0 - t);
                let byy = fy + d[1] * (1.0 - t);
                for c in 0..3 {
                    let av = rgb_at(a, ax, ay, c);
                    let bv = rgb_at(b, bxx, byy, c);
                    let value = if (av - bv).abs() > 0.125 {
                        if t < 0.5 {
                            av
                        } else {
                            bv
                        }
                    } else {
                        av * (1.0 - t) + bv * t
                    };
                    out[(y * a.width + x) * 3 + c] =
                        (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        out
    }

    /// The whole algorithm, for tests and for the fallback path.
    pub fn interpolate(a: &Frame, b: &Frame, t: f32) -> Vec<u8> {
        let (blocks_x, blocks_y, mut flow) = search(a, b, SEARCH_RADIUS);
        smooth(blocks_x, blocks_y, &mut flow);
        warp(a, b, blocks_x, blocks_y, &flow, t)
    }
}

#[cfg(test)]
mod tests {
    use super::reference::*;
    use super::*;

    /// An aperiodic value, so a block's SAD has one clear minimum.
    ///
    /// A uniform patch cannot recover motion at all (every displacement matches
    /// equally well — the aperture problem, correctly reported as zero), and a
    /// periodic one matches again every period. Real footage has texture; a test
    /// that wants to measure flow recovery has to have it too.
    fn texture(x: usize, y: usize) -> u8 {
        let mut h = (x as u32)
            .wrapping_mul(374_761_393)
            .wrapping_add((y as u32).wrapping_mul(668_265_263));
        h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        (((h >> 24) & 0xff) / 2 + 60) as u8
    }

    /// A 16x16 textured square on a flat background, at `offset` pixels right.
    fn moving_square(offset: i32) -> Frame {
        let (width, height) = (64usize, 48usize);
        let mut data = vec![24u8; width * height * 3];
        for y in 0..16usize {
            for x in 0..16usize {
                let px = (24 + x as i32 + offset).clamp(0, width as i32 - 1) as usize;
                let py = 16 + y;
                let value = texture(x, y);
                let index = (py * width + px) * 3;
                data[index] = value;
                data[index + 1] = value;
                data[index + 2] = value;
            }
        }
        Frame::from_rgb8(width, height, &data)
    }

    #[test]
    fn the_reference_recovers_a_known_translation() {
        let a = moving_square(0);
        let b = moving_square(4);
        let (blocks_x, blocks_y, mut flow) = search(&a, &b, 8);
        // Block (4,3) covers x 32..40, y 24..32: inside the square at both offsets.
        let inside = 3 * blocks_x + 4;
        eprintln!("raw flow inside the square: {:?}", flow[inside]);
        assert!(
            (flow[inside][0] - 4.0).abs() < 1.5,
            "a block fully inside the square should report ~4 px, got {:?}",
            flow[inside]
        );
        smooth(blocks_x, blocks_y, &mut flow);
        assert!(
            (flow[inside][0] - 4.0).abs() < 1.5,
            "smoothed flow inside the square should still be ~4 px, got {:?}",
            flow[inside]
        );
        let _ = blocks_y;
    }

    #[test]
    fn a_flat_region_reports_no_motion() {
        let flat = Frame::from_rgb8(32, 32, &vec![128u8; 32 * 32 * 3]);
        let (blocks_x, blocks_y, mut flow) = search(&flat, &flat, 8);
        smooth(blocks_x, blocks_y, &mut flow);
        assert!(
            flow.iter().all(|v| v[0].abs() < 0.001 && v[1].abs() < 0.001),
            "a flat region must not invent motion: {flow:?}"
        );
    }
}
