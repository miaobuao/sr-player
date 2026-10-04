//! Deterministic separable Lanczos resampling for 8-bit RGB frames.
//!
//! This exists because the restoration model is fixed at 4x and the deliverable is
//! not. Real-ESRGAN turns a 720x480 source into 2880x1920 and the target canvas is
//! 1620x1080, so something has to do the 0.5625 downscale. Until now the chunk
//! encoder's filter chain did it, which put the resize *after* interpolation and
//! meant RIFE ran on 3.2x the pixels the deliverable needs.
//!
//! Doing it here instead moves the resize between restoration and interpolation.
//! That is not only a speed change — it changes what RIFE is asked to align — so the
//! two orderings have to be compared on quality before either is adopted. This
//! module makes that comparison possible; it does not decide it.
//!
//! # Why not let FFmpeg do it
//!
//! Because the frames are in Rust between two FFmpeg processes at this point. The
//! alternative would be a third process, or an image-file round trip, and both are
//! explicitly out of bounds for this pipeline.
//!
//! # Determinism
//!
//! Weights are precomputed once per size pair and applied in a fixed order with f32
//! accumulation, so the same input gives the same bytes on every run and on every
//! machine. That is what "exact" means here: not a particular rounding, but the same
//! one every time. `resize_is_deterministic` holds it.

/// Lanczos kernel radius. Three lobes is the usual quality/cost point and is what
/// FFmpeg's `lanczos` filter uses by default.
const LOBES: f32 = 3.0;

/// The source samples and weights that produce one destination pixel.
///
/// Indices are stored rather than a start plus a stride, because edge handling
/// clamps them and clamped indices are not contiguous: two taps at the very edge can
/// address the same source sample, which is the correct behaviour (edge replication)
/// and which a start-and-offset scheme cannot express.
struct Taps {
    indices: Vec<usize>,
    weights: Vec<f32>,
}

impl Taps {
    /// `count` destination samples drawn from `src_len` source samples.
    ///
    /// Destination sample `i` sits at source position `(i + 0.5) * ratio - 0.5`,
    /// where `ratio = src_len / dst_len`. For a downscale the kernel is widened by
    /// the ratio, which is what stops it aliasing; for an upscale it is left alone.
    fn build(src_len: usize, dst_len: usize) -> Vec<Taps> {
        let ratio = src_len as f32 / dst_len as f32;
        let scale = ratio.max(1.0);
        let support = LOBES * scale;
        let last = src_len.saturating_sub(1);
        let mut table = Vec::with_capacity(dst_len);

        for i in 0..dst_len {
            let center = (i as f32 + 0.5) * ratio - 0.5;
            let left = (center - support).floor() as i64;
            let right = (center + support).ceil() as i64;

            let mut indices = Vec::new();
            let mut weights = Vec::new();
            let mut total = 0.0f32;
            for j in left..=right {
                let weight = lanczos((j as f32 - center) / scale, LOBES);
                if weight == 0.0 {
                    continue;
                }
                indices.push(j.clamp(0, last as i64) as usize);
                weights.push(weight);
                total += weight;
            }

            if total.abs() > f32::EPSILON && !indices.is_empty() {
                for weight in &mut weights {
                    *weight /= total;
                }
            } else {
                // Degenerate support -- a one-pixel source, or a destination sample
                // whose whole kernel fell outside. Take the nearest sample rather
                // than dividing by something near zero.
                let nearest = center.round().clamp(0.0, last as f32) as usize;
                indices.clear();
                weights.clear();
                indices.push(nearest);
                weights.push(1.0);
            }

            table.push(Taps { indices, weights });
        }
        table
    }
}

/// The Lanczos window: `sinc(x) * sinc(x / a)`.
fn lanczos(x: f32, a: f32) -> f32 {
    if x.abs() < 1e-6 {
        return 1.0;
    }
    if x.abs() >= a {
        return 0.0;
    }
    let px = std::f32::consts::PI * x;
    (px.sin() / px) * ((px / a).sin() / (px / a))
}

/// A resize from one fixed geometry to another, with its weights precomputed.
///
/// Build once and reuse across frames: the tables depend only on the sizes, and
/// rebuilding them per frame would dominate the cost of a 1080p downscale.
pub struct Resampler {
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    horizontal: Vec<Taps>,
    vertical: Vec<Taps>,
    /// Scratch for the horizontal pass: `dst_w x src_h`, three channels interleaved.
    intermediate: Vec<f32>,
}

impl Resampler {
    pub fn new(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Resampler {
        Resampler {
            src_w,
            src_h,
            dst_w,
            dst_h,
            horizontal: Taps::build(src_w, dst_w),
            vertical: Taps::build(src_h, dst_h),
            intermediate: vec![0.0; dst_w * src_h * 3],
        }
    }

    pub fn is_identity(&self) -> bool {
        self.src_w == self.dst_w && self.src_h == self.dst_h
    }

    pub fn output_len(&self) -> usize {
        self.dst_w * self.dst_h * 3
    }

    /// Resizes `src` into `dst`. Both are interleaved RGB8.
    ///
    /// # Panics
    ///
    /// If either buffer is smaller than its declared geometry, which is a caller
    /// bug rather than a runtime condition — the sizes are known when the resampler
    /// is built.
    pub fn apply(&mut self, src: &[u8], dst: &mut [u8]) {
        assert!(
            src.len() >= self.src_w * self.src_h * 3,
            "the source buffer is {} bytes but {}x{} needs {}",
            src.len(),
            self.src_w,
            self.src_h,
            self.src_w * self.src_h * 3
        );
        assert!(
            dst.len() >= self.output_len(),
            "the destination buffer is {} bytes but {}x{} needs {}",
            dst.len(),
            self.dst_w,
            self.dst_h,
            self.output_len()
        );

        self.pass_horizontal(src);
        self.pass_vertical(dst);
    }

    /// `src_w -> dst_w`, leaving the result in `intermediate`.
    ///
    /// Accumulated as f32 from the 8-bit source rather than through an intermediate
    /// 8-bit buffer: rounding to 8 bits between the two passes costs real quality on
    /// a large downscale, and the scratch is already allocated.
    fn pass_horizontal(&mut self, src: &[u8]) {
        for y in 0..self.src_h {
            let row = &src[y * self.src_w * 3..(y + 1) * self.src_w * 3];
            let out_row = &mut self.intermediate[y * self.dst_w * 3..(y + 1) * self.dst_w * 3];
            for (x, taps) in self.horizontal.iter().enumerate() {
                let mut acc = [0.0f32; 3];
                for (sx, weight) in taps.indices.iter().zip(taps.weights.iter()) {
                    let pixel = &row[sx * 3..sx * 3 + 3];
                    acc[0] += pixel[0] as f32 * weight;
                    acc[1] += pixel[1] as f32 * weight;
                    acc[2] += pixel[2] as f32 * weight;
                }
                out_row[x * 3] = acc[0];
                out_row[x * 3 + 1] = acc[1];
                out_row[x * 3 + 2] = acc[2];
            }
        }
    }

    /// `src_h -> dst_h`, reading `intermediate` and writing the 8-bit result.
    fn pass_vertical(&mut self, dst: &mut [u8]) {
        for (y, taps) in self.vertical.iter().enumerate() {
            let out_row = &mut dst[y * self.dst_w * 3..(y + 1) * self.dst_w * 3];
            for x in 0..self.dst_w {
                let mut acc = [0.0f32; 3];
                for (sy, weight) in taps.indices.iter().zip(taps.weights.iter()) {
                    let pixel = &self.intermediate[(sy * self.dst_w + x) * 3..][..3];
                    acc[0] += pixel[0] * weight;
                    acc[1] += pixel[1] * weight;
                    acc[2] += pixel[2] * weight;
                }
                out_row[x * 3] = clamp_u8(acc[0]);
                out_row[x * 3 + 1] = clamp_u8(acc[1]);
                out_row[x * 3 + 2] = clamp_u8(acc[2]);
            }
        }
    }
}

/// Rounds and clamps, with NaN going to 0 rather than wrapping to 255.
///
/// The same rule as the native layer's `sr_clamp_u8`, for the same reason: an
/// out-of-range weight must not be able to turn a frame white.
fn clamp_u8(value: f32) -> u8 {
    if value.is_nan() {
        return 0;
    }
    value.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: usize, h: usize) -> Vec<u8> {
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 255) / w.max(1)) as u8;
                let i = (y * w + x) * 3;
                data[i] = v;
                data[i + 1] = ((y * 255) / h.max(1)) as u8;
                data[i + 2] = 128;
            }
        }
        data
    }

    #[test]
    fn an_identity_resize_returns_the_source_bytes() {
        let src = ramp(64, 48);
        let mut resampler = Resampler::new(64, 48, 64, 48);
        assert!(resampler.is_identity());
        let mut dst = vec![0u8; 64 * 48 * 3];
        resampler.apply(&src, &mut dst);
        assert_eq!(dst, src, "a same-size resize must not alter a pixel");
    }

    #[test]
    fn a_constant_field_stays_constant() {
        // The strongest available check that the weights are normalised: if they did
        // not sum to one, a flat image would come out with a gradient or a shifted
        // level, and neither would be obvious in a photograph.
        for (sw, sh, dw, dh) in [(2880, 1920, 1620, 1080), (320, 240, 100, 75), (7, 5, 13, 11)] {
            let src = vec![200u8; sw * sh * 3];
            let mut resampler = Resampler::new(sw, sh, dw, dh);
            let mut dst = vec![0u8; dw * dh * 3];
            resampler.apply(&src, &mut dst);
            let min = *dst.iter().min().unwrap();
            let max = *dst.iter().max().unwrap();
            assert!(
                max - min <= 1 && (min as i32 - 200).abs() <= 1,
                "{sw}x{sh} -> {dw}x{dh}: a constant 200 came out as {min}..{max}"
            );
        }
    }

    #[test]
    fn a_downscale_preserves_the_mean() {
        // Resampling redistributes energy; it must not create or destroy it. A mean
        // that drifts means the edge handling is dropping or double-counting samples.
        let src = ramp(640, 480);
        let mut resampler = Resampler::new(640, 480, 360, 270);
        let mut dst = vec![0u8; 360 * 270 * 3];
        resampler.apply(&src, &mut dst);
        let mean = |data: &[u8]| data.iter().map(|v| *v as f64).sum::<f64>() / data.len() as f64;
        let (before, after) = (mean(&src), mean(&dst));
        assert!(
            (before - after).abs() < 2.0,
            "mean drifted from {before:.2} to {after:.2}"
        );
    }

    #[test]
    fn resize_is_deterministic() {
        let src = ramp(300, 200);
        let mut first = vec![0u8; 160 * 108 * 3];
        let mut second = vec![0u8; 160 * 108 * 3];
        Resampler::new(300, 200, 160, 108).apply(&src, &mut first);
        Resampler::new(300, 200, 160, 108).apply(&src, &mut second);
        assert_eq!(first, second, "the same input must give the same bytes");
    }

    #[test]
    fn an_upscale_does_not_invent_values_outside_the_range() {
        let src = ramp(37, 29);
        let mut resampler = Resampler::new(37, 29, 200, 150);
        let mut dst = vec![0u8; 200 * 150 * 3];
        resampler.apply(&src, &mut dst);
        // The source spans 0..255 in red, so ringing is expected; what must not
        // happen is a wrap. Every byte is already 0..255 by type, so the real check
        // is that the extremes are clipped rather than wrapped, which shows up as
        // the blue channel -- constant 128 in the source -- staying put.
        for pixel in dst.chunks_exact(3) {
            assert!(
                (pixel[2] as i32 - 128).abs() <= 2,
                "a constant channel moved to {}",
                pixel[2]
            );
        }
    }
}
