//! Native DSP: K-weighting, ITU-R BS.1770 gated loudness, envelope followers and
//! Linkwitz-Riley crossovers.
//!
//! This is a real implementation of the standard, not an approximation of a
//! meter widget: the LDR decision the whole audio chain hangs on (see
//! [`super::dialogue`]) is only meaningful if programme and dialogue loudness
//! are measured the same, correct way. FFmpeg's `ebur128` is still run as an
//! independent cross-check — if the two disagree by more than half an LU we want
//! to know before we touch a master.

use std::collections::VecDeque;

/// Direct-form-II transposed biquad.
#[derive(Copy, Clone, Debug, Default)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl Biquad {
    pub fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Biquad {
            b0,
            b1,
            b2,
            a1,
            a2,
            s1: 0.0,
            s2: 0.0,
        }
    }

    pub fn identity() -> Self {
        Biquad::new(1.0, 0.0, 0.0, 0.0, 0.0)
    }

    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    /// 2nd-order Butterworth low pass (RBJ cookbook).
    pub fn low_pass(rate: f32, freq: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * (freq / rate).clamp(1e-6, 0.49);
        let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;
        Biquad::new(
            ((1.0 - cos_w0) / 2.0) / a0,
            (1.0 - cos_w0) / a0,
            ((1.0 - cos_w0) / 2.0) / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha) / a0,
        )
    }

    /// 2nd-order Butterworth high pass.
    pub fn high_pass(rate: f32, freq: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * (freq / rate).clamp(1e-6, 0.49);
        let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;
        Biquad::new(
            ((1.0 + cos_w0) / 2.0) / a0,
            (-(1.0 + cos_w0)) / a0,
            ((1.0 + cos_w0) / 2.0) / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha) / a0,
        )
    }
}

/// BS.1770 stage 1: the high-shelf ("head" model).
pub fn k_weighting_shelf(rate: f32) -> Biquad {
    let f0 = 1681.974450955533_f64;
    let gain_db = 3.999843853973347_f64;
    let q = 0.7071752369554196_f64;

    let k = (std::f64::consts::PI * f0 / rate as f64).tan();
    let vh = 10f64.powf(gain_db / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    Biquad::new(
        ((vh + vb * k / q + k * k) / a0) as f32,
        ((2.0 * (k * k - vh)) / a0) as f32,
        ((vh - vb * k / q + k * k) / a0) as f32,
        ((2.0 * (k * k - 1.0)) / a0) as f32,
        ((1.0 - k / q + k * k) / a0) as f32,
    )
}

/// BS.1770 stage 2: the RLB high-pass.
pub fn k_weighting_high_pass(rate: f32) -> Biquad {
    let f0 = 38.13547087602444_f64;
    let q = 0.5003270373238773_f64;
    let k = (std::f64::consts::PI * f0 / rate as f64).tan();
    let a0 = 1.0 + k / q + k * k;
    Biquad::new(
        1.0,
        -2.0,
        1.0,
        ((2.0 * (k * k - 1.0)) / a0) as f32,
        ((1.0 - k / q + k * k) / a0) as f32,
    )
}

/// The two-stage K-weighting filter for one channel.
#[derive(Clone, Debug)]
pub struct KWeighting {
    shelf: Biquad,
    high_pass: Biquad,
}

impl KWeighting {
    pub fn new(rate: u32) -> Self {
        KWeighting {
            shelf: k_weighting_shelf(rate as f32),
            high_pass: k_weighting_high_pass(rate as f32),
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        self.high_pass.process(self.shelf.process(x))
    }
}

/// BS.1770 channel weights. LFE is excluded, surrounds count 1.41.
pub fn channel_weights(layout: Option<&str>, channels: u16) -> Vec<f32> {
    let layout = layout.unwrap_or("").to_ascii_lowercase();
    let has = |needle: &str| layout.contains(needle);
    let mut weights = Vec::with_capacity(channels as usize);
    if channels == 1 {
        return vec![1.0];
    }
    if channels == 2 {
        return vec![1.0, 1.0];
    }
    // Assume FFmpeg's canonical order: FL FR FC LFE BL BR [SL SR] ...
    for index in 0..channels {
        let weight = match index {
            0 | 1 => 1.0,                     // FL, FR
            2 => 1.0,                         // FC
            3 => 0.0,                         // LFE
            _ => {
                if has("2.1") || has("quad") {
                    1.0
                } else {
                    1.41 // surrounds
                }
            }
        };
        weights.push(weight);
    }
    weights
}

pub const LOUDNESS_OFFSET: f64 = -0.691;
/// BS.1770 absolute gate.
pub const ABSOLUTE_GATE_LUFS: f64 = -70.0;
/// BS.1770 relative gate.
pub const RELATIVE_GATE_LU: f64 = -10.0;

pub fn ms_to_lufs(mean_square: f64) -> f64 {
    if mean_square <= 0.0 {
        return f64::NEG_INFINITY;
    }
    LOUDNESS_OFFSET + 10.0 * mean_square.log10()
}

/// Gated integrated loudness from per-block mean-square values (BS.1770-5).
pub fn gated_loudness(blocks: &[f64]) -> f64 {
    if blocks.is_empty() {
        return f64::NEG_INFINITY;
    }
    // Stage 1: absolute gate.
    let above_absolute: Vec<f64> = blocks
        .iter()
        .copied()
        .filter(|ms| ms_to_lufs(*ms) > ABSOLUTE_GATE_LUFS)
        .collect();
    if above_absolute.is_empty() {
        return f64::NEG_INFINITY;
    }
    // Stage 2: relative gate, 10 LU below the mean of what survived stage 1.
    let mean: f64 = above_absolute.iter().sum::<f64>() / above_absolute.len() as f64;
    let relative_threshold = ms_to_lufs(mean) + RELATIVE_GATE_LU;
    let above_relative: Vec<f64> = above_absolute
        .iter()
        .copied()
        .filter(|ms| ms_to_lufs(*ms) > relative_threshold)
        .collect();
    let final_set = if above_relative.is_empty() {
        above_absolute
    } else {
        above_relative
    };
    ms_to_lufs(final_set.iter().sum::<f64>() / final_set.len() as f64)
}

/// Streaming loudness meter.
///
/// Feed interleaved blocks of any size; internally everything is aligned to a
/// 10 ms grid, which is also the grid the dialogue detector uses, so programme
/// and dialogue loudness are always measured over identical time spans.
pub struct LoudnessMeter {
    rate: u32,
    channels: usize,
    weights: Vec<f32>,
    filters: Vec<KWeighting>,
    chunk_frames: usize,
    carry: Vec<f32>,
    chunk_sums: Vec<f64>,
    /// Weighted mean square per 10 ms chunk (the shared time grid).
    chunks: Vec<f64>,
    /// 400 ms block mean squares, one per 100 ms (75% overlap).
    blocks: Vec<f64>,
    sample_peak: f32,
    non_finite: u64,
}

impl LoudnessMeter {
    pub fn new(rate: u32, channels: u16, layout: Option<&str>) -> Self {
        let channels = channels.max(1) as usize;
        let chunk_frames = ((rate as f64 * 0.010).round() as usize).max(1);
        LoudnessMeter {
            rate,
            channels,
            weights: channel_weights(layout, channels as u16),
            filters: (0..channels).map(|_| KWeighting::new(rate)).collect(),
            chunk_frames,
            carry: Vec::with_capacity(chunk_frames * channels),
            chunk_sums: vec![0.0; channels],
            chunks: Vec::new(),
            blocks: Vec::new(),
            sample_peak: 0.0,
            non_finite: 0,
        }
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub const CHUNK_MS: u32 = 10;
    pub const BLOCK_CHUNKS: usize = 40; // 400 ms
    pub const HOP_CHUNKS: usize = 10; // 100 ms

    pub fn sample_peak_dbfs(&self) -> f32 {
        if self.sample_peak <= 0.0 {
            f32::NEG_INFINITY
        } else {
            20.0 * self.sample_peak.log10()
        }
    }

    pub fn non_finite_samples(&self) -> u64 {
        self.non_finite
    }

    /// Feeds interleaved samples.
    pub fn push(&mut self, samples: &[f32]) {
        let ch = self.channels;
        for sample in samples {
            if !sample.is_finite() {
                self.non_finite += 1;
                self.carry.push(0.0);
            } else {
                self.sample_peak = self.sample_peak.max(sample.abs());
                self.carry.push(*sample);
            }
            if self.carry.len() >= self.chunk_frames * ch {
                self.flush_chunk();
            }
        }
    }

    fn flush_chunk(&mut self) {
        let ch = self.channels;
        let frames = self.carry.len() / ch;
        if frames == 0 {
            return;
        }
        for frame in 0..frames {
            for c in 0..ch {
                let x = self.carry[frame * ch + c];
                let y = self.filters[c].process(x) as f64;
                self.chunk_sums[c] += y * y;
            }
        }
        self.carry.clear();

        let mut weighted = 0.0f64;
        for c in 0..ch {
            weighted += self.weights.get(c).copied().unwrap_or(1.0) as f64
                * (self.chunk_sums[c] / frames as f64);
            self.chunk_sums[c] = 0.0;
        }
        self.finish_chunk(weighted);
    }

    /// Closes the current partial chunk (end of stream / testing).
    pub fn flush(&mut self) {
        if !self.carry.is_empty() && self.carry.len() >= self.channels {
            self.flush_chunk();
        }
    }

    fn finish_chunk(&mut self, weighted_ms: f64) {
        self.chunks.push(weighted_ms);
        if self.chunks.len() >= Self::BLOCK_CHUNKS
            && (self.chunks.len() - Self::BLOCK_CHUNKS) % Self::HOP_CHUNKS == 0
        {
            let start = self.chunks.len() - Self::BLOCK_CHUNKS;
            let window = &self.chunks[start..];
            self.blocks.push(window.iter().sum::<f64>() / window.len() as f64);
        }
    }

    /// Block loudnesses in LUFS, one per 100 ms.
    pub fn block_loudness(&self) -> Vec<f64> {
        self.blocks.iter().copied().map(ms_to_lufs).collect()
    }

    /// Indices of `blocks` whose 400 ms window overlaps a speech mask.
    pub fn blocks_where(&self, speech_chunk: &[bool]) -> Vec<usize> {
        let mut out = Vec::new();
        for (index, _) in self.blocks.iter().enumerate() {
            let start = index * Self::HOP_CHUNKS;
            let end = (start + Self::BLOCK_CHUNKS).min(speech_chunk.len());
            if start >= speech_chunk.len() {
                break;
            }
            let speech_chunks = speech_chunk[start..end].iter().filter(|s| **s).count();
            // A block counts as dialogue if at least 30% of its window is speech.
            if speech_chunks * 10 >= (end - start).max(1) * 3 {
                out.push(index);
            }
        }
        out
    }

    pub fn integrated_lufs(&self) -> f64 {
        gated_loudness(&self.blocks)
    }

    /// Gated loudness restricted to dialogue-active blocks.
    pub fn dialogue_lufs(&self, speech_chunk: &[bool]) -> Option<f64> {
        let indices = self.blocks_where(speech_chunk);
        if indices.len() < 3 {
            return None;
        }
        let subset: Vec<f64> = indices.iter().map(|i| self.blocks[*i]).collect();
        let value = gated_loudness(&subset);
        if value.is_finite() {
            Some(value)
        } else {
            None
        }
    }

    /// Loudness range (EBU Tech 3342), from 3 s short-term windows.
    pub fn loudness_range_lu(&self) -> f64 {
        const SHORT_TERM_CHUNKS: usize = 300; // 3 s
        if self.chunks.len() < SHORT_TERM_CHUNKS {
            return 0.0;
        }
        let mut short_term: Vec<f64> = Vec::new();
        let mut index = SHORT_TERM_CHUNKS;
        while index <= self.chunks.len() {
            let window = &self.chunks[index - SHORT_TERM_CHUNKS..index];
            short_term.push(ms_to_lufs(window.iter().sum::<f64>() / window.len() as f64));
            index += Self::HOP_CHUNKS;
        }
        short_term.retain(|v| v.is_finite() && *v > ABSOLUTE_GATE_LUFS);
        if short_term.len() < 2 {
            return 0.0;
        }
        let mean = 10.0 * (short_term.iter().map(|v| 10f64.powf(v / 10.0)).sum::<f64>()
            / short_term.len() as f64)
            .log10();
        let threshold = mean - 20.0;
        let mut kept: Vec<f64> = short_term.into_iter().filter(|v| *v > threshold).collect();
        if kept.len() < 2 {
            return 0.0;
        }
        kept.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let low = percentile(&kept, 0.10);
        let high = percentile(&kept, 0.95);
        (high - low).max(0.0)
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// One-pole envelope follower operating in dB, with separate attack/release and
/// a slew limit. This is what stops a dialogue rider from pumping.
#[derive(Clone, Debug)]
pub struct EnvelopeFollower {
    attack: f32,
    release: f32,
    max_step_db: f32,
    value: Option<f32>,
}

impl EnvelopeFollower {
    pub fn new(block_ms: f32, attack_ms: f32, release_ms: f32, max_slew_db_per_second: f32) -> Self {
        let coeff = |time_ms: f32| {
            if time_ms <= 0.0 {
                1.0
            } else {
                1.0 - (-block_ms / time_ms).exp()
            }
        };
        EnvelopeFollower {
            attack: coeff(attack_ms),
            release: coeff(release_ms),
            max_step_db: max_slew_db_per_second * block_ms / 1000.0,
            value: None,
        }
    }

    pub fn process(&mut self, target_db: f32) -> f32 {
        let target = if target_db.is_finite() { target_db } else { 0.0 };
        let next = match self.value {
            None => target,
            Some(current) => {
                let coeff = if target > current {
                    self.attack
                } else {
                    self.release
                };
                let wanted = current + (target - current) * coeff;
                let delta = (wanted - current).clamp(-self.max_step_db, self.max_step_db);
                current + delta
            }
        };
        self.value = Some(next);
        next
    }

    pub fn value(&self) -> Option<f32> {
        self.value
    }
}

/// 4th-order Linkwitz-Riley split into low / mid / high bands.
///
/// LR4 is two cascaded Butterworth sections, which makes the two-way sum
/// (low + high) an allpass. The *three-way* sum is close to, but not exactly,
/// allpass, so callers that want to attenuate one band should **subtract** that
/// band (`out = x - mid * (1 - gain)`) rather than resumming the three bands:
/// subtracting leaves everything the crossover did not identify untouched, and
/// is bit-exact unity when no attenuation is applied.
pub struct ThreeBandSplitter {
    low_lo: [Biquad; 2],
    high_lo: [Biquad; 2],
    low_hi: [Biquad; 2],
    high_hi: [Biquad; 2],
}

impl ThreeBandSplitter {
    pub fn new(rate: u32, low_hz: f32, high_hz: f32) -> Self {
        let q = std::f32::consts::FRAC_1_SQRT_2;
        let (r, l, h) = (rate as f32, low_hz, high_hz);
        ThreeBandSplitter {
            low_lo: [Biquad::low_pass(r, l, q), Biquad::low_pass(r, l, q)],
            high_lo: [Biquad::high_pass(r, l, q), Biquad::high_pass(r, l, q)],
            low_hi: [Biquad::low_pass(r, h, q), Biquad::low_pass(r, h, q)],
            high_hi: [Biquad::high_pass(r, h, q), Biquad::high_pass(r, h, q)],
        }
    }

    /// Splits one channel's samples into three bands (all same length).
    pub fn split(&mut self, input: &[f32], low: &mut Vec<f32>, mid: &mut Vec<f32>, high: &mut Vec<f32>) {
        low.clear();
        mid.clear();
        high.clear();
        low.reserve(input.len());
        mid.reserve(input.len());
        high.reserve(input.len());
        for &x in input {
            let mut band_low = x;
            for stage in self.low_lo.iter_mut() {
                band_low = stage.process(band_low);
            }
            let mut below_high = x;
            for stage in self.high_lo.iter_mut() {
                below_high = stage.process(below_high);
            }
            let mut band_mid = below_high;
            for stage in self.low_hi.iter_mut() {
                band_mid = stage.process(band_mid);
            }
            let mut band_high = below_high;
            for stage in self.high_hi.iter_mut() {
                band_high = stage.process(band_high);
            }
            low.push(band_low);
            mid.push(band_mid);
            high.push(band_high);
        }
    }
}

/// Sample peak in dBFS (true peak is measured by FFmpeg's ebur128 for QC).
pub fn sample_peak_dbfs(samples: &[f32]) -> f32 {
    let peak = samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
    if peak <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * peak.log10()
    }
}

/// Sliding speech-probability smoother used by the rider.
pub fn smooth_mask(mask: &[bool], window: usize) -> Vec<f32> {
    if mask.is_empty() {
        return Vec::new();
    }
    let window = window.max(1);
    let mut out = Vec::with_capacity(mask.len());
    let mut queue: VecDeque<f32> = VecDeque::with_capacity(window);
    let mut sum = 0.0f32;
    for &flag in mask {
        queue.push_back(if flag { 1.0 } else { 0.0 });
        sum += if flag { 1.0 } else { 0.0 };
        if queue.len() > window {
            sum -= queue.pop_front().unwrap_or(0.0);
        }
        out.push(sum / queue.len() as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, rate: u32, amplitude: f32, seconds: f64) -> Vec<f32> {
        let n = (rate as f64 * seconds) as usize;
        (0..n)
            .map(|i| {
                amplitude
                    * (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin() as f32
            })
            .collect()
    }

    #[test]
    fn k_weighting_reproduces_the_published_48k_coefficients() {
        let shelf = k_weighting_shelf(48_000.0);
        assert!((shelf.b0 - 1.53512485958697).abs() < 1e-6, "b0={}", shelf.b0);
        assert!((shelf.b1 - (-2.69169618940638)).abs() < 1e-6, "b1={}", shelf.b1);
        assert!((shelf.b2 - 1.19839281085285).abs() < 1e-6, "b2={}", shelf.b2);
        assert!((shelf.a1 - (-1.69065929318241)).abs() < 1e-6, "a1={}", shelf.a1);
        assert!((shelf.a2 - 0.73248077421585).abs() < 1e-6, "a2={}", shelf.a2);

        let hp = k_weighting_high_pass(48_000.0);
        assert!((hp.b0 - 1.0).abs() < 1e-9);
        assert!((hp.b1 - (-2.0)).abs() < 1e-9);
        assert!((hp.b2 - 1.0).abs() < 1e-9);
        assert!((hp.a1 - (-1.99004745483398)).abs() < 1e-6, "a1={}", hp.a1);
        assert!((hp.a2 - 0.99007225036621).abs() < 1e-6, "a2={}", hp.a2);
    }

    #[test]
    fn loudness_scales_with_amplitude_as_expected() {
        let mut quiet = LoudnessMeter::new(48_000, 1, None);
        quiet.push(&sine(997.0, 48_000, 0.1, 2.0));
        quiet.flush();
        let mut loud = LoudnessMeter::new(48_000, 1, None);
        loud.push(&sine(997.0, 48_000, 0.2, 2.0));
        loud.flush();
        let delta = loud.integrated_lufs() - quiet.integrated_lufs();
        assert!(
            (delta - 6.02).abs() < 0.1,
            "doubling amplitude must add 6.02 LU, got {delta}"
        );
    }

    #[test]
    fn a_fuller_channel_counts_more_than_a_quiet_one() {
        let mut mono = LoudnessMeter::new(48_000, 1, None);
        mono.push(&sine(997.0, 48_000, 0.1, 1.0));
        mono.flush();
        // two identical channels: +3 LU from the channel sum
        let mut stereo = LoudnessMeter::new(48_000, 2, None);
        let one = sine(997.0, 48_000, 0.1, 1.0);
        let mut interleaved = Vec::with_capacity(one.len() * 2);
        for s in &one {
            interleaved.push(*s);
            interleaved.push(*s);
        }
        stereo.push(&interleaved);
        stereo.flush();
        let delta = stereo.integrated_lufs() - mono.integrated_lufs();
        assert!((delta - 3.01).abs() < 0.1, "expected +3 LU, got {delta}");
    }

    #[test]
    fn silence_is_minus_infinity_not_a_number() {
        let mut meter = LoudnessMeter::new(48_000, 2, None);
        meter.push(&vec![0.0; 48_000 * 2]);
        meter.flush();
        assert!(meter.integrated_lufs().is_infinite());
    }

    #[test]
    fn gating_ignores_quiet_blocks() {
        // A loud section plus a very quiet tail: the tail must not drag the
        // integrated value down (that is the whole point of the gate).
        let mut loud_only = LoudnessMeter::new(48_000, 1, None);
        let mut with_tail = LoudnessMeter::new(48_000, 1, None);
        let loud = sine(997.0, 48_000, 0.5, 5.0);
        let quiet = sine(997.0, 48_000, 0.0005, 20.0);
        loud_only.push(&loud);
        loud_only.flush();
        with_tail.push(&loud);
        with_tail.push(&quiet);
        with_tail.flush();
        let delta = (with_tail.integrated_lufs() - loud_only.integrated_lufs()).abs();
        assert!(delta < 0.5, "quiet tail moved the integrated value by {delta} LU");
    }

    #[test]
    fn channel_weights_exclude_lfe() {
        let w = channel_weights(Some("5.1"), 6);
        assert_eq!(w, vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41]);
        assert_eq!(channel_weights(None, 2), vec![1.0, 1.0]);
        assert_eq!(channel_weights(None, 1), vec![1.0]);
    }

    #[test]
    fn non_finite_samples_are_counted_not_propagated() {
        let mut meter = LoudnessMeter::new(48_000, 1, None);
        meter.push(&[0.5, f32::NAN, f32::INFINITY, 0.5]);
        meter.flush();
        assert_eq!(meter.non_finite_samples(), 2);
        assert!(meter.integrated_lufs().is_finite() || meter.integrated_lufs().is_infinite());
    }

    #[test]
    fn envelope_follower_attacks_faster_than_it_releases() {
        let mut env = EnvelopeFollower::new(100.0, 100.0, 1000.0, 1000.0);
        env.process(0.0);
        let rise = env.process(6.0);
        assert!(rise > 0.0 && rise < 6.0, "one attack block must not jump: {rise}");
        for _ in 0..20 {
            env.process(6.0);
        }
        let settled = env.process(6.0);
        assert!((settled - 6.0).abs() < 0.2, "settled at {settled}");

        let mut release = EnvelopeFollower::new(100.0, 100.0, 1000.0, 1000.0);
        release.process(6.0);
        for _ in 0..10 {
            release.process(6.0);
        }
        let down = release.process(0.0);
        assert!(
            6.0 - down < rise,
            "release must be slower than attack: fell {:.2} dB vs a rise of {rise:.2} dB",
            6.0 - down
        );
    }

    #[test]
    fn slew_limit_caps_the_gain_ramp() {
        let mut env = EnvelopeFollower::new(100.0, 1.0, 1.0, 3.0); // 3 dB per second
        env.process(0.0);
        let next = env.process(30.0);
        assert!(next <= 0.31, "one 100 ms block may not jump more than 0.3 dB");
    }

    #[test]
    fn three_band_splitter_separates_the_bands() {
        let mut splitter = ThreeBandSplitter::new(48_000, 300.0, 6000.0);
        let input = sine(1000.0, 48_000, 0.5, 1.0);
        let (mut low, mut mid, mut high) = (Vec::new(), Vec::new(), Vec::new());
        splitter.split(&input, &mut low, &mut mid, &mut high);
        assert_eq!(low.len(), input.len());

        let start = 2_000;
        let energy =
            |v: &[f32]| -> f64 { v[start..].iter().map(|s| (*s as f64) * (*s as f64)).sum() };
        let mid_energy = energy(&mid);
        let low_energy = energy(&low);
        let high_energy = energy(&high);
        // A 1 kHz tone must land in the 300 Hz - 6 kHz band and nowhere else.
        assert!(
            mid_energy > low_energy * 100.0 && mid_energy > high_energy * 100.0,
            "1 kHz must land in the mid band: low {low_energy}, mid {mid_energy}, high {high_energy}"
        );
        // The three-way sum is close to allpass but not exactly, which is why the
        // rider subtracts a band instead of resumming it. Bound the deviation.
        let summed: Vec<f32> = (start..input.len())
            .map(|i| low[i] + mid[i] + high[i])
            .collect();
        let input_energy = energy(&input);
        let sum_energy = energy(&summed);
        assert!(
            (sum_energy - input_energy).abs() / input_energy < 0.10,
            "band sum must stay close to the input energy: {input_energy} vs {sum_energy}"
        );
    }

    #[test]
    fn mask_smoothing_spreads_speech_activity() {
        let mask = vec![false, true, false, false];
        let smoothed = smooth_mask(&mask, 3);
        assert_eq!(smoothed.len(), 4);
        assert!(smoothed[1] > 0.0 && smoothed[1] <= 1.0);
        assert!(smoothed[2] > 0.0, "hangover must extend past the speech frame");
    }

    #[test]
    fn percentile_is_clamped_at_both_ends() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 1.0), 4.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
}
