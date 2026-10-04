//! Fixture construction that the tests control sample by sample.
//!
//! The engine's integration tests used to build audio with `lavfi` filter graphs
//! — `sine` sources joined into a 5.1 layout. That turned out to be untrustworthy:
//! `volumedetect` on the result showed the speech in channel 1 rather than the
//! centre, and every channel about 15 dB below what the graph asked for. A test
//! fixture whose contents have to be *verified* before the test can assert
//! anything is not a fixture, it is another thing to debug.
//!
//! So the channels are written here, in Rust, where the content is a fact rather
//! than a claim: the synthesis and the expectations share one piece of code.
//! FFmpeg is then used only to encode and mux what it is handed.

#![allow(dead_code)]

use std::path::Path;

/// Channel masks for the layouts that occur in film material.
pub mod mask {
    pub const MONO: u32 = 0x4; // FC
    pub const STEREO: u32 = 0x3; // FL FR
    pub const SURROUND_5_1: u32 = 0x3F; // FL FR FC LFE BL BR
    pub const SURROUND_7_1: u32 = 0x63F; // FL FR FC LFE BL BR SL SR
}

/// Which channel of a 5.1 track holds what. Kept next to the mask so the two
/// cannot drift apart: `FC` is index 2 here *and* in the header we write.
pub mod channel {
    pub const FL: usize = 0;
    pub const FR: usize = 1;
    pub const FC: usize = 2;
    pub const LFE: usize = 3;
    pub const BL: usize = 4;
    pub const BR: usize = 5;
}

/// A float WAV under construction, one closure per channel.
pub struct WavBuilder {
    rate: u32,
    channels: usize,
    mask: u32,
    seconds: f64,
    samples: Vec<f32>,
}

impl WavBuilder {
    pub fn new(rate: u32, channels: usize, mask: u32, seconds: f64) -> Self {
        let frames = (rate as f64 * seconds).round() as usize;
        WavBuilder {
            rate,
            channels,
            mask,
            seconds,
            samples: vec![0.0; frames * channels],
        }
    }

    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels
    }

    /// Fills one channel from `f(channel, t)`.
    pub fn channel<F: Fn(f64) -> f32>(&mut self, channel: usize, f: F) -> &mut Self {
        assert!(channel < self.channels, "channel {channel} out of range");
        for frame in 0..self.frames() {
            let t = frame as f64 / self.rate as f64;
            self.samples[frame * self.channels + channel] = f(t);
        }
        self
    }

    /// Mean and peak level of one channel, in dBFS, plus the crest factor.
    ///
    /// This is the measurement the tests reason with, and it comes from the same
    /// buffer that gets written: there is no second implementation to disagree
    /// with.
    pub fn levels(&self, channel: usize) -> Levels {
        let mut sum = 0.0f64;
        let mut peak = 0.0f64;
        for frame in 0..self.frames() {
            let value = self.samples[frame * self.channels + channel].abs() as f64;
            sum += value * value;
            peak = peak.max(value);
        }
        let rms = (sum / self.frames().max(1) as f64).sqrt();
        Levels {
            mean_dbfs: dbfs(rms),
            peak_dbfs: dbfs(peak),
        }
    }

    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data_bytes = (self.samples.len() * 4) as u32;
        let mut out = Vec::with_capacity(68 + data_bytes as usize);
        let block_align = (self.channels * 4) as u16;
        let byte_rate = self.rate * block_align as u32;

        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(60u32 + data_bytes).to_le_bytes());
        out.extend_from_slice(b"WAVE");

        // WAVE_FORMAT_EXTENSIBLE: the only way for a WAV to say which channel is
        // which. Without the mask FFmpeg guesses from the channel count, and a
        // fixture that is *probably* 5.1 is not good enough to test a rule that
        // depends on which channel carries the dialogue.
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&40u32.to_le_bytes());
        out.extend_from_slice(&0xFFFEu16.to_le_bytes()); // WAVE_FORMAT_EXTENSIBLE
        out.extend_from_slice(&(self.channels as u16).to_le_bytes());
        out.extend_from_slice(&self.rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes()); // bits per sample
        out.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        out.extend_from_slice(&32u16.to_le_bytes()); // valid bits
        out.extend_from_slice(&self.mask.to_le_bytes());
        // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
        out.extend_from_slice(&[
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
            0x9B, 0x71,
        ]);

        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_bytes.to_le_bytes());
        for sample in &self.samples {
            out.extend_from_slice(&sample.to_le_bytes());
        }
        std::fs::write(path, out)
    }

    pub fn duration_seconds(&self) -> f64 {
        self.seconds
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Levels {
    pub mean_dbfs: f64,
    pub peak_dbfs: f64,
}

impl Levels {
    /// How far the peaks stand above the mean. Speech and other bursty material
    /// is wide; a steady tone is exactly 3 dB, whatever its frequency.
    pub fn crest_db(&self) -> f64 {
        self.peak_dbfs - self.mean_dbfs
    }
}

fn dbfs(amplitude: f64) -> f64 {
    if amplitude <= 1e-12 {
        -240.0
    } else {
        20.0 * amplitude.log10()
    }
}

// ---- signal shapes --------------------------------------------------------

/// A tone.
pub fn tone(frequency: f64, amplitude: f64) -> impl Fn(f64) -> f32 {
    move |t| (amplitude * (2.0 * std::f64::consts::PI * frequency * t).sin()) as f32
}

/// Dialogue-shaped: bursts separated by near-silence.
///
/// Not speech — the detector is a classical band-energy detector, and what it
/// responds to is a band-limited signal that is present for part of the time and
/// absent for the rest. A fixture that claims to be dialogue when it is a
/// continuous tone is how a test ends up measuring the fixture instead of the
/// engine.
pub fn dialogue_bursts(
    frequency: f64,
    amplitude: f64,
    period: f64,
    duty: f64,
    floor: f64,
) -> impl Fn(f64) -> f32 {
    move |t| {
        let phase = t % period;
        let value = if phase < period * duty {
            amplitude * (2.0 * std::f64::consts::PI * frequency * t).sin()
        } else {
            // A floor rather than digital silence: the detector's noise-floor
            // estimate behaves better with something to sit on, and real
            // recordings are never digitally silent between lines.
            floor * (2.0 * std::f64::consts::PI * frequency * 0.1 * t).sin()
        };
        value as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tone_has_a_three_decibel_crest_factor() {
        let mut wav = WavBuilder::new(48_000, 2, mask::STEREO, 1.0);
        wav.channel(0, tone(1_000.0, 0.5));
        wav.channel(1, tone(1_000.0, 0.25));
        let left = wav.levels(0);
        let right = wav.levels(1);
        assert!((left.crest_db() - 3.0).abs() < 0.2, "{left:?}");
        // 0.5 amplitude is -6 dBFS peak and -9 dBFS RMS.
        assert!((left.peak_dbfs + 6.0).abs() < 0.2, "{left:?}");
        assert!((left.mean_dbfs + 9.0).abs() < 0.3, "{left:?}");
        assert!((right.peak_dbfs + 12.0).abs() < 0.2, "{right:?}");
    }

    #[test]
    fn bursts_have_a_wide_crest_factor_and_the_same_peak_as_a_tone() {
        let duty = 1.0 / 3.0;
        let mut wav = WavBuilder::new(48_000, 1, mask::MONO, 3.0);
        wav.channel(0, dialogue_bursts(1_200.0, 0.25, 3.0, duty, 0.0005));
        let levels = wav.levels(0);
        // Peaks are the same as a tone at 0.25, but the mean is much lower,
        // because two thirds of the runtime is nearly silent. The relationship is
        // exact rather than approximate: a tone contributes 3 dB of crest, and
        // being silent for all but `duty` of the time adds 10*log10(1/duty).
        assert!((levels.peak_dbfs + 12.0).abs() < 0.5, "{levels:?}");
        let expected = 3.0 + 10.0 * (1.0 / duty).log10();
        assert!(
            (levels.crest_db() - expected).abs() < 0.5,
            "expected a crest of {expected:.2} dB for a duty cycle of {duty:.3}, got {:.2}",
            levels.crest_db()
        );
    }

    #[test]
    fn the_written_header_is_extensible_and_carries_the_mask() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("six.wav");
        let mut wav = WavBuilder::new(48_000, 6, mask::SURROUND_5_1, 0.5);
        for channel in 0..6 {
            wav.channel(channel, tone(400.0 + channel as f64 * 100.0, 0.1));
        }
        wav.write(&path).expect("write");

        let bytes = std::fs::read(&path).expect("read back");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 0xFFFE);
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 6);
        assert_eq!(
            u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]),
            mask::SURROUND_5_1
        );
        assert_eq!(&bytes[60..64], b"data");
        let data_bytes = u32::from_le_bytes([bytes[64], bytes[65], bytes[66], bytes[67]]);
        assert_eq!(data_bytes as usize, bytes.len() - 68);
        assert_eq!(data_bytes, (48_000 * 6 * 4) / 2);
    }
}
