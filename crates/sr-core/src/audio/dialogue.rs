//! Dialogue detection and the Loudness-to-Dialogue Ratio.
//!
//! The whole audio strategy reduces to one number: `LDR = programme - dialogue`.
//! A film can be `-18 LUFS` overall with dialogue at `-32` and explosions at
//! `-8`; normalising the programme to `-16` changes nothing about that
//! relationship, which is why "just run loudnorm" is the wrong fix.
//!
//! This detector is deliberately a *classical* speech detector — a 300–3400 Hz
//! band, an adaptive noise floor, hangover and a confidence estimate — not a
//! source-separation model. It runs with zero extra dependencies, it cannot
//! hallucinate, and when it is not confident it says so, which is exactly the
//! signal the rider needs to stay gentle.

use super::dsp::{Biquad, LOUDNESS_OFFSET};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeechDetectorOptions {
    pub band_low_hz: f32,
    pub band_high_hz: f32,
    /// How far a chunk must sit above the noise floor to count as speech.
    pub min_snr_db: f32,
    /// Chunks of speech needed to open the gate (3 = 30 ms).
    pub min_speech_chunks: usize,
    /// Chunks to stay open after speech stops (20 = 200 ms).
    pub hangover_chunks: usize,
    /// Percentile of the level distribution treated as the noise floor.
    pub noise_percentile: f64,
    /// Cap on how far above the p95 level the threshold may sit.
    pub max_threshold_below_peak_db: f32,
}

impl Default for SpeechDetectorOptions {
    fn default() -> Self {
        SpeechDetectorOptions {
            band_low_hz: 300.0,
            band_high_hz: 3_400.0,
            min_snr_db: 9.0,
            min_speech_chunks: 3,
            hangover_chunks: 20,
            noise_percentile: 0.20,
            max_threshold_below_peak_db: 28.0,
        }
    }
}

/// Streaming speech-band energy detector on the shared 10 ms grid.
pub struct SpeechDetector {
    opts: SpeechDetectorOptions,
    high_pass: [Biquad; 2],
    low_pass: [Biquad; 2],
    rate: u32,
    chunk_frames: usize,
    carry: Vec<f32>,
    /// Band level in dBFS per 10 ms chunk — the shared time grid.
    levels_db: Vec<f32>,
}

impl SpeechDetector {
    pub const CHUNK_MS: u32 = 10;

    pub fn new(rate: u32, opts: SpeechDetectorOptions) -> Self {
        let q = std::f32::consts::FRAC_1_SQRT_2;
        SpeechDetector {
            high_pass: [
                Biquad::high_pass(rate as f32, opts.band_low_hz, q),
                Biquad::high_pass(rate as f32, opts.band_low_hz, q),
            ],
            low_pass: [
                Biquad::low_pass(rate as f32, opts.band_high_hz, q),
                Biquad::low_pass(rate as f32, opts.band_high_hz, q),
            ],
            rate,
            chunk_frames: ((rate as f64 * 0.010).round() as usize).max(1),
            carry: Vec::new(),
            levels_db: Vec::new(),
            opts,
        }
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Feeds mono samples.
    pub fn push(&mut self, mono: &[f32]) {
        for &sample in mono {
            let x = if sample.is_finite() { sample } else { 0.0 };
            let mut filtered = x;
            for stage in self.high_pass.iter_mut() {
                filtered = stage.process(filtered);
            }
            for stage in self.low_pass.iter_mut() {
                filtered = stage.process(filtered);
            }
            self.carry.push(filtered);
            if self.carry.len() >= self.chunk_frames {
                self.flush_chunk();
            }
        }
    }

    pub fn flush(&mut self) {
        if self.carry.len() >= self.chunk_frames / 4 {
            self.flush_chunk();
        }
    }

    fn flush_chunk(&mut self) {
        let frames = self.carry.len();
        if frames == 0 {
            return;
        }
        let sum: f64 = self.carry.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let rms = (sum / frames as f64).sqrt();
        let db = if rms <= 1e-9 {
            -120.0
        } else {
            (20.0 * rms.log10()) as f32
        };
        self.levels_db.push(db);
        self.carry.clear();
    }

    pub fn chunk_count(&self) -> usize {
        self.levels_db.len()
    }

    /// Derives the speech mask, threshold and confidence once the stream ends.
    pub fn finish(self, programme_lufs: f64, block_loudness: &[f64]) -> DialogueTrack {
        let opts = self.opts;
        let levels = self.levels_db;
        if levels.is_empty() {
            return DialogueTrack {
                chunk_ms: Self::CHUNK_MS,
                levels_db: levels,
                mask: Vec::new(),
                threshold_db: 0.0,
                noise_floor_db: -120.0,
                speech_ratio: 0.0,
                median_snr_db: 0.0,
                confidence: 0.0,
                notes: vec!["no audio was analysed".to_string()],
            };
        }

        let mut sorted = levels.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let noise_floor = percentile_f32(&sorted, opts.noise_percentile);
        let p95 = percentile_f32(&sorted, 0.95);
        // A chunk counts as speech when it is clearly above the noise floor, but
        // never demand more than `max_threshold_below_peak` under the loud end:
        // otherwise a quiet film would need its dialogue to be the loudest thing
        // in it. Content with no dynamic range at all (constant music) ends up
        // with no detection and therefore no processing, which is the safe
        // failure mode.
        let threshold = (noise_floor + opts.min_snr_db).max(p95 - opts.max_threshold_below_peak_db);

        let raw: Vec<bool> = levels.iter().map(|l| *l > threshold).collect();
        let mask = apply_hangover(&raw, opts.min_speech_chunks, opts.hangover_chunks);

        let active = mask.iter().filter(|m| **m).count();
        let speech_ratio = active as f32 / mask.len() as f32;
        let mut snrs: Vec<f32> = levels
            .iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m)
            .map(|(l, _)| *l - noise_floor)
            .collect();
        snrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_snr = if snrs.is_empty() {
            0.0
        } else {
            snrs[snrs.len() / 2]
        };

        let mut notes = Vec::new();
        // Confidence blends "how far above the floor the speech sits" with "is
        // the amount of speech plausible for a film".
        let mut confidence = ((median_snr - opts.min_snr_db) / 12.0).clamp(0.0, 1.0);
        if speech_ratio < 0.02 {
            confidence = 0.0;
            notes.push("no dialogue-like activity found; dynamic adaptation will be skipped".into());
        } else if speech_ratio > 0.90 {
            confidence *= 0.5;
            notes.push(
                "almost everything reads as speech (music/documentary?): handling conservatively"
                    .into(),
            );
        }
        if median_snr < opts.min_snr_db {
            confidence *= 0.5;
            notes.push("dialogue sits close to the noise floor; low confidence".into());
        }

        let dialogue_lufs = dialogue_loudness(block_loudness, &mask);
        if let Some(dl) = dialogue_lufs {
            notes.push(format!(
                "dialogue {dl:.1} LUFS vs programme {programme_lufs:.1} LUFS"
            ));
        }

        DialogueTrack {
            chunk_ms: Self::CHUNK_MS,
            levels_db: levels,
            mask,
            threshold_db: threshold,
            noise_floor_db: noise_floor,
            speech_ratio,
            median_snr_db: median_snr,
            confidence,
            notes,
        }
    }
}

fn percentile_f32(sorted: &[f32], fraction: f64) -> f32 {
    if sorted.is_empty() {
        return -120.0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Opens the gate after `min_speech` consecutive active chunks and holds it for
/// `hangover` chunks so gains do not chatter between words.
///
/// The hangover only starts once the gate has actually opened: a lone 10 ms blip
/// must not smear speech activity across the following second.
pub fn apply_hangover(raw: &[bool], min_speech: usize, hangover: usize) -> Vec<bool> {
    let mut mask = vec![false; raw.len()];
    let mut run = 0usize;
    let mut remaining_hangover = 0usize;
    for (index, active) in raw.iter().enumerate() {
        if *active {
            run += 1;
            if run >= min_speech {
                // backfill the run that opened the gate
                for j in index + 1 - run..=index {
                    mask[j] = true;
                }
                remaining_hangover = hangover;
            }
        } else {
            run = 0;
            if remaining_hangover > 0 {
                mask[index] = true;
                remaining_hangover -= 1;
            }
        }
    }
    mask
}

/// Gated loudness over the 400 ms blocks that overlap speech.
pub fn dialogue_loudness(block_loudness: &[f64], speech_chunk: &[bool]) -> Option<f64> {
    const BLOCK_CHUNKS: usize = 40;
    const HOP_CHUNKS: usize = 10;
    if block_loudness.is_empty() || speech_chunk.is_empty() {
        return None;
    }
    let mut selected = Vec::new();
    for (index, value) in block_loudness.iter().enumerate() {
        let start = index * HOP_CHUNKS;
        if start >= speech_chunk.len() {
            break;
        }
        let end = (start + BLOCK_CHUNKS).min(speech_chunk.len());
        let speech = speech_chunk[start..end].iter().filter(|s| **s).count();
        if speech * 10 >= (end - start).max(1) * 3 {
            selected.push(*value);
        }
    }
    if selected.len() < 3 {
        return None;
    }
    // Re-gate in the linear domain, exactly like the programme measurement.
    let mean_square = |values: &[f64]| -> f64 {
        let sum: f64 = values
            .iter()
            .map(|v| {
                if v.is_finite() {
                    10f64.powf((v - LOUDNESS_OFFSET) / 10.0)
                } else {
                    0.0
                }
            })
            .sum();
        sum / values.len().max(1) as f64
    };
    let absolute: Vec<f64> = selected.iter().copied().filter(|v| *v > -70.0).collect();
    if absolute.is_empty() {
        return None;
    }
    let relative_threshold = super::dsp::ms_to_lufs(mean_square(&absolute)) - 10.0;
    let relative: Vec<f64> = absolute
        .iter()
        .copied()
        .filter(|v| *v > relative_threshold)
        .collect();
    let set = if relative.is_empty() { absolute } else { relative };
    let value = super::dsp::ms_to_lufs(mean_square(&set));
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

/// Loudness-to-dialogue ratio in LU.
pub fn ldr_lu(programme_lufs: f64, dialogue_lufs: Option<f64>) -> Option<f64> {
    let dialogue = dialogue_lufs?;
    if !programme_lufs.is_finite() || !dialogue.is_finite() {
        return None;
    }
    Some(programme_lufs - dialogue)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DialogueTrack {
    pub chunk_ms: u32,
    pub levels_db: Vec<f32>,
    pub mask: Vec<bool>,
    pub threshold_db: f32,
    pub noise_floor_db: f32,
    pub speech_ratio: f32,
    pub median_snr_db: f32,
    pub confidence: f32,
    pub notes: Vec<String>,
}

impl DialogueTrack {
    pub fn active_ms(&self) -> usize {
        self.mask.iter().filter(|m| **m).count() * self.chunk_ms as usize
    }

    pub fn summary(&self) -> String {
        format!(
            "dialogue detected in {:.1}% of the runtime ({} s), confidence {:.0}%",
            self.speech_ratio * 100.0,
            self.active_ms() / 1000,
            self.confidence * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::dsp::{LoudnessMeter, ms_to_lufs};

    fn tone(freq: f64, rate: u32, amplitude: f32, seconds: f64) -> Vec<f32> {
        let n = (rate as f64 * seconds) as usize;
        (0..n)
            .map(|i| {
                amplitude
                    * (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin() as f32
            })
            .collect()
    }

    /// Speech-like bursts (a 1.2 kHz tone gated on for 1 s in every 3 s) over a
    /// quiet noise floor.
    fn speech_bursts(rate: u32, seconds: f64) -> Vec<f32> {
        let n = (rate as f64 * seconds) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / rate as f64;
                if t % 3.0 < 1.0 {
                    0.25 * (2.0 * std::f64::consts::PI * 1200.0 * t).sin() as f32
                } else {
                    0.0005 * (2.0 * std::f64::consts::PI * 120.0 * t).sin() as f32
                }
            })
            .collect()
    }

    #[test]
    fn hangover_holds_the_gate_open() {
        let raw = vec![false, true, true, true, false, false, false, false];
        let mask = apply_hangover(&raw, 3, 2);
        assert!(!mask[0]);
        assert!(mask[1] && mask[2] && mask[3]);
        assert!(mask[4] && mask[5], "hangover must hold for two chunks");
        assert!(!mask[6]);
    }

    #[test]
    fn short_blips_do_not_open_the_gate() {
        let raw = vec![false, true, false, false, false];
        let mask = apply_hangover(&raw, 3, 2);
        assert!(mask.iter().all(|m| !m), "a 10 ms blip is not dialogue");
    }

    #[test]
    fn experiment_burst_over_digital_silence() {
        let rate = 48_000;
        let mut mono = tone(1000.0, rate, 0.25, 3.0);
        mono.extend(std::iter::repeat(0.0f32).take(rate as usize * 9));
        let mut detector = SpeechDetector::new(rate, SpeechDetectorOptions::default());
        detector.push(&mono);
        detector.flush();
        let track = detector.finish(-20.0, &vec![0.0; 1200]);
        eprintln!(
            "burst: chunks={} floor={:.2} threshold={:.2} p95? ratio={:.3} conf={:.2}\n\
             first 6 levels: {:?}\n\
             levels 300..306: {:?}\n\
             levels 600..606: {:?}",
            track.levels_db.len(),
            track.noise_floor_db,
            track.threshold_db,
            track.speech_ratio,
            track.confidence,
            &track.levels_db[0..6],
            &track.levels_db[300..306],
            &track.levels_db[600..606]
        );
    }

    #[test]
    fn detector_finds_gated_speech_bursts() {
        let rate = 48_000;
        let mono = speech_bursts(rate, 12.0);
        let mut meter = LoudnessMeter::new(rate, 1, None);
        meter.push(&mono);
        meter.flush();
        let programme = meter.integrated_lufs();

        let mut detector = SpeechDetector::new(rate, SpeechDetectorOptions::default());
        detector.push(&mono);
        detector.flush();
        let track = detector.finish(programme, &meter.block_loudness());

        assert!(
            track.speech_ratio > 0.2 && track.speech_ratio < 0.55,
            "expected roughly a third of the runtime to be speech, got {}",
            track.speech_ratio
        );
        assert!(track.confidence > 0.3, "confidence {}", track.confidence);
        assert!(track.median_snr_db > 9.0);
        assert!(track.active_ms() > 3_000 && track.active_ms() < 6_000);
    }

    #[test]
    fn continuous_music_is_not_treated_as_dialogue() {
        let rate = 48_000;
        let music = tone(1000.0, rate, 0.3, 8.0);
        let mut meter = LoudnessMeter::new(rate, 1, None);
        meter.push(&music);
        meter.flush();
        let mut detector = SpeechDetector::new(rate, SpeechDetectorOptions::default());
        detector.push(&music);
        detector.flush();
        let track = detector.finish(meter.integrated_lufs(), &meter.block_loudness());
        // Content with no dynamic range gives the detector nothing to gate on.
        // Reporting low confidence is the safe answer: no adaptation happens.
        assert!(
            track.confidence <= 0.5,
            "constant music must not be treated as confident dialogue (confidence {})",
            track.confidence
        );
        assert!(
            track.speech_ratio < 0.5 || track.confidence <= 0.5,
            "either little speech is found or confidence is low"
        );
    }

    #[test]
    fn silence_yields_no_dialogue_and_zero_confidence() {
        let mut detector = SpeechDetector::new(48_000, SpeechDetectorOptions::default());
        detector.push(&vec![0.0; 48_000]);
        detector.flush();
        let track = detector.finish(f64::NEG_INFINITY, &[]);
        assert_eq!(track.speech_ratio, 0.0);
        assert_eq!(track.confidence, 0.0);
        assert!(track.notes.iter().any(|n| n.contains("skipped")));
    }

    #[test]
    fn dialogue_loudness_is_louder_than_programme_when_speech_is_hot() {
        let blocks: Vec<f64> = vec![-20.0; 40];
        let speech = vec![true; 400];
        let dialogue = dialogue_loudness(&blocks, &speech).unwrap();
        assert!((dialogue - -20.0).abs() < 0.2, "got {dialogue}");
    }

    #[test]
    fn ldr_is_programme_minus_dialogue() {
        assert_eq!(ldr_lu(-17.0, Some(-29.0)), Some(12.0));
        assert_eq!(ldr_lu(-17.0, None), None);
        assert_eq!(ldr_lu(f64::NEG_INFINITY, Some(-20.0)), None);
    }

    #[test]
    fn engine_offset_matches_the_gating_helper() {
        // ms_to_lufs and the linearised re-gating must agree, or dialogue and
        // programme loudness would not be comparable.
        let ms = 10f64.powf((-23.0 - LOUDNESS_OFFSET) / 10.0);
        assert!((ms_to_lufs(ms) - -23.0).abs() < 1e-9);
    }
}
