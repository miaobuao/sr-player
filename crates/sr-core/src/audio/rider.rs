//! The dialogue rider and the mid-band ducker.
//!
//! What this deliberately is *not*: a compressor. A compressor on the whole mix
//! makes explosions quieter, dialogue louder and destroys the mix's intent. What
//! EBU R128 S4 actually asks for is to adapt the *relationship* between dialogue
//! and programme loudness until the ratio is inside ~5 LU — and if it already
//! is, to do nothing at all.
//!
//! So the chain is:
//!
//! 1. a **static, measurement-derived gain** applied only while dialogue is
//!    present, smoothed with a 120 ms attack / 800 ms release and a slew limit
//!    (a "dialogue level" control, not a compressor);
//! 2. a **–4 dB maximum duck of the 300 Hz–6 kHz band** while dialogue is
//!    present, which is the band that actually masks speech — the low end of an
//!    explosion survives intact;
//! 3. nothing else. The final loudness normalisation and true-peak limiting
//!    happen later, in FFmpeg, after this stage.

use super::dialogue::{ldr_lu, DialogueTrack};
use super::dsp::{EnvelopeFollower, ThreeBandSplitter};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiderSettings {
    /// Dialogue anchor, EBU R128 S4 style.
    pub target_dialogue_lufs: f32,
    /// Do not adapt unless LDR exceeds this.
    pub target_ldr_lu: f32,
    pub max_gain_db: f32,
    pub max_attenuation_db: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub max_slew_db_per_second: f32,
    /// Maximum attenuation applied to the dialogue band (positive dB).
    pub duck_max_db: f32,
    pub duck_band_low_hz: f32,
    pub duck_band_high_hz: f32,
}

impl Default for RiderSettings {
    fn default() -> Self {
        RiderSettings {
            target_dialogue_lufs: -24.0,
            target_ldr_lu: 5.0,
            max_gain_db: 6.0,
            max_attenuation_db: 3.0,
            attack_ms: 120.0,
            release_ms: 800.0,
            // 18 dB/s: a dialogue lift engages within ~250 ms of speech onset,
            // which is fast enough to be useful and slow enough to be inaudible.
            max_slew_db_per_second: 18.0,
            duck_max_db: 4.0,
            duck_band_low_hz: 300.0,
            duck_band_high_hz: 6_000.0,
        }
    }
}

/// What the analysis recommends, and why. The `reason` string goes straight into
/// the log, because "why did it not touch my audio?" must always be answerable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemasterDecision {
    pub apply: bool,
    pub reason: String,
    pub dialogue_gain_db: f32,
    /// Positive dB of attenuation applied to masking content during dialogue.
    pub duck_db: f32,
    pub ldr_before_lu: Option<f64>,
    /// Estimate: the duck only removes part of the programme energy.
    pub ldr_after_lu: Option<f64>,
    pub confidence: f32,
}

impl RemasterDecision {
    pub fn skipped(reason: impl Into<String>, ldr: Option<f64>, confidence: f32) -> Self {
        RemasterDecision {
            apply: false,
            reason: reason.into(),
            dialogue_gain_db: 0.0,
            duck_db: 0.0,
            ldr_before_lu: ldr,
            ldr_after_lu: ldr,
            confidence,
        }
    }

    pub fn summary(&self) -> String {
        if !self.apply {
            return format!("no dynamic adaptation: {}", self.reason);
        }
        format!(
            "dialogue {:+.1} dB, masking band -{:.1} dB, LDR {:.1} → ~{:.1} LU ({})",
            self.dialogue_gain_db,
            self.duck_db,
            self.ldr_before_lu.unwrap_or(0.0),
            self.ldr_after_lu.unwrap_or(0.0),
            self.reason
        )
    }
}

/// The EBU R128 S4 decision.
pub fn decide(
    programme_lufs: f64,
    dialogue_lufs: Option<f64>,
    confidence: f32,
    speech_ratio: f32,
    settings: &RiderSettings,
) -> RemasterDecision {
    let ldr = ldr_lu(programme_lufs, dialogue_lufs);
    let confidence = confidence.clamp(0.0, 1.0);

    let dialogue = match dialogue_lufs {
        Some(d) if d.is_finite() => d,
        _ => {
            return RemasterDecision::skipped(
                "no measurable dialogue loudness to anchor to",
                ldr,
                confidence,
            )
        }
    };
    if speech_ratio < 0.02 {
        return RemasterDecision::skipped(
            "virtually no dialogue in this programme",
            ldr,
            confidence,
        );
    }
    let ldr = match ldr {
        Some(v) => v,
        None => {
            return RemasterDecision::skipped(
                "programme loudness is not measurable",
                None,
                confidence,
            )
        }
    };
    if ldr <= settings.target_ldr_lu as f64 {
        return RemasterDecision::skipped(
            format!(
                "LDR {ldr:.1} LU is already within the {:.1} LU target (EBU R128 S4: do not adapt further)",
                settings.target_ldr_lu
            ),
            Some(ldr),
            confidence,
        );
    }

    let wanted = settings.target_dialogue_lufs as f64 - dialogue;
    let clamped = wanted.clamp(
        -(settings.max_attenuation_db as f64),
        settings.max_gain_db as f64,
    ) as f32;
    // Low confidence means a gentle touch, never a confident mistake.
    let gain = (clamped * confidence).clamp(
        -settings.max_attenuation_db,
        settings.max_gain_db,
    );
    let excess = (ldr - settings.target_ldr_lu as f64).max(0.0) as f32;
    let duck = (excess.min(settings.duck_max_db) * confidence).clamp(0.0, settings.duck_max_db);

    let reason = if confidence < 0.5 {
        format!("LDR {ldr:.1} LU above target; applying a reduced correction (confidence {:.0}%)", confidence * 100.0)
    } else {
        format!("LDR {ldr:.1} LU above the {:.1} LU target", settings.target_ldr_lu)
    };

    RemasterDecision {
        apply: true,
        reason,
        dialogue_gain_db: gain,
        duck_db: duck,
        ldr_before_lu: Some(ldr),
        // Raising dialogue by g and removing part of the masking programme
        // energy both reduce LDR; the factor approximates the mid-band share.
        ldr_after_lu: Some(ldr - gain as f64 - duck as f64 * 0.5),
        confidence,
    }
}

/// Per-chunk gain and duck schedule, on the same 10 ms grid as the detector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GainCurve {
    pub chunk_ms: u32,
    pub gains_db: Vec<f32>,
    pub duck_db: Vec<f32>,
}

impl GainCurve {
    pub fn flat(chunks: usize, chunk_ms: u32) -> Self {
        GainCurve {
            chunk_ms,
            gains_db: vec![0.0; chunks],
            duck_db: vec![0.0; chunks],
        }
    }

    pub fn from_track(
        track: &DialogueTrack,
        decision: &RemasterDecision,
        settings: &RiderSettings,
    ) -> Self {
        let block_ms = track.chunk_ms.max(1) as f32;
        let mut rider = EnvelopeFollower::new(
            block_ms,
            settings.attack_ms,
            settings.release_ms,
            settings.max_slew_db_per_second,
        );
        let mut ducker = EnvelopeFollower::new(
            block_ms,
            settings.attack_ms,
            settings.release_ms,
            settings.max_slew_db_per_second,
        );
        // Start from "do nothing" so the first speech block ramps in.
        rider.process(0.0);
        ducker.process(0.0);

        let mut gains_db = Vec::with_capacity(track.mask.len());
        let mut duck_db = Vec::with_capacity(track.mask.len());
        for active in &track.mask {
            let speech = *active && decision.apply;
            gains_db.push(rider.process(if speech { decision.dialogue_gain_db } else { 0.0 }));
            duck_db.push(ducker.process(if speech { decision.duck_db } else { 0.0 }));
        }
        GainCurve {
            chunk_ms: track.chunk_ms,
            gains_db,
            duck_db,
        }
    }

    pub fn len(&self) -> usize {
        self.gains_db.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gains_db.is_empty()
    }

    pub fn gain_at(&self, chunk: usize) -> f32 {
        self.gains_db.get(chunk).copied().unwrap_or(0.0)
    }

    pub fn duck_at(&self, chunk: usize) -> f32 {
        self.duck_db.get(chunk).copied().unwrap_or(0.0)
    }

    pub fn gain_linear(&self, chunk: usize) -> f32 {
        10f32.powf(self.gain_at(chunk) / 20.0)
    }

    pub fn duck_linear(&self, chunk: usize) -> f32 {
        10f32.powf(-self.duck_at(chunk) / 20.0)
    }

    pub fn max_gain_db(&self) -> f32 {
        self.gains_db.iter().copied().fold(f32::MIN, f32::max)
    }

    pub fn max_duck_db(&self) -> f32 {
        self.duck_db.iter().copied().fold(0.0f32, f32::max)
    }
}

/// Applies a [`GainCurve`] to interleaved PCM, in place, block by block.
pub struct RemasterProcessor {
    curve: GainCurve,
    channels: usize,
    chunk_frames: usize,
    chunk_index: usize,
    frame_in_chunk: usize,
    /// Channels whose 300 Hz–6 kHz band is ducked. Surrounds only on 5.1+, so
    /// the front stage keeps its tonal balance.
    duck_channels: Vec<bool>,
    splitters: Vec<ThreeBandSplitter>,
    scratch_channel: Vec<f32>,
    scratch_low: Vec<f32>,
    scratch_mid: Vec<f32>,
    scratch_high: Vec<f32>,
    frames_processed: u64,
    peak_gain_linear: f32,
}

impl RemasterProcessor {
    pub fn new(curve: GainCurve, channels: u16, rate: u32, settings: &RiderSettings) -> Self {
        let channels = channels.max(1) as usize;
        // Stereo (and mono): duck the band on every channel. Multichannel: ducks
        // the surrounds, leaving centre/front untouched.
        let duck_channels = if channels <= 2 {
            vec![true; channels]
        } else {
            (0..channels).map(|c| c >= 4).collect()
        };
        let chunk_frames =
            ((rate as f64 * curve.chunk_ms.max(1) as f64 / 1000.0).round() as usize).max(1);
        RemasterProcessor {
            splitters: (0..channels)
                .map(|_| ThreeBandSplitter::new(rate, settings.duck_band_low_hz, settings.duck_band_high_hz))
                .collect(),
            curve,
            channels,
            chunk_frames,
            chunk_index: 0,
            frame_in_chunk: 0,
            duck_channels,
            scratch_channel: Vec::new(),
            scratch_low: Vec::new(),
            scratch_mid: Vec::new(),
            scratch_high: Vec::new(),
            frames_processed: 0,
            peak_gain_linear: 1.0,
        }
    }

    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    pub fn peak_gain_linear(&self) -> f32 {
        self.peak_gain_linear
    }

    pub fn chunk_index(&self) -> usize {
        self.chunk_index
    }

    /// Processes interleaved samples in place. Block sizes need not align to the
    /// chunk grid; state is carried across calls.
    pub fn process_block(&mut self, data: &mut [f32]) {
        let ch = self.channels;
        if ch == 0 || data.is_empty() {
            return;
        }
        let mut splitters = std::mem::take(&mut self.splitters);
        let mut channel_buf = std::mem::take(&mut self.scratch_channel);
        let mut low = std::mem::take(&mut self.scratch_low);
        let mut mid = std::mem::take(&mut self.scratch_mid);
        let mut high = std::mem::take(&mut self.scratch_high);

        let mut position = 0usize;
        while position < data.len() {
            let frames_available = (data.len() - position) / ch;
            if frames_available == 0 {
                break;
            }
            let frames = (self.chunk_frames - self.frame_in_chunk).min(frames_available);
            let gain = self.curve.gain_linear(self.chunk_index);
            let duck = self.curve.duck_linear(self.chunk_index);
            self.peak_gain_linear = self.peak_gain_linear.max(gain);

            for channel in 0..ch {
                channel_buf.clear();
                channel_buf.extend(
                    (0..frames).map(|frame| data[position + frame * ch + channel] * gain),
                );

                if self.duck_channels.get(channel).copied().unwrap_or(false) && duck < 0.999 {
                    // Subtract the attenuated band instead of resumming: the
                    // result is exact unity when nothing is ducked, and anything
                    // the crossover did not identify as mid-band stays untouched.
                    splitters[channel].split(&channel_buf, &mut low, &mut mid, &mut high);
                    let reduction = 1.0 - duck;
                    for index in 0..frames {
                        channel_buf[index] -= mid[index] * reduction;
                    }
                }

                for (frame, value) in channel_buf.iter().enumerate() {
                    data[position + frame * ch + channel] = *value;
                }
            }

            position += frames * ch;
            self.frames_processed += frames as u64;
            self.frame_in_chunk += frames;
            if self.frame_in_chunk >= self.chunk_frames {
                self.frame_in_chunk = 0;
                self.chunk_index += 1;
            }
        }

        self.splitters = splitters;
        self.scratch_channel = channel_buf;
        self.scratch_low = low;
        self.scratch_mid = mid;
        self.scratch_high = high;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::dialogue::{DialogueTrack, SpeechDetector};

    fn decision_for(programme: f64, dialogue: Option<f64>, confidence: f32) -> RemasterDecision {
        decide(programme, dialogue, confidence, 0.4, &RiderSettings::default())
    }

    #[test]
    fn a_well_behaved_film_is_left_alone() {
        // -17 programme, -20 dialogue: LDR 3 LU, already inside the target.
        let decision = decision_for(-17.0, Some(-20.0), 1.0);
        assert!(!decision.apply);
        assert!(decision.reason.contains("EBU R128 S4"));
        assert_eq!(decision.dialogue_gain_db, 0.0);
        assert_eq!(decision.ldr_before_lu, decision.ldr_after_lu);
    }

    #[test]
    fn a_cinema_mix_with_shouting_explosions_gets_a_dialogue_lift() {
        // -17 programme, -29 dialogue: LDR 12 LU, the classic case.
        let decision = decision_for(-17.0, Some(-29.0), 1.0);
        assert!(decision.apply);
        assert!(
            (decision.dialogue_gain_db - 5.0).abs() < 0.01,
            "wanted +5 dB, got {}",
            decision.dialogue_gain_db
        );
        assert!(decision.duck_db > 0.0 && decision.duck_db <= 4.0);
        assert!(decision.ldr_after_lu.unwrap() < decision.ldr_before_lu.unwrap());
    }

    #[test]
    fn dialogue_louder_than_the_anchor_is_attenuated_not_boosted() {
        // LDR 8 LU (dialogue *louder* than the dialogue anchor): the rider must
        // pull dialogue down, never boost it.
        let decision = decision_for(-8.0, Some(-16.0), 1.0);
        assert!(decision.apply);
        assert!(decision.dialogue_gain_db < 0.0);
        assert!(
            decision.dialogue_gain_db >= -3.0,
            "attenuation is capped at -3 dB, got {}",
            decision.dialogue_gain_db
        );
    }

    #[test]
    fn gain_and_duck_are_capped_by_the_settings() {
        let decision = decision_for(-5.0, Some(-40.0), 1.0);
        assert!(decision.dialogue_gain_db <= 6.0);
        assert!(decision.duck_db <= 4.0);
    }

    #[test]
    fn low_confidence_means_a_gentle_touch() {
        let confident = decision_for(-17.0, Some(-29.0), 1.0);
        let unsure = decision_for(-17.0, Some(-29.0), 0.3);
        assert!(unsure.apply);
        assert!(unsure.dialogue_gain_db < confident.dialogue_gain_db);
        assert!(unsure.duck_db < confident.duck_db);
        assert!(unsure.reason.contains("reduced correction"));
    }

    #[test]
    fn no_dialogue_measurement_means_no_processing() {
        let decision = decision_for(-17.0, None, 0.9);
        assert!(!decision.apply);
        assert!(decision.reason.contains("no measurable dialogue"));
    }

    #[test]
    fn gain_is_applied_only_while_dialogue_is_present() {
        let track = DialogueTrack {
            chunk_ms: 10,
            levels_db: vec![-20.0; 100],
            mask: {
                let mut mask = vec![false; 100];
                for slot in mask.iter_mut().take(50).skip(10) {
                    *slot = true;
                }
                mask
            },
            threshold_db: -40.0,
            noise_floor_db: -50.0,
            speech_ratio: 0.4,
            median_snr_db: 20.0,
            confidence: 1.0,
            notes: vec![],
        };
        let settings = RiderSettings::default();
        let decision = decision_for(-17.0, Some(-29.0), 1.0);
        let curve = GainCurve::from_track(&track, &decision, &settings);

        assert_eq!(curve.len(), 100);
        assert_eq!(curve.gain_at(0), 0.0, "starts at unity");
        assert!(
            curve.gain_at(49) > 3.0,
            "should reach most of the gain inside speech, got {}",
            curve.gain_at(49)
        );
        assert_eq!(curve.duck_at(0), 0.0);
        assert!(curve.max_duck_db() > 1.0);
        // Every step respects the slew limit (18 dB/s at a 10 ms chunk = 0.18 dB).
        for pair in curve.gains_db.windows(2) {
            assert!(
                (pair[1] - pair[0]).abs() <= 0.181,
                "slew limit violated: {} -> {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn a_skipped_decision_produces_a_flat_curve() {
        let track = DialogueTrack {
            chunk_ms: 10,
            levels_db: vec![-20.0; 20],
            mask: vec![true; 20],
            threshold_db: -40.0,
            noise_floor_db: -50.0,
            speech_ratio: 1.0,
            median_snr_db: 20.0,
            confidence: 1.0,
            notes: vec![],
        };
        let decision = decision_for(-17.0, Some(-20.0), 1.0);
        let curve = GainCurve::from_track(&track, &decision, &RiderSettings::default());
        assert_eq!(curve.max_gain_db(), 0.0);
        assert_eq!(curve.max_duck_db(), 0.0);
    }

    #[test]
    fn processor_boosts_only_the_speech_region() {
        let settings = RiderSettings::default();
        let mut track = DialogueTrack {
            chunk_ms: 10,
            levels_db: vec![-20.0; 200],
            mask: vec![false; 200],
            threshold_db: -40.0,
            noise_floor_db: -50.0,
            speech_ratio: 0.5,
            median_snr_db: 20.0,
            confidence: 1.0,
            notes: vec![],
        };
        for slot in track.mask.iter_mut().take(150).skip(50) {
            *slot = true;
        }
        let decision = decision_for(-17.0, Some(-29.0), 1.0);
        let curve = GainCurve::from_track(&track, &decision, &settings);
        let rate = 48_000u32;
        let mut processor = RemasterProcessor::new(curve, 1, rate, &settings);

        // Five seconds of a constant tone; 10 ms chunks put speech (chunks
        // 50..150) at samples 24_000..72_000. 100 Hz sits *below* the ducked
        // band, so this test isolates the rider gain from the duck.
        let total = rate as usize * 5;
        let mut data: Vec<f32> = (0..total)
            .map(|i| 0.2 * (2.0 * std::f64::consts::PI * 100.0 * i as f64 / rate as f64).sin() as f32)
            .collect();
        let original = data.clone();
        // Deliberately awkward block sizes: the processor must carry chunk state
        // across calls that do not align to the 10 ms grid.
        let mut position = 0usize;
        for block in [777usize, 1_000, 4_096] {
            let end = (position + block).min(total);
            processor.process_block(&mut data[position..end]);
            position = end;
        }
        processor.process_block(&mut data[position..]);

        fn rms(samples: &[f32]) -> f32 {
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
        }
        let before = rms(&original[0..20_000]);
        let during = rms(&data[40_000..70_000]);
        let late = rms(&data[170_000..230_000]);

        assert!(
            during > before * 1.6,
            "speech region should be boosted: {before} -> {during}"
        );
        assert!(
            (late - before).abs() < before * 0.1,
            "gain must return to unity after speech: {before} -> {late}"
        );
        assert!(data.iter().all(|s| s.is_finite()));
        assert_eq!(processor.frames_processed(), total as u64);
    }

    #[test]
    fn processor_ducks_the_mid_band_on_a_stereo_mix() {
        let settings = RiderSettings::default();
        let track = DialogueTrack {
            chunk_ms: 10,
            levels_db: vec![-20.0; 100],
            mask: vec![true; 100],
            threshold_db: -40.0,
            noise_floor_db: -50.0,
            speech_ratio: 1.0,
            median_snr_db: 25.0,
            confidence: 1.0,
            notes: vec![],
        };
        let decision = decision_for(-17.0, Some(-29.0), 1.0);
        let curve = GainCurve::from_track(&track, &decision, &settings);
        let rate = 48_000u32;
        let mut processor = RemasterProcessor::new(curve, 2, rate, &settings);

        // A 1 kHz tone sits inside the ducked band; a 60 Hz tone does not.
        let frames = 48_000;
        let mut interleaved = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let t = i as f64 / rate as f64;
            let mid = 0.2 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32;
            let low = 0.2 * (2.0 * std::f64::consts::PI * 60.0 * t).sin() as f32;
            interleaved.push(mid);
            interleaved.push(low);
        }
        let before_mid = interleaved[40_000];
        let before_low = interleaved[40_001];
        processor.process_block(&mut interleaved);
        let after_mid = interleaved[40_000];
        let after_low = interleaved[40_001];
        // Gain rises over the first chunks, so compare the tail where it has settled.
        let tail_mid: f32 = interleaved[80_000..90_000].iter().step_by(2).map(|s| s.abs()).sum();
        let tail_low: f32 = interleaved[80_001..90_000].iter().step_by(2).map(|s| s.abs()).sum();
        assert!(tail_mid.is_finite() && tail_low.is_finite());
        let _ = (before_mid, before_low, after_mid, after_low);
        // The low band keeps more energy than the ducked mid band once the duck
        // has settled (4 dB of duck plus the +5 dB rider gain on both).
        assert!(
            tail_low > tail_mid * 1.4,
            "low band should survive the mid-band duck: low {tail_low} vs mid {tail_mid}"
        );
    }

    #[test]
    fn speech_detector_and_rider_agree_end_to_end() {
        // Synthetic "film": quiet dialogue segments separated by loud effects.
        let rate = 48_000u32;
        let seconds = 12.0;
        let frames = (rate as f64 * seconds) as usize;
        let mut mono = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = i as f64 / rate as f64;
            // 1 s of dialogue every 3 s; the effects sit at 80 Hz, outside the
            // speech band, exactly like the low end of an explosion.
            let dialogue = t % 3.0 < 1.0;
            let sample = if dialogue {
                0.05 * (2.0 * std::f64::consts::PI * 1200.0 * t).sin()
            } else {
                0.8 * (2.0 * std::f64::consts::PI * 80.0 * t).sin()
            };
            mono.push(sample as f32);
        }
        let mut meter = crate::audio::dsp::LoudnessMeter::new(rate, 1, None);
        meter.push(&mono);
        meter.flush();
        let programme = meter.integrated_lufs();
        let mut detector = SpeechDetector::new(rate, Default::default());
        detector.push(&mono);
        detector.flush();
        let track = detector.finish(programme, &meter.block_loudness());
        assert!(
            track.speech_ratio > 0.15,
            "the detector should have found the dialogue segments: {}",
            track.summary()
        );
        let dialogue_lufs =
            crate::audio::dialogue::dialogue_loudness(&meter.block_loudness(), &track.mask);
        let decision = decide(
            programme,
            dialogue_lufs,
            track.confidence,
            track.speech_ratio,
            &RiderSettings::default(),
        );
        // Whatever the analysis concluded, the decision must be well formed and
        // must never make the ratio worse.
        assert!(decision.reason.len() > 10);
        if decision.apply {
            let before = decision.ldr_before_lu.expect("LDR must be known when applying");
            let after = decision.ldr_after_lu.expect("predicted LDR must be reported");
            assert!(
                after < before,
                "a correction must reduce LDR: {before} -> {after}"
            );
            assert!(decision.dialogue_gain_db.abs() <= 6.0);
            assert!(decision.duck_db <= 4.0);
        }
    }
}
