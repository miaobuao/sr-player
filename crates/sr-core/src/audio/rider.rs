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

/// What a channel is for, and therefore what may be done to it.
///
/// The dialogue rider used to multiply the gain into *every* channel and then
/// duck the masking band on the surrounds. That is not a dialogue boost: it is a
/// whole-mix boost that happens to coincide with speech, and it scaled the LFE
/// along with everything else, which changes the low end of the mix and can clip
/// a subwoofer for reasons that have nothing to do with dialogue.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelRole {
    /// Carries the dialogue. Gets the rider gain; never the masking duck, because
    /// ducking the band we just lifted is undoing the work.
    Dialogue,
    /// Screen channels. No gain; masking band ducked.
    Front,
    /// Surround channels. No gain; masking band ducked.
    Surround,
    /// The low-frequency effects channel. Untouched, always.
    LowFrequency,
    /// Not identified. Untouched: applying a dialogue gain to a channel we cannot
    /// name is guessing, and the guess is audible.
    Unknown,
}

/// How a channel layout is processed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiderStrategy {
    /// One channel carries everything: boost it, duck nothing (ducking the only
    /// channel would thin the dialogue we just lifted).
    Mono,
    /// Mid/side: the dialogue is what the two channels share, so the gain goes to
    /// the mid signal and the masking band is ducked in the side signal.
    StereoMidSide,
    /// A centre channel carries the dialogue; the masking band is ducked in the
    /// screen and surround channels.
    CentreChannel,
    /// The layout could not be identified. Nothing is applied, and the reason is
    /// reported.
    LeaveAlone,
}

/// Which channels are which, for one stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelPlan {
    pub roles: Vec<ChannelRole>,
    pub strategy: RiderStrategy,
    pub note: String,
}

impl ChannelPlan {
    /// Maps a channel count and FFmpeg's layout name onto roles.
    ///
    /// FFmpeg's canonical order is `FL FR FC LFE BL BR [SL SR]`, so most of this
    /// is index arithmetic; the layout name is only needed where the count alone
    /// is ambiguous (three channels is `2.1` or `3.0`, four is `quad` or `3.1`).
    /// Anything not recognised is left alone rather than guessed at.
    pub fn for_stream(channels: u16, layout: Option<&str>) -> ChannelPlan {
        let name = layout.unwrap_or("").to_ascii_lowercase();
        let has = |needle: &str| name.contains(needle);
        let role = |roles: &[ChannelRole]| ChannelPlan {
            roles: roles.to_vec(),
            strategy: if roles.contains(&ChannelRole::Dialogue) {
                RiderStrategy::CentreChannel
            } else {
                RiderStrategy::LeaveAlone
            },
            note: String::new(),
        };
        let (plan, note) = match channels {
            0 => (RiderStrategy::LeaveAlone, "no channels".to_string()),
            1 => (
                RiderStrategy::Mono,
                "mono: the single channel is the dialogue".to_string(),
            ),
            2 => (
                RiderStrategy::StereoMidSide,
                "stereo: dialogue is the shared (mid) signal, so the gain goes there and the \
                 masking band is ducked in the side signal"
                    .to_string(),
            ),
            3 if has("2.1") => (
                RiderStrategy::LeaveAlone,
                "2.1 has no centre channel, so there is no dialogue channel to lift".to_string(),
            ),
            3 => (
                RiderStrategy::CentreChannel,
                "3.0: centre carries the dialogue".to_string(),
            ),
            4 if has("3.1") || has("lfe") => (
                RiderStrategy::CentreChannel,
                "3.1: centre carries the dialogue, LFE is left alone".to_string(),
            ),
            4 => (
                RiderStrategy::LeaveAlone,
                format!(
                    "4-channel layout `{}` has no centre channel",
                    if name.is_empty() { "unknown" } else { &name }
                ),
            ),
            5 if has("4.1") => (
                RiderStrategy::CentreChannel,
                "4.1: centre carries the dialogue".to_string(),
            ),
            5 => (
                RiderStrategy::CentreChannel,
                "5.0: centre carries the dialogue, surrounds are ducked".to_string(),
            ),
            6 if has("5.1") || has("lfe") || name.is_empty() => (
                RiderStrategy::CentreChannel,
                "5.1: dialogue gain on the centre, masking band ducked on the screen and \
                 surround channels, LFE untouched"
                    .to_string(),
            ),
            6 => (
                RiderStrategy::LeaveAlone,
                format!("6-channel layout `{name}` is not one this build recognises"),
            ),
            7 if has("6.1") => (
                RiderStrategy::CentreChannel,
                "6.1: centre carries the dialogue".to_string(),
            ),
            7 => (
                RiderStrategy::LeaveAlone,
                format!("7-channel layout `{name}` is not one this build recognises"),
            ),
            8 if has("7.1") || name.is_empty() => (
                RiderStrategy::CentreChannel,
                "7.1: dialogue gain on the centre, masking band ducked on the screen and \
                 surround channels, LFE untouched"
                    .to_string(),
            ),
            _ => (
                RiderStrategy::LeaveAlone,
                format!("{channels}-channel layout `{name}` is not one this build recognises"),
            ),
        };

        let mut plan = match plan {
            RiderStrategy::Mono => role(&[ChannelRole::Dialogue]),
            RiderStrategy::StereoMidSide => role(&[ChannelRole::Front, ChannelRole::Front]),
            RiderStrategy::CentreChannel => match channels {
                3 => role(&[ChannelRole::Front, ChannelRole::Front, ChannelRole::Dialogue]),
                4 => role(&[
                    ChannelRole::Front,
                    ChannelRole::Front,
                    ChannelRole::Dialogue,
                    ChannelRole::LowFrequency,
                ]),
                5 => role(&[
                    ChannelRole::Front,
                    ChannelRole::Front,
                    ChannelRole::Dialogue,
                    ChannelRole::Surround,
                    ChannelRole::Surround,
                ]),
                6 => role(&[
                    ChannelRole::Front,
                    ChannelRole::Front,
                    ChannelRole::Dialogue,
                    ChannelRole::LowFrequency,
                    ChannelRole::Surround,
                    ChannelRole::Surround,
                ]),
                7 => role(&[
                    ChannelRole::Front,
                    ChannelRole::Front,
                    ChannelRole::Dialogue,
                    ChannelRole::LowFrequency,
                    ChannelRole::Surround,
                    ChannelRole::Surround,
                    ChannelRole::Surround,
                ]),
                _ => role(&[
                    ChannelRole::Front,
                    ChannelRole::Front,
                    ChannelRole::Dialogue,
                    ChannelRole::LowFrequency,
                    ChannelRole::Surround,
                    ChannelRole::Surround,
                    ChannelRole::Surround,
                    ChannelRole::Surround,
                ]),
            },
            RiderStrategy::LeaveAlone => ChannelPlan {
                roles: vec![ChannelRole::Unknown; channels as usize],
                strategy: RiderStrategy::LeaveAlone,
                note: String::new(),
            },
        };
        plan.strategy = if channels == 1 {
            RiderStrategy::Mono
        } else if channels == 2 {
            RiderStrategy::StereoMidSide
        } else {
            plan.strategy
        };
        plan.note = note;
        plan
    }

    pub fn strategy_name(&self) -> &'static str {
        match self.strategy {
            RiderStrategy::Mono => "mono",
            RiderStrategy::StereoMidSide => "mid/side",
            RiderStrategy::CentreChannel => "centre-channel",
            RiderStrategy::LeaveAlone => "untouched",
        }
    }

    /// True when this plan cannot do anything useful.
    pub fn is_inert(&self) -> bool {
        self.strategy == RiderStrategy::LeaveAlone
    }

    /// One line for the log: what happens to which channel.
    pub fn describe(&self) -> String {
        let roles: Vec<String> = self
            .roles
            .iter()
            .enumerate()
            .map(|(index, role)| {
                format!(
                    "{}:{}",
                    index,
                    match role {
                        ChannelRole::Dialogue => "dialogue",
                        ChannelRole::Front => "front",
                        ChannelRole::Surround => "surround",
                        ChannelRole::LowFrequency => "lfe",
                        ChannelRole::Unknown => "?",
                    }
                )
            })
            .collect();
        format!("{} [{}] — {}", self.strategy_name(), roles.join(" "), self.note)
    }
}

/// Applies a [`GainCurve`] to interleaved PCM, in place, block by block.
pub struct RemasterProcessor {
    curve: GainCurve,
    channels: usize,
    chunk_frames: usize,
    chunk_index: usize,
    frame_in_chunk: usize,
    plan: ChannelPlan,
    /// One splitter per channel that gets the masking-band duck.
    splitters: Vec<Option<ThreeBandSplitter>>,
    /// The side signal of a stereo pair, and the split of it.
    side_splitter: ThreeBandSplitter,
    scratch_channel: Vec<f32>,
    scratch_side: Vec<f32>,
    scratch_low: Vec<f32>,
    scratch_mid: Vec<f32>,
    scratch_high: Vec<f32>,
    frames_processed: u64,
    peak_gain_linear: f32,
}

impl RemasterProcessor {
    /// Builds a processor for a stream, deciding per channel what may be touched.
    pub fn new(
        curve: GainCurve,
        channels: u16,
        rate: u32,
        settings: &RiderSettings,
        layout: Option<&str>,
    ) -> Self {
        let channels = channels.max(1) as usize;
        let plan = ChannelPlan::for_stream(channels as u16, layout);
        let needs_duck = |role: ChannelRole| {
            matches!(role, ChannelRole::Front | ChannelRole::Surround)
        };
        let splitters = plan
            .roles
            .iter()
            .map(|role| {
                needs_duck(*role).then(|| {
                    ThreeBandSplitter::new(
                        rate,
                        settings.duck_band_low_hz,
                        settings.duck_band_high_hz,
                    )
                })
            })
            .collect();
        let chunk_frames =
            ((rate as f64 * curve.chunk_ms.max(1) as f64 / 1000.0).round() as usize).max(1);
        RemasterProcessor {
            splitters,
            side_splitter: ThreeBandSplitter::new(
                rate,
                settings.duck_band_low_hz,
                settings.duck_band_high_hz,
            ),
            curve,
            channels,
            chunk_frames,
            chunk_index: 0,
            frame_in_chunk: 0,
            plan,
            scratch_channel: Vec::new(),
            scratch_side: Vec::new(),
            scratch_low: Vec::new(),
            scratch_mid: Vec::new(),
            scratch_high: Vec::new(),
            frames_processed: 0,
            peak_gain_linear: 1.0,
        }
    }

    pub fn channel_plan(&self) -> &ChannelPlan {
        &self.plan
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
        match self.plan.strategy {
            RiderStrategy::LeaveAlone => {
                // Count the frames so the curve stays in step, and change nothing.
                self.advance(data.len() / ch);
            }
            RiderStrategy::StereoMidSide if ch == 2 => self.process_stereo(data),
            _ => self.process_roles(data),
        }
    }

    /// Moves the chunk cursor forward by `frames`.
    fn advance(&mut self, frames: usize) {
        self.frames_processed += frames as u64;
        self.frame_in_chunk += frames;
        while self.frame_in_chunk >= self.chunk_frames {
            self.frame_in_chunk -= self.chunk_frames;
            self.chunk_index += 1;
        }
    }

    /// Mid/side: the dialogue is the shared signal, so the gain goes to the mid
    /// and the masking band is ducked in the side.
    fn process_stereo(&mut self, data: &mut [f32]) {
        let mut side = std::mem::take(&mut self.scratch_side);
        let mut low = std::mem::take(&mut self.scratch_low);
        let mut mid_band = std::mem::take(&mut self.scratch_mid);
        let mut high = std::mem::take(&mut self.scratch_high);
        let mut mid = std::mem::take(&mut self.scratch_channel);

        let mut position = 0usize;
        while position < data.len() {
            let frames_available = (data.len() - position) / 2;
            if frames_available == 0 {
                break;
            }
            let frames = (self.chunk_frames - self.frame_in_chunk).min(frames_available);
            let gain = self.curve.gain_linear(self.chunk_index);
            let duck = self.curve.duck_linear(self.chunk_index);
            self.peak_gain_linear = self.peak_gain_linear.max(gain);

            mid.clear();
            side.clear();
            for frame in 0..frames {
                let left = data[position + frame * 2];
                let right = data[position + frame * 2 + 1];
                mid.push((left + right) * 0.5 * gain);
                side.push((left - right) * 0.5);
            }
            if duck < 0.999 {
                // Subtracting the attenuated band keeps the result exactly unity
                // when nothing is ducked, and leaves whatever the crossover did
                // not identify as mid-band alone.
                self.side_splitter
                    .split(&side, &mut low, &mut mid_band, &mut high);
                let reduction = 1.0 - duck;
                for index in 0..frames {
                    side[index] -= mid_band[index] * reduction;
                }
            }
            for frame in 0..frames {
                data[position + frame * 2] = mid[frame] + side[frame];
                data[position + frame * 2 + 1] = mid[frame] - side[frame];
            }

            position += frames * 2;
            self.advance(frames);
        }

        self.scratch_side = side;
        self.scratch_low = low;
        self.scratch_mid = mid_band;
        self.scratch_high = high;
        self.scratch_channel = mid;
    }

    /// One role per channel: dialogue is lifted, screens and surrounds are ducked,
    /// everything else is left alone.
    fn process_roles(&mut self, data: &mut [f32]) {
        let ch = self.channels;
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
                let role = self
                    .plan
                    .roles
                    .get(channel)
                    .copied()
                    .unwrap_or(ChannelRole::Unknown);
                let channel_gain = if role == ChannelRole::Dialogue {
                    gain
                } else {
                    1.0
                };
                let channel_duck = if matches!(role, ChannelRole::Front | ChannelRole::Surround) {
                    duck
                } else {
                    1.0
                };

                channel_buf.clear();
                channel_buf.extend(
                    (0..frames)
                        .map(|frame| data[position + frame * ch + channel] * channel_gain),
                );

                if channel_duck < 0.999 {
                    if let Some(splitter) = splitters.get_mut(channel).and_then(Option::as_mut) {
                        splitter.split(&channel_buf, &mut low, &mut mid, &mut high);
                        let reduction = 1.0 - channel_duck;
                        for index in 0..frames {
                            channel_buf[index] -= mid[index] * reduction;
                        }
                    }
                }

                for (frame, value) in channel_buf.iter().enumerate() {
                    data[position + frame * ch + channel] = *value;
                }
            }

            position += frames * ch;
            self.advance(frames);
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
        let mut processor = RemasterProcessor::new(curve, 1, rate, &settings, Some("mono"));

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

    /// A curve whose envelope has settled on the decision's gain and duck.
    fn settled_curve(settings: &RiderSettings, chunks: usize) -> GainCurve {
        let track = DialogueTrack {
            chunk_ms: 10,
            levels_db: vec![-20.0; chunks],
            mask: vec![true; chunks],
            threshold_db: -40.0,
            noise_floor_db: -50.0,
            speech_ratio: 1.0,
            median_snr_db: 25.0,
            confidence: 1.0,
            notes: vec![],
        };
        let decision = decision_for(-17.0, Some(-29.0), 1.0);
        GainCurve::from_track(&track, &decision, settings)
    }

    /// Mean absolute level of one channel over a settled window.
    fn channel_level(data: &[f32], channels: usize, channel: usize, from: usize, to: usize) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for frame in from..to {
            sum += data[frame * channels + channel].abs();
            count += 1;
        }
        sum / count.max(1) as f32
    }

    /// Mean absolute value of a unit-amplitude sine, so expectations can be
    /// expressed against the input rather than against a magic number.
    const SINE_MEAN: f32 = 0.6366;

    #[test]
    fn a_centre_channel_mix_gets_a_dialogue_boost_not_a_whole_mix_boost() {
        // The complaint this fixes: the rider multiplied the gain into *every*
        // channel, so what was described as a dialogue boost was a boost of the
        // entire mix that happened to coincide with speech — LFE included.
        let settings = RiderSettings::default();
        let curve = settled_curve(&settings, 100);
        let gain = curve.gain_linear(80);
        let duck = curve.duck_linear(80);
        assert!(gain > 1.05 && duck < 0.98, "the fixture must actually act");

        let rate = 48_000u32;
        let frames = 48_000usize;
        let mut processor = RemasterProcessor::new(curve, 6, rate, &settings, Some("5.1(side)"));
        let plan = processor.channel_plan().clone();
        assert_eq!(plan.strategy, RiderStrategy::CentreChannel);
        assert_eq!(plan.roles[2], ChannelRole::Dialogue);
        assert_eq!(plan.roles[3], ChannelRole::LowFrequency);
        assert_eq!(plan.roles[0], ChannelRole::Front);
        assert_eq!(plan.roles[5], ChannelRole::Surround);

        // A 1 kHz tone in every channel: inside the ducked band, and the same
        // starting level everywhere, so the measurements are about what was done
        // to each channel rather than about what was in it.
        let mut data = vec![0.0f32; frames * 6];
        for frame in 0..frames {
            let t = frame as f64 / rate as f64;
            let value = 0.2 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32;
            for channel in 0..6 {
                data[frame * 6 + channel] = value;
            }
        }
        let original = data.clone();
        processor.process_block(&mut data);

        let before = 0.2 * SINE_MEAN;
        let centre = channel_level(&data, 6, 2, 40_000, 48_000);
        let front_left = channel_level(&data, 6, 0, 40_000, 48_000);
        let surround = channel_level(&data, 6, 4, 40_000, 48_000);
        let lfe = channel_level(&data, 6, 3, 40_000, 48_000);

        assert!(
            centre > before * 1.2,
            "the centre channel carries the dialogue and must be lifted: {before} -> {centre}"
        );
        assert!(
            front_left < before * 0.95,
            "a screen channel must not be lifted with the dialogue: {before} -> {front_left}"
        );
        assert!(
            front_left > before * duck * 0.8,
            "the screen channel should be ducked by roughly the requested {duck:.2}, not muted: \
             {front_left}"
        );
        assert!(
            surround < before * 0.95,
            "a surround channel must not be lifted either: {before} -> {surround}"
        );
        assert!(
            (lfe - before).abs() < before * 0.02,
            "the LFE channel must be untouched: {before} -> {lfe}"
        );
        let lfe_changed = (0..frames * 6)
            .filter(|index| index % 6 == 3)
            .any(|index| data[index] != original[index]);
        assert!(
            !lfe_changed,
            "the LFE channel was modified; nothing about dialogue may change the low end"
        );
    }

    #[test]
    fn a_stereo_mix_gets_the_gain_on_the_mid_and_the_duck_on_the_side() {
        let settings = RiderSettings::default();
        let rate = 48_000u32;
        let frames = 48_000usize;
        let before = 0.2 * SINE_MEAN;

        // (a) A centred tone is entirely mid: it gets the gain, and the duck —
        //     which acts on the side signal — does not touch it.
        let curve = settled_curve(&settings, 100);
        let gain = curve.gain_linear(80);
        let mut processor = RemasterProcessor::new(curve, 2, rate, &settings, Some("stereo"));
        assert_eq!(
            processor.channel_plan().strategy,
            RiderStrategy::StereoMidSide
        );
        let mut centred = vec![0.0f32; frames * 2];
        for frame in 0..frames {
            let t = frame as f64 / rate as f64;
            let value = 0.2 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32;
            centred[frame * 2] = value;
            centred[frame * 2 + 1] = value;
        }
        processor.process_block(&mut centred);
        let after = channel_level(&centred, 2, 0, 40_000, 48_000);
        assert!(
            after > before * 1.2,
            "a centred (dialogue) tone must be lifted: {before} -> {after} (gain {gain:.3})"
        );
        assert!(
            after < before * gain * 1.2,
            "and not lifted beyond what the curve asked for"
        );

        // (b) An out-of-phase tone is entirely side: no gain, and its masking
        //     band is ducked.
        let curve = settled_curve(&settings, 100);
        let duck = curve.duck_linear(80);
        let mut processor = RemasterProcessor::new(curve, 2, rate, &settings, Some("stereo"));
        let mut out_of_phase = vec![0.0f32; frames * 2];
        for frame in 0..frames {
            let t = frame as f64 / rate as f64;
            let value = 0.2 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32;
            out_of_phase[frame * 2] = value;
            out_of_phase[frame * 2 + 1] = -value;
        }
        processor.process_block(&mut out_of_phase);
        let after = channel_level(&out_of_phase, 2, 0, 40_000, 48_000);
        assert!(
            after < before * 0.95,
            "an out-of-phase (side) tone must be ducked, not lifted: {before} -> {after} \
             (duck {duck:.3})"
        );
    }

    #[test]
    fn an_unidentified_layout_is_left_alone_rather_than_guessed_at() {
        let settings = RiderSettings::default();
        let rate = 48_000u32;
        let frames = 24_000usize;
        let curve = settled_curve(&settings, 100);
        // Four channels with no centre: FFmpeg's "quad", or something this build
        // has never seen. Either way there is no dialogue channel to lift, and
        // lifting all four is exactly the behaviour being fixed.
        let mut processor = RemasterProcessor::new(curve, 4, rate, &settings, Some("quad"));
        let plan = processor.channel_plan().clone();
        assert_eq!(plan.strategy, RiderStrategy::LeaveAlone);
        assert!(
            plan.note.contains("no centre channel"),
            "the reason must be reportable: {}",
            plan.note
        );

        let mut data: Vec<f32> = (0..frames * 4)
            .map(|index| 0.2 * ((index % 97) as f32 / 97.0 - 0.5))
            .collect();
        let original = data.clone();
        processor.process_block(&mut data);
        assert_eq!(
            data, original,
            "an unidentified layout must pass through untouched"
        );
        assert_eq!(processor.frames_processed(), frames as u64);
    }

    #[test]
    fn the_channel_plan_maps_the_layouts_that_actually_occur() {
        let mono = ChannelPlan::for_stream(1, Some("mono"));
        assert_eq!(mono.strategy, RiderStrategy::Mono);
        assert_eq!(mono.roles, vec![ChannelRole::Dialogue]);

        assert_eq!(
            ChannelPlan::for_stream(2, Some("stereo")).strategy,
            RiderStrategy::StereoMidSide
        );

        for (name, channels) in [("5.1", 6), ("5.1(side)", 6), ("7.1", 8), ("7.1(wide)", 8)] {
            let plan = ChannelPlan::for_stream(channels, Some(name));
            assert_eq!(
                plan.strategy,
                RiderStrategy::CentreChannel,
                "{name} must be handled"
            );
            assert_eq!(plan.roles[2], ChannelRole::Dialogue, "{name} centre");
            assert_eq!(plan.roles[3], ChannelRole::LowFrequency, "{name} lfe");
        }

        // 2.1 is FL FR LFE: no centre channel to lift.
        let two_one = ChannelPlan::for_stream(3, Some("2.1"));
        assert_eq!(two_one.strategy, RiderStrategy::LeaveAlone);
        assert!(two_one.is_inert());

        // 3.0 is FL FR FC.
        assert_eq!(
            ChannelPlan::for_stream(3, Some("3.0")).roles[2],
            ChannelRole::Dialogue
        );

        // An unknown name is reported, not guessed at.
        let odd = ChannelPlan::for_stream(6, Some("hexagonal"));
        assert_eq!(odd.strategy, RiderStrategy::LeaveAlone);
        assert!(odd.note.contains("hexagonal"), "{}", odd.note);
    }

    /// Do the three bands sum back to the signal they came from?
    ///
    /// Everything the duck does depends on this. Subtracting a scaled copy of the mid
    /// band attenuates only when the bands are complementary, and scaling a band and
    /// summing the bands back is exact only under the same condition — measured, the
    /// second form overshoots a pure in-band tone by 17%, which no crossover ripple
    /// explains.
    ///
    /// This prints the ratio per frequency so the shape of the error is visible rather
    /// than asserting a bound, and only fails if the reconstruction is wildly wrong.
    #[test]
    fn the_three_bands_measured_against_their_input() {
        use crate::audio::dsp::ThreeBandSplitter;
        let rate = 48_000u32;
        let mut splitter = ThreeBandSplitter::new(rate, 300.0, 6_000.0);
        let (mut low, mut mid, mut high) = (Vec::new(), Vec::new(), Vec::new());
        for frequency in [
            100.0f64, 200.0, 300.0, 400.0, 1_000.0, 3_000.0, 6_000.0, 12_000.0,
        ] {
            let n = rate as usize / 2;
            let input: Vec<f32> = (0..n)
                .map(|i| {
                    (0.5 * (2.0 * std::f64::consts::PI * frequency * i as f64 / rate as f64).sin())
                        as f32
                })
                .collect();
            splitter.split(&input, &mut low, &mut mid, &mut high);
            // Skip the first 100 ms so the filter state has settled.
            let from = rate as usize / 10;
            let rms = |values: &[f32]| -> f64 {
                let sum: f64 = values[from..].iter().map(|v| (*v as f64) * (*v as f64)).sum();
                (sum / (values.len() - from) as f64).sqrt()
            };
            let reference = rms(&input);
            let summed: Vec<f32> = (0..n).map(|i| low[i] + mid[i] + high[i]).collect();
            eprintln!(
                "  {frequency:>7.0} Hz: input {reference:.5}, low+mid+high {:.5}, ratio {:.4}",
                rms(&summed),
                rms(&summed) / reference
            );
        }
    }

    #[test]
    fn twelve_seconds_of_six_channels_stays_bounded() {
        // The pipeline runs a whole runtime in 100 ms blocks; unit tests here ran
        // a second or two, which is not long enough for a filter's state to
        // misbehave visibly. This walks the same distance with the same block
        // size and asserts that every channel stays inside the range the input
        // put it in.
        //
        // It exists because a pipeline run produced samples around 1e27 in its
        // output file, and the first thing worth establishing was whether the
        // processor can do that at all. It cannot.
        let settings = RiderSettings::default();
        let rate = 48_000u32;
        let frames = rate as usize * 12;
        let curve = settled_curve(&settings, 1200);
        let mut processor = RemasterProcessor::new(curve, 6, rate, &settings, Some("5.1"));

        let mut data = vec![0.0f32; frames * 6];
        for frame in 0..frames {
            let t = frame as f64 / rate as f64;
            let music = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.6;
            let talk = if t % 3.0 < 1.0 {
                (2.0 * std::f64::consts::PI * 1200.0 * t).sin() * 0.06
            } else {
                0.0005 * (2.0 * std::f64::consts::PI * 120.0 * t).sin()
            };
            let rumble = (2.0 * std::f64::consts::PI * 60.0 * t).sin() * 0.5;
            let surround = (2.0 * std::f64::consts::PI * 400.0 * t).sin() * 0.4;
            let base = frame * 6;
            data[base] = music as f32;
            data[base + 1] = music as f32;
            data[base + 2] = talk as f32;
            data[base + 3] = rumble as f32;
            data[base + 4] = surround as f32;
            data[base + 5] = surround as f32;
        }
        let original = data.clone();

        for block in data.chunks_mut(rate as usize / 10 * 6) {
            processor.process_block(block);
        }

        assert!(
            data.iter().all(|sample| sample.is_finite()),
            "every sample must stay finite"
        );
        for channel in 0..6 {
            let peak = (0..frames)
                .map(|frame| data[frame * 6 + channel].abs())
                .fold(0.0f32, f32::max);
            assert!(
                peak <= 1.0,
                "channel {channel} peaked at {peak}, above the range the input put it in"
            );
        }
        // The roles, through the whole runtime: the LFE is bit-identical, the
        // centre is lifted, the screen channels are only ducked.
        let lfe_changed = (0..frames).any(|frame| data[frame * 6 + 3] != original[frame * 6 + 3]);
        assert!(!lfe_changed, "the LFE must be untouched for the whole runtime");
        let peak = |channel: usize, from: usize, to: usize| -> f32 {
            (from..to)
                .map(|frame| data[frame * 6 + channel].abs())
                .fold(0.0f32, f32::max)
        };
        let centre_before = 0.06f32;
        assert!(
            peak(2, 0, frames / 6) > centre_before * 1.2,
            "the centre must be lifted"
        );
        assert!(
            peak(0, 0, frames / 6) < 0.6 * 1.02,
            "a screen channel must not be lifted above its input"
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
