//! The production profile.
//!
//! These are the values the engine runs with when nobody overrides anything:
//! the 16 GB "safe" configuration. The important ones are not preferences, they
//! are constraints learned from how these models actually fail on consumer cards:
//!
//! * **3B FP8, not FP16.** FP16 3B nominally fits in 16 GB and then OOMs the
//!   moment the desktop, the CUDA context and the VAE activations claim theirs.
//! * **VAE in BF16, not FP16.** FP16 VAE produces dark patches: the 3D causal
//!   convolutions overflow in the intermediate accumulators.
//! * **temporal batch of 4n+1, and only 5 or 1.** Batch 3 does not fit the
//!   model's temporal structure and produces boundary artefacts.
//! * **one model resident at a time.** SeedVR2 and an interpolator together do
//!   not fit; the stages are sequenced and each worker process exits.

use crate::audio::dialogue::SpeechDetectorOptions;
use crate::audio::rider::RiderSettings;
use crate::media::classify::ClassifyOptions;
use crate::media::scene::SceneOptions;
use serde::{Deserialize, Serialize};

/// Temporal batches that are valid for the restoration model: 4n+1.
pub const VALID_TEMPORAL_BATCHES: [u32; 6] = [1, 5, 9, 13, 17, 21];

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationMethod {
    /// Nothing: keep the source cadence.
    Off,
    /// Frame duplication. Deterministic, free, invents no motion.
    Duplicate,
    /// FFmpeg's motion-compensated interpolator. Slow, no model.
    Minterpolate,
    /// A plugin model (RIFE-class) when one is installed.
    Plugin,
}

impl InterpolationMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            InterpolationMethod::Off => "off",
            InterpolationMethod::Duplicate => "duplicate",
            InterpolationMethod::Minterpolate => "minterpolate",
            InterpolationMethod::Plugin => "plugin",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "off" | "none" => Some(InterpolationMethod::Off),
            "duplicate" | "dup" => Some(InterpolationMethod::Duplicate),
            "minterpolate" | "mci" => Some(InterpolationMethod::Minterpolate),
            "plugin" | "rife" => Some(InterpolationMethod::Plugin),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RestorationSettings {
    pub enabled: bool,
    pub model: String,
    pub weights: String,
    /// BF16 is required: FP16 VAE causes dark patches on this architecture.
    pub vae_dtype: String,
    pub preferred_batch: u32,
    pub valid_batches: Vec<u32>,
    pub temporal_overlap: u32,
    pub uniform_batch: bool,
    pub block_swap: u32,
    pub block_swap_max: u32,
    pub vae_tile: u32,
    pub vae_tile_min: u32,
    pub offload: String,
    pub attention: String,
    pub color_fix: String,
    /// Refuse to upscale beyond this factor, however large the target is.
    pub max_upscale: f32,
}

impl Default for RestorationSettings {
    fn default() -> Self {
        RestorationSettings {
            // The safe-16gb profile *asks* for the model, because that is what
            // the product is for. When no plugin is installed the planner
            // downgrades loudly — a warning in the plan, a note in the log and
            // `ffmpeg-single-pass` in the executor row — rather than quietly
            // pretending restoration happened.
            enabled: true,
            model: "SeedVR2-3B".into(),
            weights: "fp8_e4m3fn".into(),
            vae_dtype: "bfloat16".into(),
            preferred_batch: 5,
            valid_batches: VALID_TEMPORAL_BATCHES.to_vec(),
            temporal_overlap: 2,
            uniform_batch: true,
            block_swap: 16,
            block_swap_max: 24,
            vae_tile: 512,
            vae_tile_min: 256,
            offload: "cpu".into(),
            attention: "sdpa".into(),
            color_fix: "wavelet".into(),
            max_upscale: 4.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterpolationSettings {
    /// "Film mode": double the frame rate, never 24 -> 60.
    pub multiplier: u32,
    pub method: InterpolationMethod,
    /// Never synthesise a frame across a cut. Not optional.
    pub scene_cut_protection: bool,
    /// Do not spend GPU time on frames that are identical.
    pub skip_static_frames: bool,
}

impl Default for InterpolationSettings {
    fn default() -> Self {
        InterpolationSettings {
            multiplier: 2,
            // "Film mode" asks for the model; `resolve_interpolation` downgrades
            // to frame duplication, with a warning, when no plugin is installed.
            method: InterpolationMethod::Plugin,
            scene_cut_protection: true,
            skip_static_frames: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoudnessTarget {
    /// Anchor for the final normalisation; EBU R128 S4's worked example lands
    /// near -19 LUFS when dialogue sits at -24.
    pub i_lufs: f32,
    pub true_peak_dbtp: f32,
    pub lra_lu: f32,
}

impl Default for LoudnessTarget {
    fn default() -> Self {
        LoudnessTarget {
            i_lufs: -19.0,
            true_peak_dbtp: -1.5,
            lra_lu: 20.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioSettings {
    pub enabled: bool,
    pub stream_index: usize,
    pub sample_rate: u32,
    /// Keep every original track alongside the enhanced one.
    pub keep_original_tracks: bool,
    pub lossless_enhanced: bool,
    pub loudness: LoudnessTarget,
    pub rider: RiderSettings,
    pub dialogue: SpeechDetectorOptions,
}

impl Default for AudioSettings {
    fn default() -> Self {
        AudioSettings {
            enabled: true,
            stream_index: 0,
            sample_rate: 48_000,
            keep_original_tracks: true,
            lossless_enhanced: true,
            loudness: LoudnessTarget::default(),
            rider: RiderSettings::default(),
            dialogue: SpeechDetectorOptions::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputSettings {
    pub container: String,
    pub quality: i32,
    pub prefer_hardware: bool,
    pub prefer_10bit: bool,
    pub preserve_subtitles: bool,
    pub preserve_chapters: bool,
    pub preserve_attachments: bool,
    /// Re-grain strength; 0 disables it.
    ///
    /// Grain is expensive to encode (it fights the encoder) and the per-shot
    /// estimator is not part of this build, so it is off by default rather than
    /// applied with a guessed value.
    pub regrain_strength: f32,
}

impl Default for OutputSettings {
    fn default() -> Self {
        OutputSettings {
            container: "matroska".into(),
            quality: 24,
            prefer_hardware: true,
            prefer_10bit: true,
            preserve_subtitles: true,
            preserve_chapters: true,
            preserve_attachments: true,
            regrain_strength: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuPolicy {
    /// Hard ceiling for AI work, in MiB. 13 GiB on a 16 GB card.
    pub max_ai_vram_mib: u64,
    /// Always leave this much for the OS, the desktop and the driver.
    pub reserve_mib: u64,
    /// Only one model may be resident at a time.
    pub concurrent_models: u32,
}

impl Default for GpuPolicy {
    fn default() -> Self {
        GpuPolicy {
            max_ai_vram_mib: 13_312, // 13 GiB
            reserve_mib: 2_560,      // 2.5 GiB
            concurrent_models: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisSettings {
    pub probe: bool,
    pub temporal: bool,
    pub scenes: bool,
    pub audio: bool,
    /// Sample windows for `idet`.
    pub classify: ClassifyOptions,
    pub scene: SceneOptions,
}

impl Default for AnalysisSettings {
    fn default() -> Self {
        AnalysisSettings {
            probe: true,
            temporal: true,
            scenes: true,
            audio: true,
            classify: ClassifyOptions::default(),
            scene: SceneOptions::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RestorationProfile {
    pub name: String,
    pub description: String,
    pub restoration: RestorationSettings,
    pub interpolation: InterpolationSettings,
    pub audio: AudioSettings,
    pub output: OutputSettings,
    pub gpu: GpuPolicy,
    pub analysis: AnalysisSettings,
}

impl Default for RestorationProfile {
    fn default() -> Self {
        RestorationProfile::safe_16gb()
    }
}

impl RestorationProfile {
    /// The profile this project is built around: a 16 GB consumer card,
    /// unattended, no per-title tuning.
    pub fn safe_16gb() -> Self {
        RestorationProfile {
            name: "safe-16gb".into(),
            description:
                "16 GB consumer GPU, unattended: 3B FP8 restoration (when a model plugin is \
                 installed), BF16 VAE, 4n+1 temporal batches, 13 GiB AI ceiling, 2x film-mode \
                 interpolation, EBU R128 S4 dialogue handling."
                    .into(),
            restoration: RestorationSettings::default(),
            interpolation: InterpolationSettings::default(),
            audio: AudioSettings::default(),
            output: OutputSettings::default(),
            gpu: GpuPolicy::default(),
            analysis: AnalysisSettings::default(),
        }
    }

    /// For a machine with no GPU model at all: analysis plus deterministic
    /// resample, still a useful, correct output.
    pub fn deterministic() -> Self {
        let mut profile = RestorationProfile::safe_16gb();
        profile.name = "deterministic".into();
        profile.description =
            "No model backend: analysis, cadence correction, resampling, audio remaster and a \
             hardware-accelerated encode. Invents nothing."
                .into();
        profile.restoration.enabled = false;
        profile.interpolation.method = InterpolationMethod::Duplicate;
        profile
    }

    /// Fast preview: smaller output, faster encode, same correctness rules.
    pub fn preview() -> Self {
        let mut profile = RestorationProfile::safe_16gb();
        profile.name = "preview".into();
        profile.description = "Half-resolution preview encode for checking a pipeline run.".into();
        profile.output.quality = 30;
        profile.restoration.enabled = false;
        profile.interpolation.multiplier = 1;
        profile.interpolation.method = InterpolationMethod::Off;
        profile
    }

    pub fn builtin() -> Vec<RestorationProfile> {
        vec![
            RestorationProfile::safe_16gb(),
            RestorationProfile::deterministic(),
            RestorationProfile::preview(),
        ]
    }

    pub fn by_name(name: &str) -> Option<RestorationProfile> {
        RestorationProfile::builtin()
            .into_iter()
            .find(|p| p.name == name)
    }

    /// Temporal batches this profile may use, restricted to 4n+1.
    pub fn batch_ladder(&self) -> Vec<u32> {
        let mut ladder: Vec<u32> = self
            .restoration
            .valid_batches
            .iter()
            .copied()
            .filter(|b| VALID_TEMPORAL_BATCHES.contains(b))
            .collect();
        ladder.sort_unstable();
        // Prefer the configured batch, then step down.
        ladder.retain(|b| *b <= self.restoration.preferred_batch);
        ladder.sort_by(|a, b| b.cmp(a));
        if ladder.is_empty() {
            ladder.push(1);
        }
        ladder
    }

    pub fn summary_rows(&self) -> Vec<(String, String)> {
        vec![
            ("profile".into(), self.name.clone()),
            (
                "restoration".into(),
                if self.restoration.enabled {
                    format!(
                        "{} {} (VAE {})",
                        self.restoration.model,
                        self.restoration.weights,
                        self.restoration.vae_dtype
                    )
                } else {
                    "disabled (no model backend selected)".into()
                },
            ),
            (
                "temporal batch".into(),
                format!(
                    "{} (valid: {:?})",
                    self.restoration.preferred_batch, self.restoration.valid_batches
                ),
            ),
            (
                "block swap".into(),
                format!(
                    "{} (max {})",
                    self.restoration.block_swap, self.restoration.block_swap_max
                ),
            ),
            (
                "interpolation".into(),
                format!(
                    "{}x via {}",
                    self.interpolation.multiplier,
                    self.interpolation.method.as_str()
                ),
            ),
            (
                "audio target".into(),
                format!(
                    "dialogue {:.1} LUFS, LDR <= {:.1} LU, TP {:.1} dBTP",
                    self.audio.rider.target_dialogue_lufs,
                    self.audio.rider.target_ldr_lu,
                    self.audio.loudness.true_peak_dbtp
                ),
            ),
            (
                "gpu ceiling".into(),
                format!(
                    "min({} MiB, free - {} MiB), {} model at a time",
                    self.gpu.max_ai_vram_mib, self.gpu.reserve_mib, self.gpu.concurrent_models
                ),
            ),
            (
                "container".into(),
                format!(
                    "{} · {}",
                    self.output.container,
                    if self.output.prefer_10bit {
                        "10-bit preferred"
                    } else {
                        "8-bit"
                    }
                ),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_profile_is_the_safe_16gb_one() {
        let profile = RestorationProfile::default();
        assert_eq!(profile.name, "safe-16gb");
        assert_eq!(profile.restoration.weights, "fp8_e4m3fn");
        assert_eq!(
            profile.restoration.vae_dtype, "bfloat16",
            "FP16 VAE causes dark patches"
        );
        assert_eq!(profile.restoration.preferred_batch, 5);
        assert_eq!(profile.gpu.max_ai_vram_mib, 13_312);
        assert_eq!(profile.interpolation.multiplier, 2);
        assert!(profile.interpolation.scene_cut_protection);
    }

    #[test]
    fn batch_ladder_never_contains_an_invalid_temporal_batch() {
        let profile = RestorationProfile::safe_16gb();
        let ladder = profile.batch_ladder();
        assert_eq!(ladder, vec![5, 1], "3 is not a valid 4n+1 batch");
        for batch in &ladder {
            assert_eq!((batch - 1) % 4, 0, "{batch} is not 4n+1");
        }
    }

    #[test]
    fn deterministic_profile_claims_no_restoration() {
        let profile = RestorationProfile::deterministic();
        assert!(!profile.restoration.enabled);
        assert_eq!(profile.interpolation.method, InterpolationMethod::Duplicate);
    }

    #[test]
    fn profiles_are_looked_up_by_name() {
        assert!(RestorationProfile::by_name("preview").is_some());
        assert!(RestorationProfile::by_name("nope").is_none());
        for profile in RestorationProfile::builtin() {
            assert!(!profile.summary_rows().is_empty());
        }
    }

    #[test]
    fn interpolation_methods_round_trip_through_text() {
        for method in [
            InterpolationMethod::Off,
            InterpolationMethod::Duplicate,
            InterpolationMethod::Minterpolate,
            InterpolationMethod::Plugin,
        ] {
            assert_eq!(InterpolationMethod::parse(method.as_str()), Some(method));
        }
        assert_eq!(InterpolationMethod::parse("RIFE"), Some(InterpolationMethod::Plugin));
        assert_eq!(InterpolationMethod::parse("nonsense"), None);
    }
}
