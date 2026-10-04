//! The production profile.
//!
//! These are the values the engine runs with when nobody overrides anything.
//! They describe exactly two model tasks, because the product has exactly two:
//!
//! | task | model | runtime |
//! |---|---|---|
//! | restoration | `RealESRGAN_x4plus` | ncnn, Vulkan, in-process |
//! | interpolation | `RIFE 4.25` (`ensemble = false`) | ncnn, Vulkan, in-process |
//!
//! There is deliberately **no** temporal batch, **no** block swap, **no** VAE
//! dtype and **no** CPU-offload setting. Those belonged to a diffusion video
//! model this project does not run; keeping their configuration would be keeping
//! a description of a pipeline that no longer exists, and the next person to read
//! this file would have no way to tell which numbers were load-bearing.
//!
//! The numbers that *are* load-bearing:
//!
//! * **The tile ladder.** `0` means the runtime picks; after an out-of-memory
//!   failure the ladder steps down through [`TILE_LADDER`]. A smaller tile costs
//!   throughput and can show seams, so it is spent last.
//! * **The upscale ceiling.** A model scale is a *resampling* factor, not a
//!   decision about the output raster; see [`crate::pipeline::plan`].

use crate::audio::dialogue::SpeechDetectorOptions;
use crate::audio::rider::RiderSettings;
use crate::media::classify::ClassifyOptions;
use crate::media::scene::SceneOptions;
use serde::{Deserialize, Serialize};

/// Tile edges the restoration runtime is allowed to fall back to, largest first.
///
/// Below 128 the tile is smaller than the model's receptive field and seams stop
/// being subtle, so the ladder ends there rather than pretending a smaller tile
/// is still a quality-preserving choice.
pub const TILE_LADDER: [u32; 5] = [512, 384, 256, 192, 128];

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationMethod {
    /// Keep the source cadence.
    Off,
    /// RIFE 4.25 through the native ncnn runtime.
    ///
    /// There is no second value here on purpose. In particular there is no
    /// `minterpolate` and no "duplicate": a request for interpolation that the
    /// runtime cannot serve is refused, never answered with a different
    /// algorithm that happens to increase the frame count.
    Rife,
}

impl InterpolationMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            InterpolationMethod::Off => "off",
            InterpolationMethod::Rife => "rife",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "off" | "none" => Some(InterpolationMethod::Off),
            "rife" | "rife4.25" | "4.25" => Some(InterpolationMethod::Rife),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RestorationSettings {
    pub enabled: bool,
    /// The model directory name under the models root.
    pub model: String,
    /// The model's own scale factor: `RealESRGAN_x4plus` is 4.
    ///
    /// This is what the network multiplies by. It is **not** the output size;
    /// the target raster is computed from the source geometry and the encoder,
    /// and a deterministic resize lands on it afterwards.
    pub scale: u32,
    /// Tile edge in pixels. `0` lets the runtime choose.
    pub tile: u32,
    pub tile_min: u32,
    /// Refuse to upscale beyond this factor, however large the target is.
    pub max_upscale: f32,
}

impl Default for RestorationSettings {
    fn default() -> Self {
        RestorationSettings {
            // The profile *asks* for the model, because that is what the product
            // is for. When the runtime or the weights are missing the planner
            // fails with a message naming what is absent; it does not quietly
            // substitute a resize and call it restoration.
            enabled: true,
            model: "RealESRGAN_x4plus".into(),
            scale: 4,
            tile: 0,
            tile_min: 128,
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
            method: InterpolationMethod::Rife,
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
    /// When 0 the per-shot estimator is not consulted at all. When positive the
    /// estimator measures each shot and the *measured* strength is used, so this
    /// is a switch rather than a value.
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
                "16 GB consumer GPU, unattended: Real-ESRGAN x4plus restoration and RIFE 4.25 \
                 interpolation through the native ncnn runtime, 13 GiB AI ceiling, 2x film-mode \
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

    /// Analysis, cadence correction, resampling, audio remaster and a
    /// hardware-accelerated encode — with no model in the path.
    ///
    /// This is a real product configuration, not a degraded one: plenty of
    /// sources need the temporal and geometric corrections and nothing else, and
    /// running a restoration network over them would invent detail that was
    /// never there.
    pub fn deterministic() -> Self {
        let mut profile = RestorationProfile::safe_16gb();
        profile.name = "deterministic".into();
        profile.description =
            "No model in the path: analysis, cadence correction, resampling, audio remaster and a \
             hardware-accelerated encode. Invents nothing."
                .into();
        profile.restoration.enabled = false;
        profile.interpolation.method = InterpolationMethod::Off;
        profile.interpolation.multiplier = 1;
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

    /// Tile edges this profile may use, largest first.
    ///
    /// `0` (auto) is not in the list: it is the starting point, and the ladder is
    /// what to fall back *to* once auto has failed.
    pub fn tile_ladder(&self) -> Vec<u32> {
        TILE_LADDER
            .iter()
            .copied()
            .filter(|tile| *tile >= self.restoration.tile_min)
            .collect()
    }

    pub fn summary_rows(&self) -> Vec<(String, String)> {
        vec![
            ("profile".into(), self.name.clone()),
            (
                "restoration".into(),
                if self.restoration.enabled {
                    format!(
                        "{} x{} (tile {})",
                        self.restoration.model,
                        self.restoration.scale,
                        if self.restoration.tile == 0 {
                            "auto".to_string()
                        } else {
                            self.restoration.tile.to_string()
                        }
                    )
                } else {
                    "off (no model in the path)".into()
                },
            ),
            (
                "tile ladder".into(),
                format!("auto -> {:?}", self.tile_ladder()),
            ),
            (
                "interpolation".into(),
                match self.interpolation.method {
                    InterpolationMethod::Off => "off".to_string(),
                    InterpolationMethod::Rife => format!(
                        "{}x via RIFE 4.25 (ensemble off)",
                        self.interpolation.multiplier
                    ),
                },
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
        assert_eq!(profile.restoration.model, "RealESRGAN_x4plus");
        assert_eq!(profile.restoration.scale, 4);
        assert_eq!(profile.gpu.max_ai_vram_mib, 13_312);
        assert_eq!(profile.interpolation.multiplier, 2);
        assert!(profile.interpolation.scene_cut_protection);
    }

    #[test]
    fn there_is_no_second_interpolation_algorithm_to_fall_back_to() {
        // The whole point: a caller cannot ask for "something else that adds
        // frames". If RIFE is not available the request fails; it does not become
        // `minterpolate`.
        assert_eq!(
            InterpolationMethod::parse("minterpolate"),
            None,
            "minterpolate must not be reachable through the profile"
        );
        assert_eq!(InterpolationMethod::parse("mci"), None);
        assert_eq!(InterpolationMethod::parse("duplicate"), None);
        assert_eq!(
            InterpolationMethod::parse("rife"),
            Some(InterpolationMethod::Rife)
        );
    }

    #[test]
    fn the_tile_ladder_drops_only_towards_smaller_tiles() {
        let profile = RestorationProfile::safe_16gb();
        let ladder = profile.tile_ladder();
        assert_eq!(ladder, vec![512, 384, 256, 192, 128]);
        assert!(
            ladder.windows(2).all(|w| w[0] > w[1]),
            "the ladder must be strictly descending: {ladder:?}"
        );
        assert_eq!(*ladder.last().unwrap(), profile.restoration.tile_min);
    }

    #[test]
    fn deterministic_profile_claims_no_model_task_at_all() {
        let profile = RestorationProfile::deterministic();
        assert!(!profile.restoration.enabled);
        assert_eq!(profile.interpolation.method, InterpolationMethod::Off);
        assert_eq!(profile.interpolation.multiplier, 1);
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
        for method in [InterpolationMethod::Off, InterpolationMethod::Rife] {
            assert_eq!(InterpolationMethod::parse(method.as_str()), Some(method));
        }
        assert_eq!(InterpolationMethod::parse("RIFE"), Some(InterpolationMethod::Rife));
        assert_eq!(InterpolationMethod::parse("nonsense"), None);
    }
}
