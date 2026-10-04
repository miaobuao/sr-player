//! VRAM budgeting and the out-of-memory degrade ladder.
//!
//! Two rules make unattended operation possible:
//!
//! 1. **Budget from what is free, not from what is installed.**
//!    `min(13 GiB, free - 2.5 GiB)` — because the user's browser, the desktop
//!    compositor and the driver are all sharing the same 16 GB.
//! 2. **On OOM, degrade in a fixed order and retry the same chunk.** For the two
//!    networks this project runs there is exactly one knob that reduces peak
//!    memory: the restoration tile edge. RIFE already works a pair of frames at a
//!    time and needs no budget of its own, so the ladder is the tile ladder and
//!    nothing else — a short ladder that is honest about what it can and cannot
//!    buy, rather than a long one whose lower rungs do nothing.

use crate::gpu::{self, GpuInfo};
use crate::pipeline::profile::{RestorationSettings, TILE_LADDER};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VramBudget {
    pub total_mib: Option<u64>,
    pub free_mib: Option<u64>,
    /// The ceiling AI work may use.
    pub ai_budget_mib: u64,
    pub reserve_mib: u64,
    /// Which probe produced the numbers.
    pub source: String,
    pub notes: Vec<String>,
}

impl VramBudget {
    pub fn describe(&self) -> String {
        match (self.total_mib, self.free_mib) {
            (Some(total), Some(free)) => format!(
                "{:.1} GiB card, {:.1} GiB free, {:.1} GiB budget for models ({})",
                total as f64 / 1024.0,
                free as f64 / 1024.0,
                self.ai_budget_mib as f64 / 1024.0,
                self.source
            ),
            _ => format!(
                "{:.1} GiB static budget for models (no GPU telemetry: {})",
                self.ai_budget_mib as f64 / 1024.0,
                self.source
            ),
        }
    }

    /// Does a proposed working set fit?
    pub fn fits(&self, required_mib: u64) -> bool {
        required_mib <= self.ai_budget_mib
    }
}

/// `min(hard ceiling, free - reserve)`.
///
/// The native runtime also knows a free-memory figure of its own, and in time
/// that figure should take precedence because the runtime is what allocates. It
/// is not consulted yet: the number would come from the same Vulkan heap the
/// device probe already reads, so preferring it today would add a second source
/// of truth without adding information.
pub fn plan_vram(hard_ceiling_mib: u64, reserve_mib: u64, gpus: &[GpuInfo]) -> VramBudget {
    let free = gpu::free_mib(gpus);
    let mut notes = Vec::new();
    let mut budget = hard_ceiling_mib;
    if let Some(free) = free {
        let available = free.saturating_sub(reserve_mib);
        if available < budget {
            notes.push(format!(
                "only {:.1} GiB free after reserving {:.1} GiB: budget reduced from {:.1} GiB",
                available as f64 / 1024.0,
                reserve_mib as f64 / 1024.0,
                hard_ceiling_mib as f64 / 1024.0
            ));
            budget = available;
        }
        if available < 2_048 {
            notes.push(
                "less than 2 GiB is available for models; expect the pipeline to walk the \
                 degrade ladder immediately"
                    .into(),
            );
        }
    } else {
        notes.push(format!(
            "no GPU telemetry ({}): assuming the full {:.1} GiB ceiling",
            gpu::describe(gpus),
            hard_ceiling_mib as f64 / 1024.0
        ));
    }
    VramBudget {
        total_mib: gpus.iter().filter_map(|g| g.total_mib).max(),
        free_mib: free,
        ai_budget_mib: budget,
        reserve_mib,
        source: if gpus.is_empty() {
            "static".into()
        } else {
            gpus[0].source.clone()
        },
        notes,
    }
}

/// Current working point of the restore stage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkingSet {
    /// Tile edge in pixels. `0` means the runtime chooses.
    pub tile: u32,
    pub tile_min: u32,
    /// Model scale, carried so the memory estimate knows the activation shape.
    pub scale: u32,
    /// How many times we have degraded for this job.
    pub rung: u32,
}

impl WorkingSet {
    pub fn initial(settings: &RestorationSettings) -> Self {
        WorkingSet {
            tile: settings.tile,
            tile_min: settings.tile_min,
            scale: settings.scale.max(1),
            rung: 0,
        }
    }

    pub fn describe(&self) -> String {
        if self.tile == 0 {
            "tile auto".to_string()
        } else {
            format!("tile {}", self.tile)
        }
    }

    /// The tile edge to hand the runtime, or `None` for "the runtime decides".
    ///
    /// This is the number that crosses the C ABI; `0` there means automatic.
    pub fn tile_arg(&self) -> u32 {
        self.tile
    }

    /// Estimated resident working set, purely for the "does it fit" conversation.
    ///
    /// The model is 67 MB of weights; everything else is activation. A tiled
    /// x4 network holds roughly `tile^2 * scale^2 * 3` outputs, and each of those
    /// costs several bytes across the intermediate feature maps — the factor of
    /// 48 below is the sum of that, and it is an estimate to start a
    /// conversation, not a measurement. It is only ever used to print a number
    /// and to decide when to warn.
    pub fn estimated_mib(&self) -> u64 {
        // With no tile chosen yet, assume the largest the ladder would pick.
        let tile = if self.tile == 0 {
            TILE_LADDER[0]
        } else {
            self.tile
        } as u64;
        let weights = 67;
        let scale = self.scale as u64;
        let activations = tile * tile * scale * scale * 3 * 48 / 1_048_576;
        // Input, output and the encoder's hand-off copy of a 4K frame.
        let frame_io = 3 * (3840u64 * 2160 * 3 / 1_048_576);
        weights + activations + frame_io
    }

    /// Next degradation step. `None` when the ladder is exhausted.
    ///
    /// The only rung is a smaller tile, and the ladder is strictly descending so
    /// repeated failures cannot loop on the same value.
    pub fn degrade(&mut self) -> Option<String> {
        self.rung += 1;
        if self.tile == 0 {
            // Automatic failed: start at the top of the ladder. The top rung is
            // not a reduction, it is the first *explicit* choice, and saying so
            // keeps the log readable when `auto` was the thing that did not fit.
            let next = TILE_LADDER
                .iter()
                .copied()
                .find(|tile| *tile >= self.tile_min)
                .unwrap_or(self.tile_min);
            let message = format!("tile auto -> {next} (explicit tile after an out-of-memory failure)");
            self.tile = next;
            return Some(message);
        }
        let next = TILE_LADDER
            .iter()
            .copied()
            .filter(|tile| *tile < self.tile && *tile >= self.tile_min)
            .max();
        match next {
            Some(next) => {
                let message = format!(
                    "reducing tile {} -> {} (smaller activation; watch for tile seams)",
                    self.tile, next
                );
                self.tile = next;
                Some(message)
            }
            None => {
                self.rung -= 1;
                None
            }
        }
    }

    pub fn is_exhausted(&self) -> bool {
        let mut probe = self.clone();
        probe.degrade().is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuDeviceType, GpuVendor};
    use crate::pipeline::profile::RestorationProfile;

    fn gpu(total: u64, used: u64) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "RTX 5070 Ti".into(),
            total_mib: Some(total),
            used_mib: Some(used),
            device_type: GpuDeviceType::Discrete,
            source: "nvidia-smi".into(),
            ..GpuInfo::default()
        }
    }

    #[test]
    fn budget_is_the_hard_ceiling_on_an_idle_16gb_card() {
        let profile = RestorationProfile::safe_16gb();
        let budget = plan_vram(profile.gpu.max_ai_vram_mib, profile.gpu.reserve_mib, &[gpu(16_384, 500)]);
        assert_eq!(budget.ai_budget_mib, 13_312, "13 GiB ceiling applies");
        assert_eq!(budget.free_mib, Some(15_884));
    }

    #[test]
    fn budget_shrinks_when_the_desktop_is_using_the_card() {
        let profile = RestorationProfile::safe_16gb();
        let budget = plan_vram(profile.gpu.max_ai_vram_mib, profile.gpu.reserve_mib, &[gpu(16_384, 6_000)]);
        // 16384 total - 6000 used = 10384 free, minus the 2560 reserve.
        assert_eq!(budget.free_mib, Some(10_384));
        assert_eq!(budget.ai_budget_mib, 7_824);
        assert!(budget.ai_budget_mib < profile.gpu.max_ai_vram_mib);
        assert!(budget.notes.iter().any(|n| n.contains("budget reduced")));
    }

    #[test]
    fn no_telemetry_falls_back_to_the_static_ceiling_and_says_so() {
        let budget = plan_vram(13_312, 2_560, &[]);
        assert_eq!(budget.ai_budget_mib, 13_312);
        assert_eq!(budget.source, "static");
        assert!(budget.describe().contains("static budget"));
        assert!(budget.notes.iter().any(|n| n.contains("no GPU telemetry")));
    }

    #[test]
    fn a_full_card_warns_instead_of_silently_failing() {
        let budget = plan_vram(13_312, 2_560, &[gpu(16_384, 15_500)]);
        assert!(budget.ai_budget_mib < 2_048);
        assert!(budget.notes.iter().any(|n| n.contains("degrade ladder")));
    }

    #[test]
    fn the_ladder_starts_at_auto_and_only_ever_shrinks() {
        let profile = RestorationProfile::safe_16gb();
        let mut set = WorkingSet::initial(&profile.restoration);
        assert_eq!(set.tile, 0, "the profile asks the runtime to choose");

        let first = set.degrade().expect("auto is the first rung");
        assert!(first.contains("auto -> 512"), "got: {first}");
        assert_eq!(set.tile, 512);

        let mut seen = vec![set.tile];
        while let Some(change) = set.degrade() {
            assert!(change.contains("reducing tile"), "got: {change}");
            seen.push(set.tile);
        }
        assert_eq!(seen, vec![512, 384, 256, 192, 128]);
        assert!(
            seen.windows(2).all(|w| w[0] > w[1]),
            "the ladder must be strictly descending: {seen:?}"
        );
        assert!(set.is_exhausted());
    }

    #[test]
    fn an_explicit_tile_below_the_cap_still_walks_down_to_the_floor() {
        let mut settings = RestorationProfile::safe_16gb().restoration;
        settings.tile = 384;
        settings.tile_min = 192;
        let mut set = WorkingSet::initial(&settings);
        let mut seen = vec![set.tile];
        while set.degrade().is_some() {
            seen.push(set.tile);
        }
        assert_eq!(seen, vec![384, 256, 192], "the floor is tile_min, not 128");
        assert!(set.is_exhausted());
    }

    #[test]
    fn the_ladder_terminates_and_reports_exhaustion() {
        let profile = RestorationProfile::safe_16gb();
        let mut set = WorkingSet::initial(&profile.restoration);
        let mut steps = 0;
        while set.degrade().is_some() {
            steps += 1;
            assert!(steps < 20, "the ladder must terminate");
        }
        assert!(set.is_exhausted());
        assert_eq!(set.tile, set.tile_min);
    }

    #[test]
    fn working_set_estimate_never_rises_as_the_ladder_descends() {
        let profile = RestorationProfile::safe_16gb();
        let mut set = WorkingSet::initial(&profile.restoration);
        let mut estimates = vec![set.estimated_mib()];
        while set.degrade().is_some() {
            estimates.push(set.estimated_mib());
        }
        assert_eq!(estimates.len(), 6, "auto plus five rungs: {estimates:?}");
        assert!(
            estimates.windows(2).all(|w| w[1] <= w[0]),
            "the estimate must never rise while degrading: {estimates:?}"
        );
        // The first rung only makes `auto` explicit, and the estimate already
        // assumed `auto` would pick the same 512 tile, so that one step is flat by
        // construction rather than by accident. Every rung below it must really
        // reduce the working set, or the ladder is not buying anything.
        assert_eq!(estimates[0], estimates[1], "auto -> 512 is not a reduction");
        assert!(
            estimates[2..].windows(2).all(|w| w[1] < w[0]),
            "every rung below the first must strictly reduce the estimate: {estimates:?}"
        );
        assert!(estimates.last().unwrap() < estimates.first().unwrap());
        assert!(set.describe().contains("tile 128"));
        assert_eq!(set.tile_arg(), 128);
    }

    #[test]
    fn an_untiled_working_set_is_reported_as_automatic() {
        let profile = RestorationProfile::safe_16gb();
        let set = WorkingSet::initial(&profile.restoration);
        assert_eq!(set.tile_arg(), 0, "0 is the runtime's own default");
        assert_eq!(set.describe(), "tile auto");
    }
}
