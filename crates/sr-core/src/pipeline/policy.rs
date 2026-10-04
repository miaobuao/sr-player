//! VRAM budgeting and the OOM degrade ladder.
//!
//! Two rules make unattended operation possible:
//!
//! 1. **Budget from what is free, not from what is installed.**
//!    `min(13 GiB, free - 2.5 GiB)` — because the user's browser, the desktop
//!    compositor and the CUDA context are all sharing the same 16 GB.
//! 2. **On OOM, degrade in a fixed order and retry the same shot.** Raising
//!    block swap costs throughput; lowering the temporal batch costs temporal
//!    consistency; shrinking the VAE tile costs quality. So the ladder spends
//!    throughput first and quality last, and it never produces a batch that the
//!    model cannot accept.

use crate::gpu::{self, GpuInfo};
use crate::pipeline::profile::{RestorationSettings, VALID_TEMPORAL_BATCHES};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffloadMode {
    /// Everything resident.
    None,
    /// Weights streamed from pinned host memory.
    Cpu,
}

impl OffloadMode {
    pub fn as_str(self) -> &'static str {
        match self {
            OffloadMode::None => "none",
            OffloadMode::Cpu => "cpu",
        }
    }
}

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
    pub block_swap: u32,
    pub block_swap_max: u32,
    pub vae_tile: u32,
    pub vae_tile_min: u32,
    pub offload: OffloadMode,
    pub batch: u32,
    /// How many times we have degraded for this job.
    pub rung: u32,
}

impl WorkingSet {
    pub fn initial(settings: &RestorationSettings) -> Self {
        WorkingSet {
            block_swap: settings.block_swap,
            block_swap_max: settings.block_swap_max,
            vae_tile: settings.vae_tile,
            vae_tile_min: settings.vae_tile_min,
            offload: if settings.offload.eq_ignore_ascii_case("cpu") {
                OffloadMode::Cpu
            } else {
                OffloadMode::None
            },
            batch: settings.preferred_batch,
            rung: 0,
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "batch {}, block swap {}/{}, vae tile {}, offload {}",
            self.batch,
            self.block_swap,
            self.block_swap_max,
            self.vae_tile,
            self.offload.as_str()
        )
    }

    /// Estimated resident working set, purely for the "does it fit" conversation.
    pub fn estimated_mib(&self) -> u64 {
        // Weights at FP8 for a 3B model, plus activations that scale with the
        // temporal batch, plus the VAE tile, minus what block swap streams out.
        let weights = 3_400u64;
        let activations = 900 * self.batch as u64;
        let vae = (self.vae_tile as u64 * self.vae_tile as u64) / 256;
        let streamed = (self.block_swap as u64) * 90;
        weights + activations + vae + 800 - streamed.min(2_000)
    }

    /// Next degradation step. `None` when the ladder is exhausted.
    ///
    /// Order is deliberate: throughput first (block swap), then quality (VAE
    /// tile), then the temporal batch — and the batch only ever takes values the
    /// model accepts.
    pub fn degrade(&mut self) -> Option<String> {
        self.rung += 1;
        if self.block_swap < self.block_swap_max {
            let next = (self.block_swap + 4).min(self.block_swap_max);
            let message = format!(
                "increasing block swap {} -> {} (a little slower, same result)",
                self.block_swap, next
            );
            self.block_swap = next;
            return Some(message);
        }
        if self.vae_tile > self.vae_tile_min {
            let next = (self.vae_tile / 2).max(self.vae_tile_min);
            let message = format!(
                "reducing VAE tile {} -> {} (watch for tile seams)",
                self.vae_tile, next
            );
            self.vae_tile = next;
            return Some(message);
        }
        if self.offload == OffloadMode::None {
            self.offload = OffloadMode::Cpu;
            return Some("enabling CPU offload with pinned memory".to_string());
        }
        let valid: Vec<u32> = VALID_TEMPORAL_BATCHES
            .iter()
            .copied()
            .filter(|b| *b < self.batch)
            .collect();
        if let Some(next) = valid.last().copied() {
            let message = format!(
                "reducing temporal batch {} -> {} (valid 4n+1 value; temporal consistency suffers)",
                self.batch, next
            );
            self.batch = next;
            return Some(message);
        }
        self.rung -= 1;
        None
    }

    pub fn is_exhausted(&self) -> bool {
        let mut probe = self.clone();
        probe.degrade().is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuVendor;
    use crate::pipeline::profile::RestorationProfile;

    fn gpu(total: u64, used: u64) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "RTX 5070 Ti".into(),
            total_mib: Some(total),
            used_mib: Some(used),
            source: "nvidia-smi".into(),
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
    fn degrade_ladder_spends_throughput_before_quality() {
        let profile = RestorationProfile::safe_16gb();
        let mut set = WorkingSet::initial(&profile.restoration);
        assert_eq!(set.batch, 5);
        // The default profile already offloads, so the ladder runs
        // block swap -> VAE tile -> temporal batch.
        assert_eq!(set.offload, OffloadMode::Cpu);

        let first = set.degrade().unwrap();
        assert!(first.contains("block swap"), "got: {first}");
        assert_eq!(set.batch, 5, "the batch must not drop first");

        set.degrade();
        assert_eq!(set.block_swap, set.block_swap_max);

        let third = set.degrade().unwrap();
        assert!(third.contains("VAE tile"), "got: {third}");
        assert_eq!(set.batch, 5, "quality is spent after throughput, not before");

        let fourth = set.degrade().unwrap();
        assert!(fourth.contains("temporal batch"), "got: {fourth}");
        assert_eq!(set.batch, 1);
        assert_eq!((set.batch - 1) % 4, 0);
        assert!(set.degrade().is_none(), "the ladder must end");
    }

    #[test]
    fn enabling_offload_is_a_rung_when_it_was_off() {
        let mut settings = RestorationProfile::safe_16gb().restoration;
        settings.offload = "none".into();
        let mut set = WorkingSet::initial(&settings);
        assert_eq!(set.offload, OffloadMode::None);
        // spend block swap, then the VAE tile, and only then reach for offload
        let mut changes = Vec::new();
        while let Some(change) = set.degrade() {
            changes.push(change);
        }
        let offload_rung = changes
            .iter()
            .position(|c| c.contains("offload"))
            .expect("offload must be on the ladder");
        let tile_rung = changes
            .iter()
            .position(|c| c.contains("VAE tile"))
            .expect("VAE tile must be on the ladder");
        let batch_rung = changes
            .iter()
            .position(|c| c.contains("temporal batch"))
            .expect("batch must be on the ladder");
        assert!(tile_rung < offload_rung && offload_rung < batch_rung);
    }

    #[test]
    fn the_ladder_never_produces_batch_three() {
        let profile = RestorationProfile::safe_16gb();
        let mut set = WorkingSet::initial(&profile.restoration);
        let mut seen = vec![set.batch];
        while let Some(_) = set.degrade() {
            seen.push(set.batch);
        }
        assert!(!seen.contains(&3), "batch 3 is not a valid 4n+1 value: {seen:?}");
        for batch in &seen {
            assert_eq!((batch - 1) % 4, 0, "{batch} is not 4n+1");
        }
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
        assert_eq!(set.batch, 1);
        assert_eq!(set.vae_tile, set.vae_tile_min);
        assert_eq!(set.offload, OffloadMode::Cpu);
    }

    #[test]
    fn working_set_estimate_falls_as_we_degrade() {
        let profile = RestorationProfile::safe_16gb();
        let mut set = WorkingSet::initial(&profile.restoration);
        let initial = set.estimated_mib();
        set.degrade();
        set.degrade();
        set.degrade();
        assert!(
            set.estimated_mib() < initial,
            "degrading must reduce the estimated working set"
        );
        assert!(set.describe().contains("block swap"));
    }
}
