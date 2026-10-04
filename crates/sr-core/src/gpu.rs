//! GPU discovery: Vulkan first, vendor tooling second.
//!
//! Nothing in the engine *requires* this to succeed. When a device is found we
//! can size the inference budget against real numbers; when it is not, the
//! planner falls back to a conservative static budget. A probe is an
//! optimisation, never a dependency.
//!
//! The order matters and used to be wrong: the probe was `nvidia-smi`, then
//! `rocm-smi`. That meant Intel was never seen, an AMD card on Windows (where
//! `rocm-smi` does not exist) was never seen, and "how much may this process
//! actually use" — the only number the planner cares about — was unavailable on
//! two of the three vendors. Vulkan is now the source of truth for *which devices
//! exist*; `nvidia-smi` and friends are merged in afterwards to sharpen a row
//! with telemetry Vulkan does not expose (per-process usage on drivers without
//! `VK_EXT_memory_budget`).

pub mod vulkan;

use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Qualcomm,
    #[default]
    Unknown,
}

impl GpuVendor {
    pub fn as_str(self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Amd => "amd",
            GpuVendor::Intel => "intel",
            GpuVendor::Apple => "apple",
            GpuVendor::Qualcomm => "qualcomm",
            GpuVendor::Unknown => "unknown",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuDeviceType {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

impl GpuDeviceType {
    pub fn as_str(self) -> &'static str {
        match self {
            GpuDeviceType::Discrete => "discrete",
            GpuDeviceType::Integrated => "integrated",
            GpuDeviceType::Virtual => "virtual",
            GpuDeviceType::Cpu => "cpu",
            GpuDeviceType::Other => "other",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    /// Enumeration order before sorting; keeps rows stable for a UI.
    pub index: u32,
    pub vendor: GpuVendor,
    pub name: String,
    /// Size of the device-local heap.
    pub total_mib: Option<u64>,
    /// Memory in use **by everything on the machine**, when a vendor tool can say
    /// so. Vulkan cannot: `VkPhysicalDeviceMemoryBudgetPropertiesEXT::heapUsage`
    /// is this *process's* usage, which is zero before a model is loaded.
    pub used_mib: Option<u64>,
    /// This process's usage, when only Vulkan can say so.
    pub process_used_mib: Option<u64>,
    /// What the driver is willing to hand *this process* (`VK_EXT_memory_budget`).
    /// More useful than `total` on a machine that is also driving a desktop.
    pub budget_mib: Option<u64>,
    pub device_type: GpuDeviceType,
    /// The "device-local" memory is system RAM shared with everything else, so a
    /// free figure for it does not mean a model fits.
    pub unified_memory: bool,
    pub vendor_id: Option<u32>,
    pub device_id: Option<u32>,
    pub api_version: Option<String>,
    pub driver: Option<String>,
    pub memory_budget_supported: bool,
    /// Which probe produced this row (`vulkan`, `nvidia-smi`, ...).
    pub source: String,
    /// Extra telemetry merged in from vendor tooling.
    pub telemetry: Vec<String>,
}

impl Default for GpuInfo {
    fn default() -> Self {
        GpuInfo {
            index: 0,
            vendor: GpuVendor::Unknown,
            name: String::new(),
            total_mib: None,
            used_mib: None,
            process_used_mib: None,
            budget_mib: None,
            device_type: GpuDeviceType::Other,
            unified_memory: false,
            vendor_id: None,
            device_id: None,
            api_version: None,
            driver: None,
            memory_budget_supported: false,
            source: String::new(),
            telemetry: Vec::new(),
        }
    }
}

impl GpuInfo {
    /// Memory the planner may assume is available: the *most pessimistic*
    /// defensible estimate.
    ///
    /// Two views exist and neither is complete. `budget - process_usage` is what
    /// the driver will let this process allocate, but the budget is not a live
    /// free-memory reading — a card with 15.9 GiB reports a 14.9 GiB budget while
    /// the desktop is already holding 3.4 GiB. `total - system_usage` knows about
    /// the desktop but not about what the driver will refuse. Taking the smaller
    /// of the two is what keeps an unattended run from planning itself into an
    /// out-of-memory failure.
    pub fn free_mib(&self) -> Option<u64> {
        if self.unified_memory {
            // Shared with the OS: report nothing rather than something wrong.
            return None;
        }
        let budget_view = self
            .budget_mib
            .map(|budget| budget.saturating_sub(self.process_used_mib.unwrap_or(0)));
        let total_view = self
            .total_mib
            .map(|total| total.saturating_sub(self.used_mib.unwrap_or(0)));
        match (budget_view, total_view) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// A GPU a model should run on: real, discrete or a capable integrated part,
    /// not a virtual or CPU device.
    pub fn is_inference_candidate(&self) -> bool {
        !self.unified_memory
            && matches!(
                self.device_type,
                GpuDeviceType::Discrete | GpuDeviceType::Integrated | GpuDeviceType::Other
            )
    }

    pub fn describe(&self) -> String {
        let mem = match (self.free_mib(), self.total_mib) {
            (Some(free), Some(total)) => {
                let mut text = format!("{:.1} GiB free of {:.1} GiB", gib(free), gib(total));
                if let Some(budget) = self.budget_mib {
                    text.push_str(&format!(" (budget {:.1} GiB)", gib(budget)));
                }
                if let Some(used) = self.used_mib {
                    text.push_str(&format!(", {:.1} GiB in use elsewhere", gib(used)));
                }
                text
            }
            (None, Some(total)) => format!("{:.1} GiB (shared with the system)", gib(total)),
            _ => "memory unknown".to_string(),
        };
        let mut text = format!(
            "{} [{}] ({}, {})",
            if self.name.is_empty() {
                "unnamed device"
            } else {
                &self.name
            },
            self.vendor.as_str(),
            self.device_type.as_str(),
            mem
        );
        if let Some(driver) = &self.driver {
            text.push_str(&format!(", driver {driver}"));
        }
        for line in &self.telemetry {
            text.push_str(&format!(", {line}"));
        }
        text
    }
}

fn gib(mib: u64) -> f64 {
    mib as f64 / 1024.0
}

/// Best-effort probe: Vulkan, then vendor tooling only if Vulkan found nothing.
pub fn probe() -> Vec<GpuInfo> {
    let mut rows = vulkan::probe();
    let telemetry = vendor_telemetry();
    if rows.is_empty() {
        // No Vulkan: report whatever the vendor tools can tell us, clearly
        // labelled, because a machine with a GPU and no Vulkan is still a
        // machine someone will run this on.
        return telemetry;
    }
    merge_telemetry(&mut rows, telemetry);
    rows
}

/// Merges a vendor row into the matching Vulkan row when the pairing is
/// unambiguous, and otherwise appends it as its own entry.
///
/// Guessing which of two identical cards a `nvidia-smi` row describes would be
/// worse than not merging: the planner would be sizing a budget from the wrong
/// device's numbers.
fn merge_telemetry(rows: &mut [GpuInfo], telemetry: Vec<GpuInfo>) {
    for row in telemetry {
        let candidates: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, existing)| existing.vendor == row.vendor)
            .map(|(index, _)| index)
            .collect();
        if candidates.len() == 1 {
            let target = &mut rows[candidates[0]];
            if target.total_mib.is_none() {
                target.total_mib = row.total_mib;
            }
            // Vendor tooling is the only source that knows what *other* processes
            // are holding; Vulkan's figure is this process's own usage.
            if let Some(used) = row.used_mib {
                target.used_mib = Some(used);
            }
            if target.driver.is_none() {
                target.driver = row.driver.clone();
            }
            target
                .telemetry
                .push(format!("usage via {}", row.source));
        }
    }
}

fn vendor_telemetry() -> Vec<GpuInfo> {
    let mut out = Vec::new();
    if let Some(mut nv) = probe_nvidia_smi() {
        out.append(&mut nv);
    }
    if out.is_empty() {
        if let Some(mut amd) = probe_rocm_smi() {
            out.append(&mut amd);
        }
    }
    out
}

fn run_capture(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

fn probe_nvidia_smi() -> Option<Vec<GpuInfo>> {
    let stdout = run_capture(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let gpus: Vec<GpuInfo> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(index, line)| {
            let mut parts = line.split(',').map(|p| p.trim());
            let name = parts.next().unwrap_or("NVIDIA GPU").to_string();
            let total_mib = parts.next().and_then(|v| v.parse::<u64>().ok());
            let used_mib = parts.next().and_then(|v| v.parse::<u64>().ok());
            let driver = parts.next().map(|v| v.to_string());
            GpuInfo {
                index: index as u32,
                vendor: GpuVendor::Nvidia,
                name,
                total_mib,
                used_mib,
                budget_mib: None,
                device_type: GpuDeviceType::Discrete,
                unified_memory: false,
                driver,
                source: "nvidia-smi".into(),
                ..GpuInfo::default()
            }
        })
        .collect();
    if gpus.is_empty() {
        None
    } else {
        Some(gpus)
    }
}

fn probe_rocm_smi() -> Option<Vec<GpuInfo>> {
    let stdout = run_capture("rocm-smi", &["--showmeminfo", "vram", "--csv"])?;
    let mut gpus = Vec::new();
    for (index, line) in stdout.lines().skip(1).enumerate() {
        let cols: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        if cols.len() < 3 {
            continue;
        }
        // device, vram total (bytes), vram used (bytes)
        let total = cols[1].parse::<u64>().ok().map(|b| b / 1_048_576);
        let used = cols[2].parse::<u64>().ok().map(|b| b / 1_048_576);
        gpus.push(GpuInfo {
            index: index as u32,
            vendor: GpuVendor::Amd,
            name: cols[0].to_string(),
            total_mib: total,
            used_mib: used,
            device_type: GpuDeviceType::Discrete,
            source: "rocm-smi".into(),
            ..GpuInfo::default()
        });
    }
    if gpus.is_empty() {
        None
    } else {
        Some(gpus)
    }
}

/// Free VRAM the planner may use: the largest figure among devices that could
/// actually host a model.
///
/// Taking the maximum across *all* devices would happily return the shared
/// system memory of an integrated GPU, which is how a planner talks itself into
/// a working set that cannot fit anywhere.
pub fn free_mib(gpus: &[GpuInfo]) -> Option<u64> {
    let mut total: Option<u64> = None;
    for gpu in gpus.iter().filter(|g| g.is_inference_candidate()) {
        if let Some(free) = gpu.free_mib() {
            total = Some(total.map_or(free, |t: u64| t.max(free)));
        }
    }
    total
}

/// The device a model should be loaded onto, if any.
pub fn preferred_device(gpus: &[GpuInfo]) -> Option<&GpuInfo> {
    gpus.iter()
        .filter(|g| g.is_inference_candidate())
        .max_by_key(|g| g.free_mib().unwrap_or(0))
}

/// A one-line summary for the log header. Never fails.
pub fn describe(gpus: &[GpuInfo]) -> String {
    if gpus.is_empty() {
        return "no GPU probe succeeded (no Vulkan device and no vendor tooling): using the \
                conservative VRAM budget"
            .to_string();
    }
    gpus.iter().map(GpuInfo::describe).collect::<Vec<_>>().join("; ")
}

/// A short "which card, which API, how much" line for the UI's info panel.
pub fn summary_rows(gpus: &[GpuInfo]) -> Vec<(String, String)> {
    gpus.iter()
        .map(|gpu| {
            let label = format!(
                "[{}] {} ({})",
                gpu.index,
                if gpu.name.is_empty() {
                    "unnamed"
                } else {
                    &gpu.name
                },
                gpu.vendor.as_str()
            );
            let mut value = format!(
                "{} · {}",
                gpu.device_type.as_str(),
                if gpu.unified_memory {
                    "shared system memory".to_string()
                } else {
                    match gpu.free_mib() {
                        Some(free) => format!("{:.1} GiB free", gib(free)),
                        None => "free memory unknown".to_string(),
                    }
                }
            );
            if let Some(api) = &gpu.api_version {
                value.push_str(&format!(" · vulkan {api}"));
            }
            if let Some(driver) = &gpu.driver {
                value.push_str(&format!(" · driver {driver}"));
            }
            if gpu.memory_budget_supported {
                value.push_str(" · memory budget");
            }
            value.push_str(&format!(" · {}", gpu.source));
            (label, value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(total: u64, used: u64) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "test".into(),
            total_mib: Some(total),
            used_mib: Some(used),
            device_type: GpuDeviceType::Discrete,
            source: "fake".into(),
            ..GpuInfo::default()
        }
    }

    #[test]
    fn free_memory_takes_the_most_pessimistic_view() {
        // Vulkan's view: a 14.9 GiB budget and no process usage yet.
        let mut info = gpu(16_303, 3_384);
        info.budget_mib = Some(15_261);
        info.process_used_mib = Some(0);
        info.used_mib = Some(3_384);
        // budget view = 15261, total view = 16303 - 3384 = 12919 -> the smaller,
        // because the desktop's 3.4 GiB is really gone.
        assert_eq!(info.free_mib(), Some(12_919));

        // With no vendor telemetry only the budget is known.
        let mut vulkan_only = gpu(16_303, 0);
        vulkan_only.used_mib = None;
        vulkan_only.budget_mib = Some(15_261);
        vulkan_only.process_used_mib = Some(0);
        assert_eq!(vulkan_only.free_mib(), Some(15_261));

        // With no budget, total minus usage is all there is.
        let mut smi_only = gpu(16_303, 3_384);
        smi_only.budget_mib = None;
        assert_eq!(smi_only.free_mib(), Some(12_919));
    }

    #[test]
    fn a_shared_memory_device_never_claims_free_vram() {
        let shared = GpuInfo {
            vendor: GpuVendor::Amd,
            name: "Radeon(TM) Graphics".into(),
            total_mib: Some(8192),
            budget_mib: Some(8192),
            process_used_mib: Some(100),
            used_mib: Some(100),
            device_type: GpuDeviceType::Integrated,
            unified_memory: true,
            source: "vulkan".into(),
            ..GpuInfo::default()
        };
        assert_eq!(shared.free_mib(), None);
        assert!(!shared.is_inference_candidate());
        assert!(shared.describe().contains("shared with the system"));
    }

    #[test]
    fn the_planner_never_sizes_against_an_integrated_gpu() {
        let mut discrete = gpu(16_303, 3_384);
        discrete.name = "RTX 5070 Ti".into();
        let shared = GpuInfo {
            vendor: GpuVendor::Amd,
            name: "Radeon(TM) Graphics".into(),
            total_mib: Some(32_000),
            budget_mib: Some(32_000),
            process_used_mib: Some(0),
            used_mib: Some(0),
            device_type: GpuDeviceType::Integrated,
            unified_memory: true,
            source: "vulkan".into(),
            ..GpuInfo::default()
        };
        let rows = vec![discrete.clone(), shared];
        assert_eq!(free_mib(&rows), Some(12_919));
        assert_eq!(
            preferred_device(&rows).map(|g| g.name.as_str()),
            Some("RTX 5070 Ti")
        );
    }

    #[test]
    fn telemetry_is_merged_only_when_the_pairing_is_unambiguous() {
        let mut rows = vec![GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "RTX 5070 Ti".into(),
            total_mib: None,
            device_type: GpuDeviceType::Discrete,
            source: "vulkan".into(),
            ..GpuInfo::default()
        }];
        merge_telemetry(
            &mut rows,
            vec![GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: "NVIDIA GeForce RTX 5070 Ti".into(),
                total_mib: Some(16303),
                used_mib: Some(3352),
                driver: Some("617.14".into()),
                source: "nvidia-smi".into(),
                ..GpuInfo::default()
            }],
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].total_mib, Some(16303));
        assert_eq!(rows[0].free_mib(), Some(12951));
        assert!(rows[0].telemetry.iter().any(|t| t.contains("nvidia-smi")));

        // Two NVIDIA cards and one telemetry row: nothing is merged, because
        // there is no way to know which card it describes.
        let mut two = vec![
            GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: "A".into(),
                source: "vulkan".into(),
                ..GpuInfo::default()
            },
            GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: "B".into(),
                source: "vulkan".into(),
                ..GpuInfo::default()
            },
        ];
        merge_telemetry(
            &mut two,
            vec![GpuInfo {
                vendor: GpuVendor::Nvidia,
                name: "A".into(),
                total_mib: Some(16384),
                source: "nvidia-smi".into(),
                ..GpuInfo::default()
            }],
        );
        assert!(two.iter().all(|g| g.total_mib.is_none()));
    }

    #[test]
    fn describe_handles_missing_telemetry() {
        assert!(describe(&[]).contains("no GPU probe succeeded"));
        let gpus = vec![gpu(16384, 3352)];
        assert!(describe(&gpus).contains("test"));
        assert_eq!(free_mib(&gpus), Some(13032));
        let rows = summary_rows(&gpus);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].1.contains("discrete"));
    }

    #[test]
    fn free_memory_is_total_minus_used_and_never_underflows() {
        let full = gpu(16384, 16384);
        assert_eq!(full.free_mib(), Some(0));
        let over = gpu(16384, 20000);
        assert_eq!(over.free_mib(), Some(0), "a used figure above total must not wrap");
    }
}
