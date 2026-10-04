//! Optional vendor GPU probe.
//!
//! Nothing in the engine *requires* this to succeed. When `nvidia-smi` or
//! `rocm-smi` happen to exist we get a real free-VRAM number and can size the
//! inference budget precisely; when they do not, the planner falls back to a
//! conservative static budget. That is the whole point of the vendor-neutral
//! design: a probe is an optimisation, never a dependency.

use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

impl GpuVendor {
    pub fn as_str(self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Amd => "amd",
            GpuVendor::Intel => "intel",
            GpuVendor::Apple => "apple",
            GpuVendor::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub total_mib: Option<u64>,
    pub used_mib: Option<u64>,
    /// Which probe produced this row (`nvidia-smi`, `rocm-smi`, ...).
    pub source: String,
}

impl GpuInfo {
    pub fn free_mib(&self) -> Option<u64> {
        match (self.total_mib, self.used_mib) {
            (Some(total), Some(used)) => Some(total.saturating_sub(used)),
            (Some(total), None) => Some(total),
            _ => None,
        }
    }
}

/// Best-effort probe. Returns an empty vec on a machine with no vendor tooling.
pub fn probe() -> Vec<GpuInfo> {
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
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let gpus: Vec<GpuInfo> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut parts = line.split(',').map(|p| p.trim());
            let name = parts.next().unwrap_or("NVIDIA GPU").to_string();
            let total_mib = parts.next().and_then(|v| v.parse::<u64>().ok());
            let used_mib = parts.next().and_then(|v| v.parse::<u64>().ok());
            GpuInfo {
                vendor: GpuVendor::Nvidia,
                name,
                total_mib,
                used_mib,
                source: "nvidia-smi".into(),
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
    for line in stdout.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        if cols.len() < 3 {
            continue;
        }
        // device, vram total (bytes), vram used (bytes)
        let total = cols[1].parse::<u64>().ok().map(|b| b / 1_048_576);
        let used = cols[2].parse::<u64>().ok().map(|b| b / 1_048_576);
        gpus.push(GpuInfo {
            vendor: GpuVendor::Amd,
            name: cols[0].to_string(),
            total_mib: total,
            used_mib: used,
            source: "rocm-smi".into(),
        });
    }
    if gpus.is_empty() {
        None
    } else {
        Some(gpus)
    }
}

/// Total free VRAM across all probed devices, if any probe succeeded.
pub fn free_mib(gpus: &[GpuInfo]) -> Option<u64> {
    let mut total: Option<u64> = None;
    for gpu in gpus {
        if let Some(free) = gpu.free_mib() {
            total = Some(total.map_or(free, |t: u64| t.max(free)));
        }
    }
    total
}

/// A one-line summary for the log header. Never fails.
pub fn describe(gpus: &[GpuInfo]) -> String {
    if gpus.is_empty() {
        return "no vendor GPU telemetry available (using conservative VRAM budget)".to_string();
    }
    gpus.iter()
        .map(|g| {
            let mem = match (g.total_mib, g.used_mib) {
                (Some(t), Some(u)) => format!("{t} MiB total / {u} MiB used"),
                (Some(t), None) => format!("{t} MiB total"),
                _ => "memory unknown".to_string(),
            };
            format!("{} [{}] ({mem})", g.name, g.vendor.as_str())
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_memory_is_total_minus_used_and_never_underflows() {
        let gpu = GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "test".into(),
            total_mib: Some(16384),
            used_mib: Some(16384),
            source: "fake".into(),
        };
        assert_eq!(gpu.free_mib(), Some(0));
        let gpu = GpuInfo {
            used_mib: Some(100),
            ..gpu
        };
        assert_eq!(gpu.free_mib(), Some(16284));
    }

    #[test]
    fn describe_handles_missing_telemetry() {
        assert!(describe(&[]).contains("conservative"));
        let gpus = vec![GpuInfo {
            vendor: GpuVendor::Amd,
            name: "Radeon".into(),
            total_mib: Some(16384),
            used_mib: None,
            source: "rocm-smi".into(),
        }];
        assert!(describe(&gpus).contains("Radeon"));
        assert_eq!(free_mib(&gpus), Some(16384));
    }
}
