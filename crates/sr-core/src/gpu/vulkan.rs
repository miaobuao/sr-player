//! Vulkan device enumeration.
//!
//! This is the engine's *primary* view of the machine. Before it existed, the
//! probe was `nvidia-smi`, then `rocm-smi` — which means Intel was never seen at
//! all, an AMD GPU on Windows (where `rocm-smi` does not exist) was never seen,
//! and the single most useful number, "how much memory may this process actually
//! use", was unavailable on two of the three vendors.
//!
//! Vulkan answers all of that from one API that every GPU driver ships:
//! `vendorID`, `deviceID`, `deviceType`, `deviceName`, driver version, and — with
//! `VK_EXT_memory_budget` — the per-heap budget and usage the driver is willing to
//! hand out. Vendor tooling is still used, but only to *sharpen* a Vulkan row,
//! never as the thing that decides whether a GPU exists.
//!
//! Everything here degrades to "no devices" if the loader is missing or the API
//! is unavailable. A machine without Vulkan still runs the deterministic path.

use super::{GpuDeviceType, GpuInfo, GpuVendor};
use std::sync::OnceLock;

/// The Vulkan loader, kept alive for the life of the process: the function
/// pointers ash hands out point into it.
static LOADER: OnceLock<Option<libloading::Library>> = OnceLock::new();

fn loader_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "vulkan-1.dll"
    } else if cfg!(target_os = "macos") {
        "libvulkan.1.dylib"
    } else {
        "libvulkan.so.1"
    }
}

fn loader() -> Option<&'static libloading::Library> {
    LOADER
        .get_or_init(|| {
            // SAFETY: loading a system library to read device properties. The
            // handle is kept for the process lifetime, which is what makes the
            // resolved function pointers valid.
            unsafe { libloading::Library::new(loader_name()) }.ok()
        })
        .as_ref()
}

fn entry() -> Option<ash::Entry> {
    let library = loader()?;
    // SAFETY: `vkGetInstanceProcAddr` has this signature by definition; the
    // library outlives every call made through it.
    let get_instance_proc_addr = unsafe {
        library
            .get::<ash::vk::PFN_vkGetInstanceProcAddr>(b"vkGetInstanceProcAddr\0")
            .ok()
            .map(|symbol| *symbol)
    }?;
    Some(unsafe {
        ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    })
}

/// Driver name and version as strings, when the driver exposes them.
///
/// `VkPhysicalDeviceDriverProperties` reports "NVIDIA" / "617.14" and "AMD
/// proprietary driver" / "26.8.1 (AMD proprietary shader compiler)". Decoding the
/// packed integer is only a fallback for drivers that do not implement it — an
/// integrated AMD part here decodes to "2.0.353" from the integer while reporting
/// 26.8.1 properly through this struct.
fn driver_strings(
    instance: &ash::Instance,
    device: ash::vk::PhysicalDevice,
    supports_driver_properties: bool,
    vendor: GpuVendor,
    raw_version: u32,
) -> (Option<String>, Option<String>) {
    if !supports_driver_properties {
        return (None, Some(driver_version_string(vendor, raw_version)));
    }
    let mut driver = ash::vk::PhysicalDeviceDriverProperties::default();
    let mut props = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
    // SAFETY: the instance is live, the device came from it, and the pNext chain
    // is valid for the duration of the call.
    unsafe { instance.get_physical_device_properties2(device, &mut props) };
    let text = |field: &[std::os::raw::c_char; 256]| -> Option<String> {
        let end = field.iter().position(|c| *c == 0).unwrap_or(field.len());
        let bytes: Vec<u8> = field[..end].iter().map(|c| *c as u8).collect();
        let value = String::from_utf8_lossy(&bytes).trim().to_string();
        (!value.is_empty()).then_some(value)
    };
    let info = text(&driver.driver_info);
    let version = match info {
        Some(info) => info,
        None => driver_version_string(vendor, raw_version),
    };
    let name = text(&driver.driver_name);
    // One readable string: "NVIDIA 617.14", "AMD proprietary driver 26.8.1".
    let combined = match name {
        Some(name) if !name.eq_ignore_ascii_case(vendor.as_str()) => format!("{name} {version}"),
        _ => version,
    };
    (None, Some(combined))
}

/// Version string for a driver version integer, decoded per vendor.
///
/// Each vendor packs major/minor/patch into the same 32 bits differently, and the
/// raw number ("100683776") tells a user nothing about whether their driver is
/// current — which is exactly the question someone asks when a model fails to
/// load.
pub fn driver_version_string(vendor: GpuVendor, raw: u32) -> String {
    match vendor {
        // major[10] minor[8] patch[8] sub[6]
        GpuVendor::Nvidia => format!(
            "{}.{}.{}.{}",
            (raw >> 22) & 0x3ff,
            (raw >> 14) & 0xff,
            (raw >> 6) & 0xff,
            raw & 0x3f
        ),
        // major[10] minor[10] patch[12]
        GpuVendor::Amd => format!(
            "{}.{}.{}",
            (raw >> 22) & 0x3ff,
            (raw >> 12) & 0x3ff,
            raw & 0xfff
        ),
        _ => format!("{}.{}.{}", (raw >> 22) & 0x3ff, (raw >> 12) & 0x3ff, raw & 0xfff),
    }
}

fn vendor_from_id(vendor_id: u32) -> GpuVendor {
    match vendor_id {
        0x10DE => GpuVendor::Nvidia,
        0x1002 | 0x1022 => GpuVendor::Amd,
        0x8086 => GpuVendor::Intel,
        0x106B => GpuVendor::Apple,
        0x5143 => GpuVendor::Qualcomm,
        _ => GpuVendor::Unknown,
    }
}

fn device_type_of(kind: ash::vk::PhysicalDeviceType) -> GpuDeviceType {
    match kind {
        ash::vk::PhysicalDeviceType::DISCRETE_GPU => GpuDeviceType::Discrete,
        ash::vk::PhysicalDeviceType::INTEGRATED_GPU => GpuDeviceType::Integrated,
        ash::vk::PhysicalDeviceType::VIRTUAL_GPU => GpuDeviceType::Virtual,
        ash::vk::PhysicalDeviceType::CPU => GpuDeviceType::Cpu,
        _ => GpuDeviceType::Other,
    }
}

fn device_name(props: &ash::vk::PhysicalDeviceProperties) -> String {
    let name = &props.device_name;
    // SAFETY: the array is NUL-terminated by the API when the name is shorter
    // than the field; a full-length name is handled by the byte scan.
    let end = name.iter().position(|c| *c == 0).unwrap_or(name.len());
    let bytes: Vec<u8> = name[..end].iter().map(|c| *c as u8).collect();
    let text = String::from_utf8_lossy(&bytes).trim().to_string();
    if text.is_empty() {
        "unnamed Vulkan device".to_string()
    } else {
        text
    }
}

fn api_version_string(version: u32) -> String {
    format!(
        "{}.{}.{}",
        ash::vk::api_version_major(version),
        ash::vk::api_version_minor(version),
        ash::vk::api_version_patch(version)
    )
}

/// The device-local heap with the most memory, plus its budget when the driver
/// exposes `VK_EXT_memory_budget`.
struct Heaps {
    total_mib: Option<u64>,
    budget_mib: Option<u64>,
    /// This process's usage, which is what `heapUsage` means. It is *not* the
    /// machine's usage: before a model is loaded it reads as zero even while the
    /// desktop holds several gigabytes.
    process_used_mib: Option<u64>,
}

fn read_heaps(
    instance: &ash::Instance,
    device: ash::vk::PhysicalDevice,
    supports_budget: bool,
) -> Heaps {
    // `get_physical_device_memory_properties2` is core in 1.1; asking for the
    // budget through it needs no logical device, which keeps this probe cheap
    // enough to run before every job.
    let mut budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = ash::vk::PhysicalDeviceMemoryProperties2::default();
    let mut got_budget = false;
    if supports_budget {
        props = props.push_next(&mut budget);
        // SAFETY: the instance is live and the pNext chain is valid for the call.
        unsafe { instance.get_physical_device_memory_properties2(device, &mut props) };
        got_budget = true;
    }
    let memory = if got_budget {
        props.memory_properties
    } else {
        // SAFETY: the instance is live and the device handle came from it.
        unsafe { instance.get_physical_device_memory_properties(device) }
    };

    let mut best: Option<(u64, usize)> = None;
    for index in 0..memory.memory_heap_count as usize {
        let heap = memory.memory_heaps[index];
        let device_local = heap
            .flags
            .contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL);
        if !device_local {
            continue;
        }
        if best.map(|(size, _)| heap.size > size).unwrap_or(true) {
            best = Some((heap.size, index));
        }
    }
    let Some((size, index)) = best else {
        return Heaps {
            total_mib: None,
            budget_mib: None,
            process_used_mib: None,
        };
    };
    let mib = |bytes: u64| bytes / (1024 * 1024);
    let (budget_mib, process_used_mib) = if got_budget {
        let granted = budget.heap_budget[index];
        let usage = budget.heap_usage[index];
        // A driver that does not implement the extension returns zeroes; a zero
        // budget is not information, it is the absence of it.
        (
            (granted > 0).then(|| mib(granted)),
            (granted > 0).then(|| mib(usage.min(granted))),
        )
    } else {
        (None, None)
    };
    Heaps {
        total_mib: Some(mib(size)),
        budget_mib,
        process_used_mib,
    }
}

/// Every Vulkan device, best first. Empty when Vulkan is unavailable.
pub fn probe() -> Vec<GpuInfo> {
    let Some(entry) = entry() else {
        tracing::debug!("no Vulkan loader ({}): falling back to vendor tooling", loader_name());
        return Vec::new();
    };

    let app = ash::vk::ApplicationInfo::default()
        .application_name(c"sr-player gpu probe")
        .application_version(ash::vk::make_api_version(0, 1, 0, 0))
        // 1.2 for `VkPhysicalDeviceDriverProperties`; every driver from the last
        // several years reports at least this.
        .api_version(ash::vk::make_api_version(0, 1, 2, 0));
    let create = ash::vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: a plain instance with no layers and no extensions.
    let instance = match unsafe { entry.create_instance(&create, None) } {
        Ok(instance) => instance,
        Err(err) => {
            // An older loader: retry at 1.1, which still has the memory query the
            // budget needs. Losing the driver strings is a cosmetic loss.
            tracing::debug!("Vulkan 1.2 instance refused ({err}); retrying at 1.1");
            let app = app.api_version(ash::vk::make_api_version(0, 1, 1, 0));
            let create = ash::vk::InstanceCreateInfo::default().application_info(&app);
            match unsafe { entry.create_instance(&create, None) } {
                Ok(instance) => instance,
                Err(err) => {
                    tracing::debug!("Vulkan instance creation failed: {err}");
                    return Vec::new();
                }
            }
        }
    };

    // SAFETY: the instance is live for the whole function.
    let devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(devices) => devices,
        Err(err) => {
            tracing::debug!("Vulkan device enumeration failed: {err}");
            // SAFETY: created above, not used again.
            unsafe { instance.destroy_instance(None) };
            return Vec::new();
        }
    };

    let mut rows: Vec<GpuInfo> = Vec::new();
    for (index, device) in devices.iter().enumerate() {
        // SAFETY: the device handle came from this instance.
        let props = unsafe { instance.get_physical_device_properties(*device) };
        let vendor = vendor_from_id(props.vendor_id);
        // SAFETY: as above.
        let extensions = unsafe { instance.enumerate_device_extension_properties(*device) }
            .unwrap_or_default();
        let has_extension = |wanted: &[u8]| {
            extensions.iter().any(|ext| {
                let name = ext.extension_name;
                let end = name.iter().position(|c| *c == 0).unwrap_or(name.len());
                let bytes: Vec<u8> = name[..end].iter().map(|c| *c as u8).collect();
                bytes == wanted
            })
        };
        let supports_budget = has_extension(b"VK_EXT_memory_budget");
        let supports_driver_properties = has_extension(b"VK_KHR_driver_properties");
        let heaps = read_heaps(&instance, *device, supports_budget);
        let (driver_name, driver_version) = driver_strings(
            &instance,
            *device,
            supports_driver_properties,
            vendor,
            props.driver_version,
        );

        let device_type = device_type_of(props.device_type);
        rows.push(GpuInfo {
            index: index as u32,
            vendor,
            name: device_name(&props),
            total_mib: heaps.total_mib,
            used_mib: None,
            process_used_mib: heaps.process_used_mib,
            budget_mib: heaps.budget_mib,
            device_type,
            // An integrated GPU's "device-local" heap is system RAM, so a free
            // figure for it says nothing about whether a model fits: it competes
            // with everything else the machine is doing.
            unified_memory: device_type == GpuDeviceType::Integrated
                || device_type == GpuDeviceType::Cpu,
            vendor_id: Some(props.vendor_id),
            device_id: Some(props.device_id),
            api_version: Some(api_version_string(props.api_version)),
            driver: driver_version,
            memory_budget_supported: supports_budget,
            source: "vulkan".to_string(),
            telemetry: driver_name.into_iter().collect(),
        });
    }

    // SAFETY: the instance was created above and is not used afterwards.
    unsafe { instance.destroy_instance(None) };

    // Discrete first, then by memory: the plan wants the card a model should live
    // on, and a machine with an iGPU and a dGPU has an obvious answer.
    rows.sort_by(|a, b| {
        let rank = |g: &GpuInfo| match g.device_type {
            GpuDeviceType::Discrete => 0,
            GpuDeviceType::Integrated => 2,
            _ => 1,
        };
        rank(a)
            .cmp(&rank(b))
            .then(b.total_mib.unwrap_or(0).cmp(&a.total_mib.unwrap_or(0)))
    });
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_versions_decode_per_vendor() {
        // 617.14 as NVIDIA packs it.
        let nvidia = (617u32 << 22) | (14u32 << 14);
        assert_eq!(driver_version_string(GpuVendor::Nvidia, nvidia), "617.14.0.0");
        // 26.8.1 as AMD packs it.
        let amd = (26u32 << 22) | (8u32 << 12) | 1;
        assert_eq!(driver_version_string(GpuVendor::Amd, amd), "26.8.1");
        // An unknown vendor still produces something readable.
        assert!(driver_version_string(GpuVendor::Unknown, 0x04000000).starts_with('1'));
    }

    #[test]
    fn vendors_map_from_pci_ids() {
        assert_eq!(vendor_from_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(vendor_from_id(0x1002), GpuVendor::Amd);
        assert_eq!(vendor_from_id(0x8086), GpuVendor::Intel);
        assert_eq!(vendor_from_id(0x9999), GpuVendor::Unknown);
    }

    #[test]
    fn the_probe_never_panics_and_reports_consistent_rows() {
        // On a machine with no Vulkan this returns nothing; on this one it should
        // find the physical devices. Either way the invariants must hold.
        let rows = probe();
        for row in &rows {
            assert_eq!(row.source, "vulkan");
            assert!(!row.name.is_empty());
            if let (Some(budget), Some(used)) = (row.budget_mib, row.used_mib) {
                assert!(used <= budget, "used more than the budget in {row:?}");
            }
            if let Some(total) = row.total_mib {
                assert!(total > 0);
            }
        }
        if !rows.is_empty() {
            eprintln!("vulkan devices: {:?}", rows.iter().map(|r| r.describe()).collect::<Vec<_>>());
            // Discrete devices sort first so the planner picks the right card.
            let ranks: Vec<u8> = rows
                .iter()
                .map(|r| match r.device_type {
                    GpuDeviceType::Discrete => 0,
                    GpuDeviceType::Integrated => 2,
                    _ => 1,
                })
                .collect();
            assert!(ranks.windows(2).all(|w| w[0] <= w[1]), "rows are not sorted: {ranks:?}");
        }
    }
}
