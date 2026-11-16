//! Process memory measurement (peak resident set), no external crates.
#[cfg(target_os = "linux")]
pub fn peak_rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).map(|l| l.to_string()))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()))
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

#[cfg(target_os = "linux")]
pub fn current_rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()))
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

#[cfg(windows)]
mod win {
    #[repr(C)]
    pub struct ProcessMemoryCounters {
        pub cb: u32,
        pub page_fault_count: u32,
        pub peak_working_set_size: usize,
        pub working_set_size: usize,
        pub quota_peak_paged_pool_usage: usize,
        pub quota_paged_pool_usage: usize,
        pub quota_peak_non_paged_pool_usage: usize,
        pub quota_non_paged_pool_usage: usize,
        pub pagefile_usage: usize,
        pub peak_pagefile_usage: usize,
    }
    extern "system" {
        pub fn GetCurrentProcess() -> isize;
        pub fn K32GetProcessMemoryInfo(h: isize, p: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }
    pub fn counters() -> Option<ProcessMemoryCounters> {
        unsafe {
            let mut c: ProcessMemoryCounters = std::mem::zeroed();
            c.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 {
                Some(c)
            } else {
                None
            }
        }
    }
}

#[cfg(windows)]
pub fn peak_rss_bytes() -> u64 {
    win::counters().map(|c| c.peak_working_set_size as u64).unwrap_or(0)
}
#[cfg(windows)]
pub fn current_rss_bytes() -> u64 {
    win::counters().map(|c| c.working_set_size as u64).unwrap_or(0)
}

#[cfg(not(any(target_os = "linux", windows)))]
pub fn peak_rss_bytes() -> u64 {
    0
}
#[cfg(not(any(target_os = "linux", windows)))]
pub fn current_rss_bytes() -> u64 {
    0
}

pub fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else {
        format!("{:.0} KB", b / 1e3)
    }
}
