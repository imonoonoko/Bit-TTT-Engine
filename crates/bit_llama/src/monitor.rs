//! # Inactive Module (Plan B Fallback)
//! This module implements VRAM monitoring using `nvml-wrapper`.
//! It is currently excluded from compilation (`lib.rs`) due to build instability on Windows/NVCC.
//! Use `cfg(feature = "cuda")` if re-enabling.

use std::time::{Duration, Instant};
use tracing::{debug, error, info};

#[cfg(feature = "cuda")]
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
#[cfg(feature = "cuda")]
use nvml_wrapper::Nvml;

pub struct VramMonitor {
    #[cfg(feature = "cuda")]
    nvml: Option<Nvml>,
    #[cfg(feature = "cuda")]
    device_idx: u32,
    last_poll: Instant,
    cache: Option<(u64, u64)>, // used_mb, total_mb
}

impl VramMonitor {
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            match Nvml::init() {
                Ok(nvml) => {
                    info!("NVML initialized successfully.");
                    Self {
                        nvml: Some(nvml),
                        device_idx: 0, // Default to first GPU
                        last_poll: Instant::now(),
                        cache: None,
                    }
                }
                Err(e) => {
                    error!("Failed to initialize NVML: {}", e);
                    Self {
                        nvml: None,
                        device_idx: 0,
                        last_poll: Instant::now(),
                        cache: None,
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            debug!("VRAM Monitor disabled (cuda feature not enabled).");
            Self {
                last_poll: Instant::now(),
                cache: None,
            }
        }
    }

    pub fn poll(&mut self) -> Option<(u64, u64)> {
        // Rate limit: 1Hz
        if self.last_poll.elapsed() < Duration::from_secs(1) {
            return self.cache;
        }
        self.last_poll = Instant::now();

        #[cfg(feature = "cuda")]
        if let Some(ref nvml) = self.nvml {
            match nvml.device_by_index(self.device_idx) {
                Ok(device) => {
                    if let Ok(mem) = device.memory_info() {
                        let used = mem.used / (1024 * 1024);
                        let total = mem.total / (1024 * 1024);
                        self.cache = Some((used, total));
                        return self.cache;
                    }
                }
                Err(e) => {
                    debug!("Failed to get device {}: {}", self.device_idx, e);
                }
            }
        }

        None
    }

    pub fn current(&self) -> Option<(u64, u64)> {
        self.cache
    }
}
