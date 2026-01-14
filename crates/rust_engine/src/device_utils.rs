use anyhow::Result;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
// DriverError is not used, removing it.

/// Returns (free_memory, total_memory) in bytes for the specified device.
/// Returns (0, 0) if CUDA is not available or disabled.
pub fn get_vram_info(_device_id: usize) -> Result<(usize, usize)> {
    #[cfg(feature = "cuda")]
    {
        // We use cudarc to query memory info.
        // CudaDevice::new(id) initializes the context.
        // It returns Arc<CudaDevice>.

        // Note: CudaDevice::new might be heavy or fail if no GPU.
        match CudaDevice::new(_device_id) {
            Ok(_dev) => {
                // Use driver::result::mem_get_info which wraps cuMemGetInfo_v2
                use cudarc::driver::result::mem_get_info;
                let (free, total) = mem_get_info()?;
                Ok((free, total))
            }
            Err(e) => {
                // If we can't initialize CUDA, assume CPU mode.
                // Log warning?
                eprintln!(
                    "Warning: Failed to initialize CUDA device {}: {:?}",
                    _device_id, e
                );
                Ok((0, 0))
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        Ok((0, 0))
    }
}
