use crate::kernels::packing::PackedTensor;
use candle_core::{Device, Result, Tensor};

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
// use candle_core::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};

// Compile time PTX embedding
#[cfg(feature = "cuda")]
const _BIT_OP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/bit_op.ptx"));
#[cfg(feature = "cuda")]
const ADAPTIVE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/adaptive_bit_op.ptx"));

pub struct BitLinearCuda;

impl BitLinearCuda {
    // Legacy fallback or future 1-bit kernel
    pub fn forward(
        input: &Tensor,         // [M, K]
        weights: &PackedTensor, // [N, K/4]
    ) -> Result<Tensor> {
        let (m, k) = input.dims2()?;
        let (n_out, k_w) = weights.shape.dims2()?;
        if k != k_w {
            candle_core::bail!(
                "Shape mismatch: Input [{}, {}] vs Weight [{}, {}]",
                m,
                k,
                n_out,
                k_w
            );
        }

        let device = input.device();
        match device {
            Device::Cuda(dev) => {
                let w_dequant = weights.unpack(&Device::Cuda(dev.clone()))?;
                let w_t = w_dequant.t()?;
                let output = input.matmul(&w_t)?;
                Ok(output)
            }
            _ => candle_core::bail!("BitLinearCuda called on non-CUDA device"),
        }
    }

    // NEW: Adaptive Fused Kernel
    #[cfg(feature = "cuda")]
    pub fn adaptive_forward(
        input: &Tensor,   // [Batch, In] (F32)
        weights: &Tensor, // [Out, In/4, NumBases, 4] -> Flattened I8
        scales: &Tensor,  // [NumBases] (F32)
    ) -> Result<Tensor> {
        use candle_core::cuda_backend::cudarc::driver::CudaDevice as DriverCudaDevice;

        let (batch, in_dim) = input.dims2()?;

        let w_dims = weights.dims();
        let out_dim = w_dims[0];

        let dev = match input.device() {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("adaptive_forward called on non-CUDA device"),
        };

        // 1. Get raw pointers
        // Use scopes to drop ReadGuards immediately after getting pointers
        let inp_ptr = {
            let inp_storage = input.storage_and_layout().0;
            match &*inp_storage {
                candle_core::Storage::Cuda(s) => *s.as_cuda_slice::<f32>()?.device_ptr(),
                _ => candle_core::bail!("Input must be CUDA F32"),
            }
        };

        let w_ptr = {
            let w_storage = weights.storage_and_layout().0;
            match &*w_storage {
                candle_core::Storage::Cuda(s) => *s.as_cuda_slice::<u8>()?.device_ptr(),
                _ => candle_core::bail!("Weights must be CUDA U8"),
            }
        };
        let w_cu_ptr = w_ptr;

        let s_ptr = {
            let s_storage = scales.storage_and_layout().0;
            match &*s_storage {
                candle_core::Storage::Cuda(s) => *s.as_cuda_slice::<f32>()?.device_ptr(),
                _ => candle_core::bail!("Scales must be CUDA F32"),
            }
        };

        // 2. Allocate Output
        let output = Tensor::zeros(
            (batch, out_dim),
            candle_core::DType::F32,
            &Device::Cuda(dev.clone()),
        )?;
        let out_ptr = {
            let out_storage = output.storage_and_layout().0;
            match &*out_storage {
                candle_core::Storage::Cuda(s) => *s.as_cuda_slice::<f32>()?.device_ptr(),
                _ => candle_core::bail!("Output alloc failed"),
            }
        };

        // 3. Launch Kernel
        let module_name = "adaptive_gemm";
        let func_name = "adaptive_gemm_n3_kernel_f32";

        let ordinal = *dev.cu_device() as usize;
        let core_dev = DriverCudaDevice::new(ordinal).map_err(candle_core::Error::wrap)?;

        core_dev
            .load_ptx(ADAPTIVE_PTX.into(), module_name, &[func_name])
            .map_err(candle_core::Error::wrap)?;

        let f = core_dev
            .get_func(module_name, func_name)
            .ok_or_else(|| anyhow::anyhow!("Kernel not found"))
            .map_err(candle_core::Error::wrap)?;

        let block_dim = 256;
        let grid_x = (out_dim as u32 + block_dim - 1) / block_dim;
        let grid_y = batch as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let params = (
            inp_ptr,  // X
            w_cu_ptr, // W (Packed)
            s_ptr,    // Scales
            out_ptr,  // Y
            batch as i32,
            in_dim as i32,
            out_dim as i32,
        );

        unsafe { f.launch(cfg, params) }.map_err(candle_core::Error::wrap)?;

        Ok(output)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn adaptive_forward(
        _input: &Tensor,
        _weights: &Tensor,
        _scales: &Tensor,
    ) -> Result<Tensor> {
        candle_core::bail!("CUDA implementation not enabled (feature 'cuda' missing)")
    }

    pub fn smoke_test_compile() -> bool {
        #[cfg(feature = "cuda")]
        return !ADAPTIVE_PTX.is_empty();
        #[cfg(not(feature = "cuda"))]
        return false;
    }
}
