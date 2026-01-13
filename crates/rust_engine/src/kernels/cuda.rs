use candle_core::{Device, Result, Storage, Tensor};
use std::sync::Arc;

// Only compile this module if CUDA feature is enabled
#[cfg(feature = "cuda")]
use candle_core::cuda::{
    cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig},
    CudaDevice, CudaStorage, CudaStorageSlice,
};
#[cfg(feature = "cuda")]
use candle_core::{DType, Shape, CustomOp2};

// Load PTX Source at Compile Time
// This requires build.rs to compile bit_op.cu -> bit_op.ptx in OUT_DIR
// We use a dummy string if not cuda to allow check/clippy to pass
#[cfg(feature = "cuda")]
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/bit_op.ptx"));
#[cfg(not(feature = "cuda"))]
const PTX_SRC: &str = "";

/// Native CUDA implementation of BitLinear (1.58-bit)
/// Stores 2-bit packed weights in Resident VRAM.
/// Implements CustomOp2 for Autograd support.
#[derive(Debug, Clone)]
pub struct BitLinearOp {
    #[cfg(feature = "cuda")]
    device: CudaDevice,

    // Forward Weights: (N, K/4)
    #[cfg(feature = "cuda")]
    packed_weights: CudaSlice<u8>,

    // Backward Weights: (K, N/4) - Transposed Packed
    // Used for dL/dx = dL/dy @ W_quant = dL/dy @ W_quant.t().t() ... wait
    // dL/dx = (Batch, Out) @ (Out, In)
    // dL/dx^T = (In, Out) @ (Batch, Out)^T = (In, Out) @ (Out, Batch)
    // This matches GEMV signature: (In, Out) * (Out, 1) -> (In, 1).
    // So we need W^T packed as (In, Out).
    #[cfg(feature = "cuda")]
    packed_weights_t: CudaSlice<u8>,

    n: usize, // Out Features
    k: usize, // In Features
    scale: f32,
}

impl BitLinearOp {
    /// Create a new BitLinearOp layer.
    /// Performs quantization (packing) immediately for both Forward and Backward weights.
    pub fn new(weights: &Tensor, scale: f32) -> Result<Arc<Self>> {
        let (n, k) = weights.dims2()?;

        if k % 4 != 0 {
            candle_core::bail!("BitLinearCuda: In_features (k={}) must be divisible by 4", k);
        }
        if n % 4 != 0 {
            candle_core::bail!("BitLinearCuda: Out_features (n={}) must be divisible by 4 for backward packing", n);
        }

        #[cfg(feature = "cuda")]
        {
            let device = match weights.device() {
                Device::Cuda(dev) => dev.clone(),
                _ => candle_core::bail!("BitLinearCuda: Weights must be on CUDA device"),
            };

            // 1. Pack Forward Weights (N, K) -> (N, K/4)
            let (storage, layout) = weights.storage_and_layout();
            if !layout.is_contiguous() {
                 candle_core::bail!("BitLinearCuda: Weights must be contiguous for packing");
            }

            let w_ptr = match &*storage {
                Storage::Cuda(s) => match &s.slice {
                    CudaStorageSlice::F32(slice) => *slice.device_ptr(),
                    _ => candle_core::bail!("BitLinearCuda: Weights must be F32"),
                },
                _ => candle_core::bail!("BitLinearCuda: Storage mismatch (expected CUDA)"),
            };

            let pack_func = device.get_or_load_func("pack_32_2", PTX_SRC)?;

            // Alloc Forward
            let packed_size = n * k / 4;
            let mut packed_weights = unsafe { device.alloc::<u8>(packed_size) }.map_err(candle_core::Error::wrap)?;

            let cfg = LaunchConfig::for_num_elems(packed_size as u32);
            let params = (w_ptr, &mut packed_weights, packed_size as i32);
            unsafe { pack_func.clone().launch(cfg, params) }.map_err(candle_core::Error::wrap)?;

            // 2. Pack Backward Weights (W^T)
            // Need transpose first: (N, K) -> (K, N)
            // We use candle to transpose, then get pointer.
            // Note: We need a temporary tensor for W^T.
            // Transpose is cheap metadata op, but we need contiguous data for our pack kernel.
            // So: weights.t() -> contiguous() -> pack.
            let w_t = weights.t()?.contiguous()?;
            let (storage_t, _) = w_t.storage_and_layout();
            let w_t_ptr = match &*storage_t {
                Storage::Cuda(s) => match &s.slice {
                    CudaStorageSlice::F32(slice) => *slice.device_ptr(),
                     _ => unreachable!(),
                },
                 _ => unreachable!(),
            };

            let packed_size_t = k * n / 4; // Should be same size, just diff shape logic? Yes.
            let mut packed_weights_t = unsafe { device.alloc::<u8>(packed_size_t) }.map_err(candle_core::Error::wrap)?;

            let cfg_t = LaunchConfig::for_num_elems(packed_size_t as u32);
            let params_t = (w_t_ptr, &mut packed_weights_t, packed_size_t as i32);
            unsafe { pack_func.launch(cfg_t, params_t) }.map_err(candle_core::Error::wrap)?;

            Ok(Arc::new(Self {
                device,
                packed_weights,
                packed_weights_t,
                n,
                k,
                scale,
            }))
        }

        #[cfg(not(feature = "cuda"))]
        {
            candle_core::bail!("BitLinearCuda is not available (compile with --features cuda)")
        }
    }

    // Helper for Raw Launch (Inference without Autograd)
    pub fn forward_raw(&self, x: &Tensor, scale: f32) -> Result<Tensor> {
         #[cfg(feature = "cuda")]
        {
            // Similar to old forward, but using self
            let (b, k_in) = x.dims2()?;
            if k_in != self.k {
                candle_core::bail!("BitLinearCuda: Shape mismatch x:{{{},{}}} vs w:{{{},{}}}", b, k_in, self.n, self.k);
            }

            let x_ptr = self.get_ptr(x)?;

            // Output Allocation
            let output_shape = Shape::from((b, self.n));
            let output = Tensor::zeros(&output_shape, DType::F32, &Device::Cuda(self.device.clone()))?;
            let y_ptr = self.get_ptr(&output)?;

            self.launch_gemv(x_ptr, &self.packed_weights, y_ptr, self.k, self.n, scale, b)?;

            Ok(output)
        }
        #[cfg(not(feature = "cuda"))]
        {
             candle_core::bail!("No Cuda")
        }
    }

    #[cfg(feature = "cuda")]
    fn get_ptr(&self, t: &Tensor) -> Result<u64> {
        let (storage, layout) = t.storage_and_layout();
        if !layout.is_contiguous() {
             candle_core::bail!("BitLinearCuda: Tensor must be contiguous");
        }
        match &*storage {
             Storage::Cuda(s) => match &s.slice {
                 CudaStorageSlice::F32(slice) => Ok(*slice.device_ptr()),
                 _ => candle_core::bail!("BitLinearCuda: Tensor must be F32"),
             },
             _ => candle_core::bail!("BitLinearCuda: Tensor must be on CUDA"),
        }
    }

    #[cfg(feature = "cuda")]
    fn launch_gemv(
        &self,
        x_ptr: u64,
        w_packed: &CudaSlice<u8>,
        y_ptr: u64,
        k: usize,
        n: usize,
        scale: f32,
        batch: usize
    ) -> Result<()> {
        let gemv_func = self.device.get_or_load_func("bitnet_gemv_fused", PTX_SRC)?;
        let grid_dim = (n as u32, 1, 1);
        let block_dim = (256, 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: 0 };

        for i in 0..batch {
             let x_offset = i * k;
             let y_offset = i * n;
             let cur_x_ptr = x_ptr + (x_offset * 4) as u64;
             let cur_y_ptr = y_ptr + (y_offset * 4) as u64;

             let params = (cur_x_ptr, w_packed, cur_y_ptr, k as i32, n as i32, scale);
             unsafe { gemv_func.clone().launch(cfg, params) }.map_err(candle_core::Error::wrap)?;
        }
        Ok(())
    }
}

// Implement CustomOp2 for Autograd
// Arguments: (x, weights_dummy)
// We need 'weights_dummy' to be in the graph so dL/dw is requested.
#[cfg(feature = "cuda")]
impl CustomOp2 for BitLinearOp {
    fn name(&self) -> &'static str {
        "bit-linear-op"
    }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &candle_core::Layout, _: &candle_core::CpuStorage, _: &candle_core::Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("BitLinearOp is CUDA only")
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage, // x
        l1: &candle_core::Layout,
        _s2: &CudaStorage, // weights (unused, we use packed)
        _l2: &candle_core::Layout,
    ) -> Result<(CudaStorage, Shape)> {
        // s1 is Inputs.
        // We know inputs are F32.
        use candle_core::backend::BackendStorage;

        let input_ptr = match &s1.slice {
            CudaStorageSlice::F32(slice) => *slice.device_ptr(),
            _ => candle_core::bail!("BitLinearOp: Inputs must be F32"),
        };

        let (b, k) = l1.shape().dims2()?;
        if k != self.k {
             candle_core::bail!("BitLinearOp: Input dim mismatch");
        }

        // Alloc Output
        let dev = s1.device.clone();
        let out_shape = Shape::from((b, self.n));
        let out_elem = out_shape.elem_count();
        let slice = unsafe { dev.alloc::<f32>(out_elem) }.map_err(candle_core::Error::wrap)?;
        let out_ptr = *slice.device_ptr();

        // Launch Forward Kernel
        // Note: we need 'scale'. self.scale is 1.0 (BitLinearCuda::new default).
        // Wait, 'new' logic in bit_linear.rs passes scaled weights.
        // So scale is effectively 1.0 here if weights were pre-scaled.
        // Yes, verify usage.
        self.launch_gemv(input_ptr, &self.packed_weights, out_ptr, self.k, self.n, self.scale, b)?;

        // Wrap in CudaStorage
        let dst = CudaStorage {
            device: dev,
            slice: CudaStorageSlice::F32(slice),
        };
        Ok((dst, out_shape))
    }

    fn bwd(
        &self,
        arg1: &Tensor, // x (Inputs)
        _arg2: &Tensor, // weights (Float)
        _res: &Tensor, // Output (y) - needed for trait, unused here
        grad: &Tensor, // dL/dy (Grad Output)
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // 1. dL/dx = dL/dy @ W^T
        // dL/dy: (Batch, Out)
        // W^T: (Out, In) (conceptually)
        // Operation: GEMM or our GEMV if Batched?
        // bitnet_gemv_fused does (In, Out) x (Out) -> (In) ??
        // No, it initializes kernel as if (N, K) x (K) -> (N).
        // Here we want: (Batch, Out) x (Out, In) -> (Batch, In).
        // If Batch=1: (1, Out) @ (Out, In) -> (1, In).
        // This is exactly GEMV if we treat W^T as the matrix (Out rows, In cols).
        // My 'packed_weights_t' is (In rows, Out cols) if packed from W^T (K, N).
        // Wait.
        // W: (N, K). W^T: (K, N).
        // packed_weights_t = pack(W^T).
        // So packed_weights_t represents a matrix of shape (K, N).
        // GEMV(matrix, vec) signature:
        // Matrix: (Rows, Cols). Vec: (Cols). Result: (Rows).
        // If we pass packed_weights_t (K, N) and grad (Batch, Out)...
        // If Batch=1, grad is (1, Out) or (Out).
        // Vector is `grad`. Cols = Out. Matrix Rows = K.
        // Result is (K) -> (1, In) input grad.
        // This works!
        // We just need to check if batch loop works.
        // Our 'launch_gemv' loop handles batching by offsetting pointers.

        // dL/dx Calculation
        let d_dx = self.forward_backward_probe(grad)?;

        // 2. dL/dw = x^T @ dL/dy
        // x: (Batch, In). dL/dy: (Batch, Out).
        // Target: (Out, In)? Or (N, K).
        // candle::matmul(x.t(), grad)?
        // shapes: (In, Batch) @ (Batch, Out) -> (In, Out) -> which is (K, N) -> W^T?
        // W is (N, K).
        // We need dL/dw of shape (N, K).
        // dL/dw = (dL/dy)^T @ x.
        // (Out, Batch) @ (Batch, In) -> (Out, In) -> (N, K).
        // Yes.
        let d_dw = grad.t()?.matmul(arg1)?;

        Ok((Some(d_dx), Some(d_dw)))
    }
}

#[cfg(feature = "cuda")]
impl BitLinearOp {
    // Helper for bwd to keep bwd func clean
    fn forward_backward_probe(&self, grad: &Tensor) -> Result<Tensor> {
        let (b, _) = grad.dims2()?;
        // We want result (Batch, In) -> (Batch, K).
        let out_shape = Shape::from((b, self.k));
        let output = Tensor::zeros(&out_shape, DType::F32, &Device::Cuda(self.device.clone()))?;

        // Launch GEMV using Transposed Weights
        let grad_ptr = self.get_ptr(grad)?;
        let out_ptr = self.get_ptr(&output)?;

        // Logical Matrix: W^T is (K, N).
        // 'n' -> K, 'k' -> N from kernel perspective.
        // Kernel args: k, n.
        // Kernel logic loops 'k' items to produce 'n' output items.
        // Here we loop 'N' items (cols of W^T) to produce 'K' output items (rows of W^T).
        // So pass k=N, n=K.
        self.launch_gemv(grad_ptr, &self.packed_weights_t, out_ptr, self.n, self.k, self.scale, b)?;

        Ok(output)
    }
}

// Wrapper to bridge non-Cuda compilation
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct BitLinearOp;

#[cfg(not(feature = "cuda"))]
impl BitLinearOp {
    pub fn new(_weights: &Tensor, _scale: f32) -> Result<Arc<Self>> {
         candle_core::bail!("No CUDA")
    }
    pub fn forward_raw(&self, _x: &Tensor, _scale: f32) -> Result<Tensor> {
        candle_core::bail!("No CUDA")
    }
}
