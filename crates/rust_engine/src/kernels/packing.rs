use candle_core::{Device, Result, Tensor};

/// Epsilon for numerical stability during Scale calculation
const EPSILON: f32 = 1e-6;

/// 1.58-bit Packed Tensor.
/// Stores weights in a compressed 2-bit format (4 weights per u8).
///
/// Mapping:
/// - 00 -> 0.0
/// - 01 -> 1.0
/// - 10 -> -1.0
/// - 11 -> Unused/Padding
#[derive(Debug, Clone)]
pub struct PackedTensor {
    pub data: Tensor, // [out_dim, in_dim/4] (u8)
    pub scale: f32,
    pub shape: candle_core::Shape, // Original shape [out_dim, in_dim]
    pub num_elem: usize,
    pub device: Device,
}

impl PackedTensor {
    /// Create new PackedTensor from raw bytes
    pub fn new(
        data: Vec<u8>,
        shape: candle_core::Shape,
        scale: f32,
        device: &Device,
    ) -> Result<Self> {
        let num_elem = shape.elem_count();
        // Calculate packed shape: [weights_len / 4] (Approx, strictly 1D for now or assume flattened)
        // For Linear layer: [Out, In] -> [Out, In/4]
        // But here we just treat as flat buffer for simplicity in storage,
        // reshape happens in kernel usage if needed.
        let capacity = num_elem.div_ceil(4);

        let tensor = Tensor::from_vec(data, (capacity,), device)?;

        Ok(Self {
            data: tensor,
            scale,
            shape: shape.clone(),
            num_elem,
            device: device.clone(),
        })
    }

    /// Pack a float tensor (containing -1.0, 0.0, 1.0 or raw weights) into PackedTensor
    pub fn pack(tensor: &Tensor) -> Result<Self> {
        let device = tensor.device();
        let shape = tensor.shape().clone();
        let num_elem = shape.elem_count();

        // 1. Calculate Scale: Mean of absolute values
        let scale_t = tensor.abs()?.mean_all()?;
        let scale = scale_t.to_scalar::<f32>()? + EPSILON;

        // 2. Quantize: W_scaled = round(clamp(W / Scale, -1, 1))
        // This maps values to {-1, 0, 1}
        let w_scaled = (tensor / scale as f64)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?.to_dtype(candle_core::DType::F32)?;

        // 3. Flatten and Pack
        let flat = w_quant.flatten_all()?;
        let vec = flat.to_vec1::<f32>()?; // CPU copy for packing logic

        let capacity = num_elem.div_ceil(4);
        let mut packed_data = Vec::with_capacity(capacity);

        for chunk in vec.chunks(4) {
            let mut byte: u8 = 0;
            for (i, &val) in chunk.iter().enumerate() {
                // val is expected to be -1.0, 0.0, or 1.0 (float)
                // We map this to 2-bit code:
                // > 0.5  => 1 (01)
                // < -0.5 => -1 (10)
                // else   => 0 (00)
                let code: u8 = if val > 0.5 {
                    1 // 01
                } else if val < -0.5 {
                    2 // 10
                } else {
                    0 // 00
                };
                byte |= code << (i * 2);
            }
            packed_data.push(byte);
        }

        // Return PackedTensor on appropriate device
        let data_tensor =
            Tensor::from_vec(packed_data, (capacity,), &Device::Cpu)?.to_device(device)?;

        Ok(Self {
            data: data_tensor,
            scale,
            shape,
            num_elem,
            device: device.clone(),
        })
    }

    /// Unpack back to f32 tensor (for verification/fallback)
    pub fn unpack(&self, device: &Device) -> Result<Tensor> {
        // Pull data to CPU to unpack
        let data_vec = self.data.to_vec1::<u8>()?;
        let mut floats = Vec::with_capacity(self.num_elem);

        for &byte in &data_vec {
            for i in 0..4 {
                if floats.len() >= self.num_elem {
                    break;
                }

                let code = (byte >> (i * 2)) & 0b11;
                let val: f32 = match code {
                    1 => 1.0,
                    2 => -1.0,
                    _ => 0.0,
                };
                floats.push(val);
            }
        }

        // Restore scale
        // Restore scale
        let t = Tensor::from_vec(floats, self.shape.clone(), device)?;
        (t * self.scale as f64)?.to_dtype(candle_core::DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to compare tensors with tolerance
    fn assert_tensor_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "Tensor lengths mismatch");
        for (i, (v1, v2)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < tol,
                "Mismatch at index {}: {} vs {} (tol {})",
                i,
                v1,
                v2,
                tol
            );
        }
    }

    #[test]
    fn test_packing_cycle_dense() -> Result<()> {
        // Case 1: Dense {-1, 1}
        // Mean(|W|) = 1.0
        // Expected Scale ~ 1.0
        let input_data = vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let tensor = Tensor::new(&input_data[..], &Device::Cpu)?;

        let packed = PackedTensor::pack(&tensor)?;

        // Scale ~1.0 + EPSILON
        assert!((packed.scale - 1.0).abs() < 1e-3);

        let unpacked = packed.unpack(&Device::Cpu)?;
        let output_data = unpacked.to_vec1::<f32>()?;

        assert_tensor_approx_eq(&input_data, &output_data, 1e-4);
        Ok(())
    }

    #[test]
    fn test_packing_cycle_sparse() -> Result<()> {
        // Case 2: Sparse {-1, 0, 1}
        let input_data: Vec<f32> = vec![1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0];
        let tensor = Tensor::new(&input_data[..], &Device::Cpu)?;

        let packed = PackedTensor::pack(&tensor)?;

        // Scale = Sum(|x|) / N = 4 / 8 = 0.5
        assert!((packed.scale - 0.5).abs() < 1e-3);

        let unpacked = packed.unpack(&Device::Cpu)?;
        let output_data = unpacked.to_vec1::<f32>()?;

        // Expected output: Input * (Scale correction?)
        // bitnet: W ~= Scale * Q(W)
        // Here Q(W) will be {-1, 0, 1}.
        // So Output = Scale * {-1, 0, 1} = 0.5 * {-1, 0, 1} = {-0.5, 0, 0.5}.
        // The original was {1, 0, -1}.
        // We lost magnitude information because the distribution was sparse?
        // Actually, BitNet assumes weights are Gaussian-ish or trained to be {-1, 0, 1} * alpha.
        // If we supply raw {-1, 0, 1}, we interpret them as weights.

        let expected_data = vec![0.5, -0.5, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0];
        assert_tensor_approx_eq(&output_data, &expected_data, 1e-4);

        Ok(())
    }

    #[test]
    fn test_packing_manual() -> Result<()> {
        // Low-level bit verification
        // 1.0 -> 01
        // -1.0 -> 10
        // 0.0 -> 00
        // 1.0 -> 01
        // Byte: 01 00 10 01 (Big Endian visual) -> Little Endian construct:
        // i=0 (1.0) -> 01
        // i=1 (-1.0) -> 10 << 2
        // i=2 (0.0) -> 00 << 4
        // i=3 (1.0) -> 01 << 6
        // 01 | 1000 | 0000 | 01000000 = 1 + 8 + 0 + 64 = 73

        let data = vec![73u8];
        let shape = candle_core::Shape::from((4,));
        let scale = 1.0;
        let device = Device::Cpu;

        let packed = PackedTensor::new(data, shape, scale, &device)?;
        let unpacked = packed.unpack(&device)?;
        let output = unpacked.to_vec1::<f32>()?;

        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], -1.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 1.0);

        Ok(())
    }

    #[test]
    fn test_packing_padding() -> Result<()> {
        // 5 elements -> 2 bytes.
        // [1, 1, 1, 1, -1]
        // Byte 1: 1,1,1,1 -> 01,01,01,01 -> 01010101 = 85
        // Byte 2: -1 -> 10 -> 2
        // Scale 1.0 (Approx)

        let input_data = vec![1.0, 1.0, 1.0, 1.0, -1.0];
        let tensor = Tensor::new(&input_data[..], &Device::Cpu)?;

        let packed = PackedTensor::pack(&tensor)?;
        assert_eq!(packed.data.dims1()?, 2);

        let data = packed.data.to_vec1::<u8>()?;
        assert_eq!(data[0], 85);
        assert_eq!(data[1], 2);

        let unpacked = packed.unpack(&Device::Cpu)?;
        let output_data = unpacked.to_vec1::<f32>()?;

        // Expect near perfect reconstruction since mean is close to 1
        // Mean = 5/5 = 1.0
        assert_tensor_approx_eq(&input_data, &output_data, 1e-4);

        Ok(())
    }
}
