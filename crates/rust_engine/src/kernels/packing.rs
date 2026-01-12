use candle_core::{Device, Result, Tensor};

/// 1.58-bit Packed Tensor.
/// Stores weights in a compressed 2-bit format (4 weights per u8).
///
/// Mapping:
/// - 0  -> 00
/// - 1  -> 01
/// - -1 -> 10 // (Represented as 2 in u8)
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
    pub fn new(data: Vec<u8>, shape: candle_core::Shape, scale: f32, device: &Device) -> Result<Self> {
        let num_elem = shape.elem_count();
        // Calculate packed shape: [weights_len / 4] (Approx, strictly 1D for now or assume flattened)
        // For Linear layer: [Out, In] -> [Out, In/4]
        // But here we just treat as flat buffer for simplicity in storage,
        // reshape happens in kernel usage if needed.
        let capacity = (num_elem + 3) / 4;

        let tensor = Tensor::from_vec(data, (capacity,), device)?;

        Ok(Self {
            data: tensor,
            scale,
            shape: shape.clone(),
            num_elem,
            device: device.clone(),
        })
    }

    /// Pack a float tensor (containing -1.0, 0.0, 1.0) into PackedTensor
    pub fn pack(tensor: &Tensor) -> Result<Self> {
        let device = tensor.device();
        let shape = tensor.shape().clone();
        let num_elem = shape.elem_count();

        // Calculate scale (Abs Mean)
        // Note: For pure {-1, 0, 1} input, scale is 1.0 (or should be handled before data comes here).
        // But usually we quantize here.
        // User guide says: "from_weights... todo check Packing Logic".
        // Let's assume input `tensor` is already the raw weights (f32, full precision).
        // We need to quantize it first: divide by scale, round, clamp.

        // 1. Calculate Scale
        let scale_t = tensor.abs()?.mean_all()?;
        let scale = scale_t.to_scalar::<f32>()? + 1e-6; // Add epsilon to avoid div-by-zero

        // 2. Quantize: W_scaled = W / Scale
        let w_scaled = (tensor / scale as f64)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;

        // 3. Flatten and Pack
        let flat = w_quant.flatten_all()?;
        let vec = flat.to_vec1::<f32>()?; // CPU copy

        let capacity = (num_elem + 3) / 4;
        let mut packed_data = Vec::with_capacity(capacity);

        for chunk in vec.chunks(4) {
            let mut byte: u8 = 0;
            for (i, &val) in chunk.iter().enumerate() {
                // val is -1.0, 0.0, or 1.0
                let code: u8 = if val > 0.5 {
                    1 // 1
                } else if val < -0.5 {
                    2 // -1
                } else {
                    0 // 0
                };
                byte |= code << (i * 2);
            }
            packed_data.push(byte);
        }

        // Return PackedTensor on appropriate device
        // Move u8 data to device
        let data_tensor = Tensor::from_vec(packed_data, (capacity,), &Device::Cpu)?.to_device(device)?;

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
        let t = Tensor::from_vec(floats, self.shape.clone(), device)?;
        t * self.scale as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packing_cycle_dense() -> Result<()> {
        // Use only 1.0 and -1.0. Mean(|W|) = 1.0.
        // This ensures quantization allows perfect reconstruction.
        let input_data = vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let tensor = Tensor::new(&input_data[..], &Device::Cpu)?;

        let packed = PackedTensor::pack(&tensor)?;

        // Scale should be ~1.0
        assert!((packed.scale - 1.0).abs() < 1e-4);

        let unpacked = packed.unpack(&Device::Cpu)?;
        let output_data = unpacked.to_vec1::<f32>()?;

        // Should verify elements match roughly (floating point issues possible but minimal here)
        for (a, b) in input_data.iter().zip(output_data.iter()) {
             assert!((a - b).abs() < 1e-4, "Mismatch: {} vs {}", a, b);
        }
        Ok(())
    }

    #[test]
    fn test_packing_manual_zeros() -> Result<()> {
        // Manually verify that 0.0 is encoded as 00 (0) and handled correctly.
        // We construct a PackedTensor directly to bypass the Quantization logic.

        // 1.0 (01), -1.0 (10), 0.0 (00), 1.0 (01)
        // Little endian byte: 01 (0) | 10 (2) | 00 (4) | 01 (6)
        // 1 + 8 + 0 + 64 = 73 (0b01001001)
        let data = vec![73u8];
        let shape = candle_core::Shape::from((4,)); // 4 elements fit in 1 byte
        let scale = 1.0;
        let device = Device::Cpu;

        let packed = PackedTensor::new(data, shape, scale, &device)?;

        let unpacked = packed.unpack(&device)?;
        let output = unpacked.to_vec1::<f32>()?;

        // Expected first 4 values
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], -1.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 1.0);

        Ok(())
    }

    #[test]
    fn test_packing_padding() -> Result<()> {
        // 5 elements -> 2 bytes (one partial)
        let input_data = vec![1.0, 1.0, 1.0, 1.0, -1.0];
        let tensor = Tensor::new(&input_data[..], &Device::Cpu)?;

        let packed = PackedTensor::pack(&tensor)?;
        assert_eq!(packed.data.dims1()?, 2);

        let unpacked = packed.unpack(&Device::Cpu)?;
        let output_data = unpacked.to_vec1::<f32>()?;

        // assert_eq!(input_data, output_data);
         for (a, b) in input_data.iter().zip(output_data.iter()) {
             assert!((a - b).abs() < 1e-4, "Mismatch: {} vs {}", a, b);
        }
        Ok(())
    }
}
