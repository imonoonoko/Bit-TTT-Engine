#![no_main]
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cortex_rust::core_engine::TTTLayer;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }

    // 1. Parse Parameters
    // hidden_dim: 4 to 64 (must be divisible by 4 for simple d_small)
    let raw_dim = data[0] as usize;
    let hidden_dim = 4 + (raw_dim % 16) * 4; // 4, 8, ..., 64
    let d_small = hidden_dim / 4;

    // inner_lr
    let inner_lr = f32::from_le_bytes([data[1], data[2], data[3], data[4]]) as f64;
    // Check for NaN or Inf? Logic should handle it, but TTTLayer might produce NaNs.
    // We are looking for Panics, not NaN correctness (unless unwrap panics on NaN).

    // Input data
    let input_bytes = &data[5..];
    let device = Device::Cpu;

    // 2. Mock Weights
    // We create a mocked VarBuilder with Random/Zero weights.
    // For fuzzing "params", we might want to read weights from data too, but let's stick to zeros/ones for stability,
    // and rely on input `x` to trigger issues.
    let mut tensors = HashMap::new();

    // down.weight: [d_small, hidden_dim]
    let down = Tensor::zeros((d_small, hidden_dim), DType::F32, &device).unwrap();
    tensors.insert("down.weight".to_string(), down);

    // up.weight: [hidden_dim, d_small]
    let up = Tensor::zeros((hidden_dim, d_small), DType::F32, &device).unwrap();
    tensors.insert("up.weight".to_string(), up);

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

    // 3. Load Model
    let layer = match TTTLayer::load(hidden_dim, inner_lr, vb) {
        Ok(l) => l,
        Err(_) => return, // Failed to load (should not happen with correct shapes)
    };

    // 4. Prepare Input
    // We need a tensor of shape [Batch, Hidden]
    // Let's assume Batch=1 for simplicity, or strict shape.
    let batch = 1;
    let input_len = batch * hidden_dim * 4; // 4 bytes per f32
    if input_bytes.len() < input_len {
        return;
    }

    let input_data: Vec<f32> = input_bytes[..input_len]
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let x = Tensor::from_vec(input_data, (batch, hidden_dim), &device).unwrap();

    // 5. Initial State [Batch, d_small, d_small]
    let w_state = Tensor::zeros((batch, d_small, d_small), DType::F32, &device).unwrap();

    // 6. Run Forward
    let _ = layer.forward_update(&w_state, &x);
});
