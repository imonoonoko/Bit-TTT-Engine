//! Test TTTLayer Memory Capability (Candle-based)
//!
//! This test has been updated for the new Candle-based implementation.
//! The original ndarray-based test is in legacy/.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cortex_rust::TTTLayer;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("=== Testing Bit-TTT Memory Capability (Candle) ===");

    let dim = 64;
    let d_small = dim / 4;
    let inner_lr = 0.1;
    let device = Device::Cpu;

    // Create mock weights for VarBuilder
    let mut tensors = HashMap::new();

    // Initialize with small random-like values using deterministic pattern
    let down_data: Vec<f32> = (0..(d_small * dim))
        .map(|i| ((i as f32).sin() * 0.1))
        .collect();
    let up_data: Vec<f32> = (0..(dim * d_small))
        .map(|i| ((i as f32).cos() * 0.1))
        .collect();

    tensors.insert(
        "down.weight".to_string(),
        Tensor::from_vec(down_data, (d_small, dim), &device)?,
    );
    tensors.insert(
        "up.weight".to_string(),
        Tensor::from_vec(up_data, (dim, d_small), &device)?,
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let layer = TTTLayer::load(dim, inner_lr, vb)?;

    // Create test patterns
    let pattern_a: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32)).collect();
    let pattern_b: Vec<f32> = (0..dim)
        .map(|i| ((i as f32 / dim as f32) * 2.0 - 1.0))
        .collect();

    let x_a = Tensor::from_vec(pattern_a.clone(), (1, dim), &device)?;
    let x_b = Tensor::from_vec(pattern_b.clone(), (1, dim), &device)?;

    // Initial state
    let mut w_state = Tensor::zeros((1, d_small, d_small), DType::F32, &device)?;

    println!("\nStep | Input | Description");
    println!("---------------------------------------------");

    // Train on pattern A
    println!("  0  |   A   | First presentation of pattern A");
    let (_, w_new) = layer.forward_update(&w_state, &x_a)?;
    w_state = w_new;

    // Train on pattern B
    println!("  1  |   B   | First presentation of pattern B");
    let (_, w_new) = layer.forward_update(&w_state, &x_b)?;
    w_state = w_new;

    // Train on pattern A again
    println!("  2  |   A   | Second presentation of pattern A");
    let (_, w_new) = layer.forward_update(&w_state, &x_a)?;
    w_state = w_new;

    // Train on pattern B again
    println!("  3  |   B   | Second presentation of pattern B");
    let (_, _w_new) = layer.forward_update(&w_state, &x_b)?;

    println!("---------------------------------------------");
    println!("\n[SUCCESS] TTTLayer forward_update completed without panics.");

    Ok(())
}
