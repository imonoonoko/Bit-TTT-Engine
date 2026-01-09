use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // 1. Setup
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);

    let b_size = 4;
    let d_dim = 128; // Hidden Dim

    // Sweep T
    let t_values = vec![1024, 4096, 16384, 65536];

    for &t_len in &t_values {
        println!("\n========================================");
        println!(
            "Benchmarking T = {} | Shape: ({}, {}, {})",
            t_len, b_size, t_len, d_dim
        );

        // Create Input: (B, T, D)
        // Note: For very large T, this might OOM on GPU.
        let input_res = Tensor::randn(0.0f32, 1.0f32, (b_size, t_len, d_dim), &device);
        if let Err(e) = input_res {
            println!("Skipping T={} due to allocation: {}", t_len, e);
            continue;
        }
        let input = input_res.unwrap();

        // --- Test 1: Sequential (Loop) ---
        // Simulating the current RNN-like behavior
        {
            print!("--- [1] Sequential Loop (RNN Style) --- ");
            // Force sync before start
            let _ = input.to_vec3::<f32>()?;

            let start = Instant::now();

            // Split along time dimension
            let chunks: Vec<Tensor> = input.chunk(t_len, 1)?; // (B, 1, D) x T

            let mut accum = Tensor::zeros((b_size, 1, d_dim), DType::F32, &device)?;

            for x in chunks.iter() {
                accum = (accum + x)?;
            }

            // Force sync
            let _res = accum.to_vec3::<f32>()?;

            let duration = start.elapsed();
            println!("Time: {:.2?}", duration);
        }

        // --- Test 2: Parallel (Candle native cumsum) ---
        // Simulating Parallel Scan
        {
            print!("--- [2] Parallel Scan (Candle native) --- ");
            // Force sync before start
            let _ = input.to_vec3::<f32>()?;

            let start = Instant::now();

            // cumsum over dimension 1 (Time)
            // Note: candle has cumulative_sum
            let scan = input.cumsum(1)?; // Result: (B, T, D)

            // Take the last time step to match sequential output (conceptually)
            let _final = scan.narrow(1, t_len - 1, 1)?;

            // Force sync
            let _res = _final.to_vec3::<f32>()?;

            let duration = start.elapsed();
            println!("Time: {:.2?}", duration);
        }
    }

    println!("\nNote: Ideally, Parallel should be significantly faster on GPU for large T.");

    Ok(())
}
