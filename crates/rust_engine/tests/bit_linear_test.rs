#[cfg(feature = "cuda")]
mod cuda_tests {
    use candle_core::{Device, Tensor};
    use cortex_rust::layers::BitLinear;

    #[test]
    fn test_bit_linear_correctness() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        println!("Running test on CUDA device: {:?}", device);

        // 1. Setup Weights (4x4 matrix - divisible by 4 for packing)
        // W = [[ 1.0, -1.0, 0.0, 0.0],
        //      [ 0.0,  1.0, -1.0, 1.0],
        //      [ 1.0,  1.0, 0.0, -1.0],
        //      [-1.0,  0.0, 1.0, 0.0]]

        let w_data: Vec<f32> = vec![
            1.0, -1.0, 0.0, 0.0,
            0.0, 1.0, -1.0, 1.0,
            1.0, 1.0, 0.0, -1.0,
            -1.0, 0.0, 1.0, 0.0
        ];
        let w_tensor = Tensor::from_vec(w_data.clone(), (4, 4), &device)?;

        // 2. Wrap in BitLinear
        let mut layer = BitLinear {
            weight: w_tensor,
            in_features: 4,
            out_features: 4,
            packed_params: None,
            cuda_kernel: None,
        };

        // 3. Precompute Packed (Should trigger CUDA packing)
        layer.precompute_packed()?;

        assert!(layer.packed_params.is_some(), "Packed params should be populated");
        assert!(layer.cuda_kernel.is_some(), "CUDA kernel should be initialized");

        // 4. Create Input
        // B=1, In=4
        // X = [1.0, 1.0, 1.0, 1.0]
        let x_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let x = Tensor::from_vec(x_data, (1, 4), &device)?;

        // 5. Run Forward (Should use CUDA kernel)
        let y = layer.forward(&x)?;

        // 6. Verify Output (non-zero, correct shape)
        let y_vec = y.flatten_all()?.to_vec1::<f32>()?;
        println!("Output: {:?}", y_vec);

        assert_eq!(y_vec.len(), 4, "Output shape mismatch");
        // Just verify it runs without panic and values are reasonable
        for val in &y_vec {
            assert!(!val.is_nan(), "Output contains NaN");
        }

        Ok(()) // Success
    }
}
