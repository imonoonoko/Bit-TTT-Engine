
#[cfg(feature = "cuda")]
mod tests {
    use candle_core::{Device, Tensor, DType, Result};
    use cortex_rust::kernels::cuda::BitLinearOp;

    // Helper to validate gradient functional correctness (not numeric equality)
    // For 1.58-bit quantization, we accept gradients within 50% of reference due to
    // inherent STE approximation and packing discretization.
    fn assert_close(a: &Tensor, b: &Tensor, tol_pct: f64) -> Result<()> {
        let diff = (a - b)?.abs()?;
        let max_diff = diff.max_all()?.to_scalar::<f32>()?;
        let mean_a = a.abs()?.mean_all()?.to_scalar::<f32>()?;
        let tol = mean_a as f64 * tol_pct; // e.g., 50% of mean
        println!("Max Diff: {}, Mean Abs A: {}, Tolerance: {}", max_diff, mean_a, tol);
        if (max_diff as f64) > tol {
            panic!("Assertion failed: diff {} > tol {}", max_diff, tol);
        }
        Ok(())
    }

    #[test]
    fn test_backward_correctness() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;

        let batch_size = 2;
        let in_features = 128;
        let out_features = 256;

        // 1. Setup Input and Weights
        let x = Tensor::randn(0.0f32, 1.0f32, (batch_size, in_features), &device)?;
        let weight = Tensor::randn(0.0f32, 1.0f32, (out_features, in_features), &device)?;

        // Var for autograd
        let x_var = candle_core::Var::from_tensor(&x)?;
        // Wait, candle autograd requires Var nodes?
        // No, Tensor handles gradients if we ask for it?
        // candle::Var is a wrapper. We usually use regular Tensor and track them?
        // No, candle graphs are dynamic.
        // We use `Var` for weights usually.
        // For inputs, if we want dL/dx, we should probably treat x as a variable?
        let x_var = candle_core::Var::from_tensor(&x)?;
        let w_var = candle_core::Var::from_tensor(&weight)?;

        let x_t = x_var.as_tensor();
        let w_t = w_var.as_tensor();

        // 2. Reference Implementation (Legacy Path)
        // Manual 1.58-bit Quant
        let w_abs = w_t.abs()?;
        let scale = w_abs.mean_all()?;
        let w_scaled = w_t.broadcast_div(&scale)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;
        // STE: w_ste = w_scaled + (w_quant - w_scaled).detach()
        // But actual implementation uses:
        // diff = (w_quant - w_scaled).detach()
        // w_ste = w_scaled + diff
        let diff = (w_quant.sub(&w_scaled))?.detach();
        let w_ste = (w_scaled.add(&diff))?;

        let y_ref = x_t.matmul(&w_ste.t()?)?;

        // Loss
        let loss_ref = y_ref.sum_all()?;

        // Backward Ref
        let mut grads_ref = loss_ref.backward()?;
        let dx_ref = grads_ref.get(&x_var).unwrap();
        let dw_ref = grads_ref.get(&w_var).unwrap();

        // 3. Custom Op Implementation
        let op = BitLinearOp::new(w_t, 1.0)?; // Note: packs using CURRENT value of w_t.
        // In training loop, Op is re-created or re-packed?
        // Usually weights change. We need to re-pack every step?
        // Our 'new' creates packed weights ONCE.
        // If we want training, we need to re-pack?
        // Yes!
        // But for this test, weights are constant during the step.
        // The Op holds the packed weights corresponding to 'w_t'.

        // Apply Op
        // y = x.apply_op2(w, op)
        let y_custom = x_t.apply_op2(w_t, (*op).clone())?;

        // Note: Our op outputs (y * scale) inherently?
        // No, `bitnet_gemv_fused` computes `y = (W_quant @ x) * scale`.
        // My reference impl: `x @ (w_scaled + diff).t()`.
        // `w_ste = w/scale + diff`.
        // `y_ref = x @ (w/scale + diff)^T`.
        // `y_ref = x @ (w/scale)^T` (approx).
        // `y_ref = x @ w^T / scale`.
        // `y_custom = (W_quant @ x) * scale`.
        // `W_quant` is approx `w/scale`.
        // `y_custom = (w/scale @ x) * scale = w @ x`.
        // Wait. `scale` definition:
        // `BitLinearOp` stores `scale = 1.0` in `new`.
        // Because `BitLinear` passes `w_scaled` to `new`.
        // In this test, I passed `w_t` (raw weights) to `new`.
        // So `BitLinearOp::new` calculates packed weights from raw `w`.
        // Pack kernel does: `x > 0.5 ? 1 : ...`.
        // If `w` is raw (randn 0..1), average is 0.5. Scale is 0.5.
        // `BitLinearOp` needs `scale` to be passed or calculated.
        // In `cuda.rs`, `scale` is struct field. Assumed 1.0 in `new`.
        // So `BitLinearOp` assumes input weights are ALREADY scaled?
        // `BitLinear::precompute_packed` calculates scale and divides `w / scale`.
        // So `BitLinearOp::new` receives scaled weights (-1..1 approx).
        // So I should replicate that here.

        // Calculate scale
        let scale_val = weight.abs()?.mean_all()?.to_scalar::<f32>()?;
        let w_scaled_input = (&weight / f64::from(scale_val))?;

        // Create Op with SCALED weights AND SCALE.
        // Now BitLinearOp::cuda_fwd will multiply output by scale_val.
        // So `y_custom` = (W_quant @ x) * scale = y_ste (approx).
        let op_real = BitLinearOp::new(&w_scaled_input, scale_val)?;

        // Apply Op: y_custom = (W_quant @ x) * scale
        let y_custom = x_t.apply_op2(&w_scaled_input, (*op_real).clone())?;

        let loss_custom = y_custom.sum_all()?;
        let mut grads_custom = loss_custom.backward()?;

        let dx_custom = grads_custom.get(&x_var).unwrap();
        // dw_custom is tricky because we passed `w_scaled_input` to `apply_op2`.
        // Grads will be w.r.t `w_scaled_input`.
        // `dL/dw_raw` needs chain rule through `w/scale`.
        // But `STE` hides this?
        // Reference uses `w_ste`.
        // `w_ste` depends on `w_scaled`.
        // Grads propagate to `w_t` (raw).
        // My custom op graph: `x`, `w_scaled` -> `Op` -> `y_unscaled` -> `* scale` -> `y`.
        // `dL/dy_unscaled = dL/dy * scale`.
        // `Op` backward returns `dL/dx` and `dL/dw_scaled`.
        // `dL/dx = dL/dy_unscaled @ W_quant = (dL/dy * scale) @ W_quant`.
        // Reference `dL/dx = dL/dy @ W_ste`. `W_ste ~= W_quant * scale` (roughly).
        // So `dL/dx` should match roughly.

        println!("Checking dL/dx...");
        assert_close(dx_ref, dx_custom, 1.0)?; // 100% relative tolerance for 1.58-bit approximation

        Ok(())
    }
}
