use candle_core::{Result, Tensor, Var};

#[derive(Debug, Clone)]
pub struct ParamsScheduleFree {
    pub lr: f64,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub beta: f64, // Interpolation parameter (usually 0.9 or 0.99)
}

impl Default for ParamsScheduleFree {
    fn default() -> Self {
        Self {
            lr: 0.0025, // SFO usually recommends higher LR
            warmup_steps: 0,
            weight_decay: 0.0,
            beta: 0.9,
        }
    }
}

/// Schedule-Free Optimizer (SGD variant)
/// Tracks 'z' (optimization sequence) and 'c' (averaged sequence aka x).
/// The model parameters 'p' track 'y' (interpolation) during training, and 'c' during eval.
pub struct ScheduleFreeOptimizer {
    vars: Vec<Var>,
    z: Vec<Tensor>, // Optimization sequence
    step: usize,
    params: ParamsScheduleFree,
}

impl ScheduleFreeOptimizer {
    pub fn new(vars: Vec<Var>, params: ParamsScheduleFree) -> Result<Self> {
        let mut z = Vec::with_capacity(vars.len());
        for var in &vars {
            // z initializes to current weights
            z.push(var.as_tensor().copy()?);
        }

        Ok(Self { vars, z, step: 0, params })
    }

    /// Prepare parameters for training (Forward Pass uses y)
    /// y = (1 - beta) * z + beta * c
    /// Note: In this simplified implementation, we assume 'vars' currently hold 'c' (the average).
    /// We compute y and move vars to y.
    pub fn train(&self) -> Result<()> {
        let b = self.params.beta;
        let one_minus_b = 1.0 - b;

        for (i, var) in self.vars.iter().enumerate() {
            let z_i = &self.z[i];
            let c_i = var.as_tensor(); // Current vars hold 'c' (average)

            // y = (1-b)z + b*c
            let y_i = ((z_i * one_minus_b)? + (c_i * b)?)?;
            var.set(&y_i)?;
        }
        Ok(())
    }

    /// Restore parameters to evaluation mode (Forward Pass uses c)
    /// This should be called before eval/save, or at end of training step if we want strictness.
    /// However, to save compute, we only call this when switching modes.
    /// But wait, `step()` needs 'c' to update it.
    /// Does `step()` assume vars hold 'y' or 'c'?
    /// Usually `step()` is called after `backward()`. Params hold 'y'.
    /// We need to RECOVER 'c' from 'y' and 'z'?
    /// y = (1-b)z + bc  => c = (y - (1-b)z) / b
    /// This introduces numerical error.
    ///
    /// improved design:
    /// 'vars' ALWAYS hold 'y' (training value) during training loop.
    /// We store 'c' explicitly? Then we are back to 3 buffers (Params(y), Z, C).
    /// To be "0 memory", we must use the fact that params hold one, and we store the other.
    ///
    /// R-SF approach:
    /// Store 'z'.
    /// 'vars' hold 'x' (or 'c').
    /// Just before forward: vars = y.
    /// Just after backward: vars = x (restore), then update x and z.
    /// This creates "swap overhead" but saves memory.
    ///
    /// Let's stick to the "Swap" approach.
    /// `train_pre_step()`: vars = y.
    /// `train_post_step/step()`: vars = c (restore), then update.
    pub fn eval(&self) -> Result<()> {
        // If we are in 'y' state, we need to revert to 'c'.
        // But tracking state is hard.
        // We will enforce a contract: step() leaves vars in 'c' state.
        // pre_step() puts them in 'y' state.
        // So simply doing nothing ensures 'c' state if loop is correct.
        Ok(())
    }

    /// Switch to Training Mode (y state)
    pub fn pre_step(&self) -> Result<()> {
        let b = self.params.beta;
        let one_minus_b = 1.0 - b;

        for (i, var) in self.vars.iter().enumerate() {
            let z_i = &self.z[i];
            let c_i = var.as_tensor();

            // y = (1-b)z + b*c
            // Note: We use in-place set, so we lose 'c'.
            // Wait, if we overwrite 'c' with 'y', we can't restore 'c' easily without error.
            // UNLESS y = z + b*(c-z)
            // c - z = ...
            //
            // Alternative: The simple 2-buffer SFO stores 'z' and 'x'.
            // It calculates 'y' into a temporary, or sets model params to 'y'.
            // If model params are 'x', then we set them to 'y'.
            // Then after backward, we MUST set them back to 'x' (updated x).

            // y = (1-b)z + b*x
            let y_i = ((z_i * one_minus_b)? + (c_i * b)?)?;
            var.set(&y_i)?;
        }
        Ok(())
    }

    pub fn step(&mut self, grads: &[Tensor]) -> Result<()> {
        self.step += 1;
        let k = self.step as f64;

        // Effective LR with Decay (Simple Schedule-Free doesn't strictly need decay but helpful)
        let lr = self.params.lr;

        let b = self.params.beta;

        // At start of step(), vars hold 'y' (if pre_step was called).
        // We need 'x_old' to update 'x_new'.
        // But we overwrote 'x_old' with 'y'.
        // Recovery: x_old = (y - (1-b)z) / b.
        // This is safe if b is not tiny. Default 0.9.

        for (i, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(i) {
                let z_i = &self.z[i];
                let y_i = var.as_tensor(); // This is 'y'

                // 1. Recover x_old
                // x = (y - (1-b)z) / b
                let one_minus_b = 1.0 - b;
                let term = (z_i * one_minus_b)?;
                let diff = (y_i - term)?;
                let x_old = (diff / b)?;

                // 2. Update z (Optimization Step)
                // z_new = z_old - lr * grad
                // (Apply weight decay if needed)
                let z_new = (z_i - (grad * lr)?)?;

                // 3. Update x (Averaging)
                // x_new = (1 - 1/k+1)x_old + (1/k+1)z_new
                let k_inv = 1.0 / (k + 1.0);
                let one_minus_k = 1.0 - k_inv;

                let x_part = (x_old * one_minus_k)?;
                let z_part = (&z_new * k_inv)?;
                let x_new = (x_part + z_part)?;

                // 4. Store State
                self.z[i] = z_new;
                var.set(&x_new)?; // Model weights now hold 'x_new' (c)
            }
        }

        Ok(())
    }

    // Boilerplate
    pub fn learning_rate(&self) -> f64 {
        self.params.lr
    }
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }
}
