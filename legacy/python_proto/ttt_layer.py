import torch
import torch.nn as nn
import torch.nn.functional as F
from bit_linear import BitLinear

class TTTLayer(nn.Module):
    """
    Test-Time Training (TTT) Layer.
    Maintains a hidden weight matrix `W_hidden` that is updated *during the forward pass*
    based on the input sequence.
    """
    def __init__(self, hidden_dim, inner_lr=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        
        # The "Learning" Model inside the layer.
        # Ideally this would vary per batch, but for prototype we'll assume batch_size=1
        # or handle state carefully. For simplicity: Batch-independent updates?
        # Standard TTT: State is per-sequence.
        
        # Projection for input to the hidden state dimension
        self.proj_down = BitLinear(hidden_dim, hidden_dim // 4)
        self.proj_up = BitLinear(hidden_dim // 4, hidden_dim)

    def forward_state_update(self, W_state, x_t):
        """
        Perform one step of gradient descent on W_state using x_t.
        Loss Objective: Autoencoder / Reconstruction.
        Goal: Compress x_t into W_state.
        """
        # 1. Prediction (Reconstruction)
        # Using a low-rank approximation: x -> down -> W -> up -> x_recon
        # W_state is (dim, dim) or similar.
        # Let's say W_state maps down_projected -> down_projected
        
        feat = self.proj_down(x_t) # (B, D_small)
        pred_feat = torch.matmul(feat, W_state) # Linear map on features
        
        # 2. Loss: Reconstruction of the feature itself (like simplified predictive coding)
        # L = || pred_feat - feat ||^2
        # Error = pred_feat - feat
        error = pred_feat - feat
        
        # 3. Gradient w.r.t W_state
        # dL/dW = error^T * feat
        grad = torch.matmul(feat.transpose(-2, -1), error)
        
        # 4. Update
        # W_new = W - lr * grad
        # In Bit-TTT, this grad would be quantized to {-1, 0, 1}
        W_state_new = W_state - self.inner_lr * grad
        
        return W_state_new, pred_feat

    def forward(self, x):
        """
        x: (Batch, Seq, HiddenDim)
        """
        B, S, D = x.shape
        D_small = D // 4
        
        # Initial State: W_h (B, D_small, D_small)
        W_state = torch.zeros(B, D_small, D_small, device=x.device)
        
        outputs = []
        
        # Sequential Processing (Causal)
        for t in range(S):
            x_t = x[:, t:t+1, :] # (B, 1, D)
            
            # Update state learning from current input (Testing Time Training)
            # In TTT-Linear, we use W_{t-1} to predict, and update to W_t using x_t.
            # But the output of the layer should benefit from the *updated* state or current?
            # Standard RNN: h_t = f(x_t, h_{t-1}). Output based on h_t.
            
            # Step 1: Update State (Learn)
            # We "learn" the current token to compress it.
            W_state, pred_feat = self.forward_state_update(W_state, x_t)
            
            # Step 2: Generate Output
            # Can use the reconstructed feature or the state itself.
            # Let's project back up.
            out_t = self.proj_up(pred_feat) # (B, 1, D)
            outputs.append(out_t + x_t) # Residual connection
            
        return torch.cat(outputs, dim=1)
