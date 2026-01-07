import torch
import matplotlib.pyplot as plt
import os
from ttt_layer import TTTLayer

def test_memory_association():
    print("=== Testing Bit-TTT Memory Capability ===")
    
    # 1. Setup
    dim = 64
    seq_len = 20
    model = TTTLayer(dim, inner_lr=0.1)
    
    # 2. Create a specific pattern sequence: A, B, C, A, B, C...
    # We want to see if the error drops when 'A' appears the second time.
    torch.manual_seed(42)
    
    # Generate 3 distinct vectors
    vocab = torch.randn(3, dim)
    vocab = vocab / vocab.norm(dim=-1, keepdim=True) # Normalize
    
    # Create sequence: 0, 1, 2, 0, 1, 2...
    indices = [0, 1, 2] * 4 # Length 12
    x = torch.stack([vocab[i] for i in indices]).unsqueeze(0) # (1, 12, dim)
    
    print(f"Sequence indices: {indices}")
    
    # 3. Manually run loop to inspect internal losses
    # The TTTLayer as implemented currently does the whole loop.
    # We'll modify it slightly or just run it and assume we can't easily peek inside 
    # unless we hook it or modify the class. 
    # Actually, let's run it step-by-step using the internal logic to capture the "Pre-Update Loss"
    
    W_state = torch.zeros(1, dim//4, dim//4)
    
    losses = []
    
    print("\nStep | Input | Reconstruction Loss (Before Update)")
    print("-" * 45)
    
    for t in range(len(indices)):
        x_t = x[:, t:t+1, :]
        idx_t = indices[t]
        
        # --- Inside TTTLayer logic ---
        # 1. Prediction (Reconstruction) *before* update
        feat = model.proj_down(x_t)
        pred_feat = torch.matmul(feat, W_state)
        
        # Loss (Are we surprised?)
        loss = (pred_feat - feat).pow(2).mean().item()
        
        # 2. Update State
        # W_state_new, _ = model.forward_state_update(W_state, x_t)
        # Note: forward_state_update computes gradients on *current* x_t.
        # Ideally, loss drops after update. But we want to see if loss is low for *future* same inputs.
        
        # Let's perform the update manually to match the class logic
        feat_detached = feat.detach() # TTT usually doesn't backprop through the "target" generation if unsupervised? 
        # Actually our implementation backprops through everything.
        
        # Recalculate for gradient
        pred_feat_grad = torch.matmul(feat, W_state)
        error = pred_feat_grad - feat
        grad = torch.matmul(feat.transpose(-2, -1), error)
        
        # Update
        W_state = W_state - model.inner_lr * grad
        
        losses.append(loss)
        print(f"{t:4d} |   {idx_t}   | {loss:.6f}")
        
    # Analysis
    # We expect loss for the second appearance of '0' (at index 3) to be lower than at index 0.
    loss_0_first = losses[0]
    loss_0_second = losses[3]
    
    print("-" * 45)
    print(f"Loss for '0' (1st time): {loss_0_first:.6f}")
    print(f"Loss for '0' (2nd time): {loss_0_second:.6f}")
    
    if loss_0_second < loss_0_first:
        print("\n[SUCCESS] Memory Effect Confirmed! Loss decreased for repeated token.")
    else:
        print("\n[FAIL] No Memory Effect observed.")
        
if __name__ == "__main__":
    test_memory_association()
