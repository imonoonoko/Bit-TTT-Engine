import torch
import torch.nn as nn
import torch.nn.functional as F

def activation_quant(x):
    """
    Simulate 8-bit quantization for activations.
    Range: [-128, 127] scaled to input range.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return y

def weight_quant(w):
    """
    Simulate 1.58-bit quantization for weights ({-1, 0, 1}).
    Scale is calculated as mean absolute value.
    """
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    y = (w * scale).round().clamp(-1, 1) / scale
    # STE: detach the quantized value for gradient calculation
    y = (y - w).detach() + w
    return y

class BitLinear(nn.Linear):
    """
    Linear layer with 1.58-bit weights and 8-bit activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    def forward(self, x):
        # 1. Quantize weights (simulated)
        w_quant = weight_quant(self.weight)
        
        # 2. Quantize activations (simulated)
        x_quant = activation_quant(x)
        
        # 3. Linear operation (using FP32/BF16 kernel under the hood for now)
        return F.linear(x_quant, w_quant, self.bias)

if __name__ == "__main__":
    # Test
    layer = BitLinear(10, 5)
    x = torch.randn(2, 10)
    y = layer(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
    print("Weights (approx 1.58bit check):", layer.weight.data[0][:5])
    print("Forward pass successful.")
