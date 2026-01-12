# Verification Plan (Phase 15)

## 1. "Mock 70B" Test (Capacity Test)
- **Problem**: We might not have a 70B model or 14GB file to test.
- **Method**:
    - Create a synthetic config describing a "Huge" model (e.g., hidden_dim=8192, layers=80).
    - Generate a dummy `.bitt` file filled with zeros (sparse file if possible, or actual 14GB dummy file).
    - Attempt to `load` this on a system with restricted VRAM (e.g., set `CUDA_VISIBLE_DEVICES` to a card with < 8GB, or artificial clamp).

## 2. Offloading Correctness (Equivalence Test)
- **Goal**: Ensure split processing (GPU+CPU) yields same result as full GPU.
- **Steps**:
    1. Run a small model (TinyLlama class) fully on GPU. Record logits.
    2. Run same model with forced split (Layers 0-2 on GPU, 3-7 on CPU).
    3. Compare logits (BF16/F32 tolerance).

## 3. Performance Benchmarking
- Measure `tokens/sec` degradation compared to full GPU.
- Metrics:
    - Load Time (First Token Latency)
    - Decoding Speed (Tokens/sec)
    - Max VRAM Usage
