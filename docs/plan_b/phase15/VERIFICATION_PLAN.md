# Verification Plan: Phase 15

## 1. Definition of Done
*   [ ] **Device Map Config**: 設定 (`config.json` or CLI args) で、「最初のN層はGPU、残りはCPU」といった指定ができること。
*   [ ] **Memory Constraints**: VRAM 4GB制限環境（シミュレーション）で、本来8GB必要なモデルがOOM (Out Of Memory) せずに動作すること。
*   [ ] **Inference Speed**: CPUオフロード時でも、許容範囲内（例: Pure CPUよりは高速）で動作すること。

## 2. Verification Steps
### Step 1: Memory Limit Test
```bash
# 擬似的にGPUメモリを制限して実行（または小さいGPU環境を用意）
cargo run --release --bin chat -- --model 7B-Model --vram-limit 4GB
```

### Step 2: Layer Offloading Check
ログ出力で各レイヤーの配置デバイスを確認する。
```text
Layer 0: Cuda(0)
Layer 1: Cuda(0)
...
Layer 10: Cpu
Layer 11: Cpu
```

## 3. Observability
*   推論中に現在の VRAM 使用量と、レイヤーのスワップ発生回数をログ出力するモニターを追加する。
