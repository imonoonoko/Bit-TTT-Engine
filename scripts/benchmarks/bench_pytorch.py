import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Configuration (Must match Bit-TTT)
CONTEXT_LEN = 128
BATCH_SIZE = 32
DIM = 256
LAYERS = 4
VOCAB = 16384
STEPS = 100

print(f"--- PyTorch Benchmark (TinyStories) ---")
print(f"Config: B={BATCH_SIZE}, T={CONTEXT_LEN}, D={DIM}, L={LAYERS}, V={VOCAB}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Dummy Data (Memory Mapped) using numpy for speed
# In real scenario, we load similar to Rust, but random data is enough for speed test
class RandomDataset(Dataset):
    def __init__(self, steps):
        self.steps = steps

    def __len__(self):
        return self.steps * BATCH_SIZE

    def __getitem__(self, idx):
        return torch.randint(0, VOCAB, (CONTEXT_LEN,)), torch.randint(0, VOCAB, (CONTEXT_LEN,))

# Simple Transformer (Llama-like)
class SimpleLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=DIM, nhead=8, dim_feedforward=DIM*4, batch_first=True, norm_first=True)
            for _ in range(LAYERS)
        ])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.head(h)
        return logits

model = SimpleLlama().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Data Loader
dataset = RandomDataset(STEPS)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)

print("Starting training loop...")
start_time = time.time()

model.train()
total_tokens = 0

for step, (x, y) in enumerate(loader):
    if step >= STEPS: break
    
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits.reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()
    optimizer.step()
    
    total_tokens += BATCH_SIZE * CONTEXT_LEN
    
    if step % 10 == 0:
        elapsed = time.time() - start_time
        tps = total_tokens / elapsed if elapsed > 0 else 0
        print(f"Step {step:4} | Loss: {loss.item():.4f} | {tps:.2f} tok/s")

total_time = time.time() - start_time
avg_tps = total_tokens / total_time
print(f"\n--- Result ---")
print(f"Total Time: {total_time:.2f} s")
print(f"Avg TPS:    {avg_tps:.2f} tok/s")
