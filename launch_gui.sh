#!/bin/bash
cd "$(dirname "$0")"
echo "🚀 Launching Bit-TTT Training GUI..."
cargo run --release --bin launcher
