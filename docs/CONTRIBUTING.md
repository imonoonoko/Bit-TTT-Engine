# Contributing to Bit-TTT Engine

Thank you for your interest in contributing to Bit-TTT Engine!

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/imonoonoko/Bit-TTT-Engine.git
cd Bit-TTT-Engine

# Build
cargo build

# Run tests
cargo test --workspace

# Check for issues
cargo clippy --workspace
```

## 📋 Development Guidelines

### Code Style
- Use `cargo fmt` before committing
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Add doc comments for public APIs

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(core): add new layer type
fix(train): resolve checkpoint loading issue
docs: update README
refactor(model): split large file into modules
```

### Pull Requests
1. Fork the repository
2. Create a feature branch (`feat/my-feature`)
3. Make your changes
4. Run `cargo test` and `cargo clippy`
5. Submit a pull request

## 🏗️ Project Structure

```
crates/
├── rust_engine/     # Core library (cortex_rust)
│   ├── layers/      # Neural network layers
│   ├── model/       # Model architecture
│   └── python.rs    # Python bindings
│
└── bit_llama/       # CLI application
    ├── train/       # Training pipeline
    ├── gui/         # Tauri GUI
    └── cli.rs       # CLI entry point
```

## 🎯 Good First Issues

- [ ] Add more unit tests for layers
- [ ] Improve error messages
- [ ] Add documentation examples
- [ ] Optimize memory usage

## 📬 Contact

- GitHub Issues: [Report bugs or request features](https://github.com/imonoonoko/Bit-TTT-Engine/issues)

---

*Thank you for contributing!*
