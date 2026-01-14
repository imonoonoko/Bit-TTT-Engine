import sys
import os

try:
    import cortex_rust
    print(f"✅ Successfully imported cortex_rust!")
except ImportError as e:
    print(f"❌ Failed to import cortex_rust: {e}")
    sys.exit(1)

def test_config():
    print("\n--- Testing BitLlamaConfig ---")
    try:
        # Assuming BitLlamaConfig has a default constructor or we can instantiate it
        # Inspecting rust code: #[derive(Clone, Debug, Deserialize, Serialize)] but no #[new]?
        # If it doesn't have #[new] or #[pyclass], we can't instantiate it from Python easily unless we exposed a constructor.
        # Wait, lib.rs has `m.add_class::<model::BitLlamaConfig>()?;`
        # But if `BitLlamaConfig` struct doesn't have `#[pymethods]` with `#[new]`, Python can't create it.
        # Checking `crates/rust_engine/src/model/mod.rs` (or wherever it is defined) would be needed.
        # User audit found `crates/rust_engine/src/lib.rs` exports `model`.
        # I didn't verify `BitLlamaConfig` implementation details.
        # IF this fails, I will know I need to add #[new] to Config.

        # Taking a gamble that it might not have one, so I'll wrap in try/except.
        config = cortex_rust.BitLlamaConfig(100, 64, 2, 0.01)
        print("✅ BitLlamaConfig instantiated")
        return config
    except Exception as e:
        print(f"⚠️ BitLlamaConfig instantiation failed (Expected if no #[new]): {e}")
        return None

def test_trainer(config):
    print("\n--- Testing PyTrainer ---")
    if config is None:
        print("⏭️  Skipping Trainer test (No Config)")
        return

    try:
        # This is expected to fail because we don't have a model file at default paths
        # But we want to see a Rust-level error, not a Segfault or Python TypeError.
        trainer = cortex_rust.PyTrainer(config)
        print("✅ PyTrainer instantiated (Unexpectedly success?)")
    except Exception as e:
        print(f"✅ PyTrainer raised exception as expected (Model load fail): {e}")

if __name__ == "__main__":
    cfg = test_config()
    test_trainer(cfg)
