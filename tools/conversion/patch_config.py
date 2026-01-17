import json
import os
import argparse

def patch_config(model_dir: str):
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return

    print(f"⚙️  Patching {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Llama-3 Specs
    # rope_theta = 500,000.0
    # max_position_embeddings = 8192 (or 128k for Instruct, but 8192 is base likely)
    # The source model config should have these, but if we don't have source handy, force them.

    updates = {
        "rope_theta": 500000.0,
        "max_position_embeddings": 8192,
        "n_gpu_layers": 18, # Validation: 18 layers safe target for 8GB VRAM + Q8 Cache reliability
        "lm_head_cpu": True
    }

    for k, v in updates.items():
        if k not in config:
            print(f"   + Adding {k}: {v}")
            config[k] = v
        else:
            print(f"   ! {k} exists: {config[k]} (Overwriting with {v})")
            config[k] = v

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print("✅ Config patched.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()
    patch_config(args.model_dir)
