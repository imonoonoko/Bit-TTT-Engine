import os
from huggingface_hub import snapshot_download

def download_tinyllama():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    local_dir = "models/TinyLlama-1.1B"

    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        allow_patterns=["*.json", "*.safetensors"],
        ignore_patterns=["*.bin", "*.pth"]
    )
    print("Download complete.")

if __name__ == "__main__":
    download_tinyllama()
