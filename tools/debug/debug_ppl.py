import sys
import os
import torch
import torch.nn.functional as F

# Ensure tools dir is in path
sys.path.append('tools')

try:
    import cortex_rust
    from cortex_rust import BitLlamaConfig, BitLlama
    print("✅ Successfully imported cortex_rust")
except ImportError as e:
    print(f"❌ Failed to import cortex_rust: {e}")
    sys.exit(1)

def debug_inference():
    # 1. Configuration (TinyLlama-1.1B specific)
    # vocab=32000, hidden=2048, layers=22, heads=32, kv_heads=4, inter=5632
    print("⚙️  Configuring Model...")
    config = BitLlamaConfig(32000, 2048, 22, 1e-4)
    config.n_kv_heads = 4       # Critical GQA setting
    config.intermediate_dim = 5632
    config.n_heads = 32         # Explicitly set

    # 2. Path
    model_path = r"c:\Users\Humin\.gemini\antigravity\scratch\new-ai-project\Bit-TTT\models\TinyLlama-Adaptive-1.1B\model.safetensors"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    # 3. Load
    print(f"🚀 Loading model from {model_path}...")
    try:
        model = BitLlama(config, model_path, "cuda")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    # Load Tokenizer First
    from transformers import AutoTokenizer
    print("   Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path.replace("model.safetensors", ""))
    except:
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompt = "Hello world"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"\n📝 Prompt: {prompt!r} -> IDs: {input_ids}")

    # Forward on LAST token
    last_token = input_ids[-1]
    # We must feed strict sequence for cache?
    # PyBitLlama.generate_tokens handles sequence?
    # forward(token) is stateless unless we manually valid cache.
    # But PyBitLlama loop uses forward_one + state.
    # For single token test, we just feed 1 token.
    # But we need context.
    # Let's use PREFILL mechanism if possible?
    # PyBitLlama doesn't expose prefill.
    # We will use generate_tokens which loops forward_one.

    print("   Generating continuation...")
    # Generate 5 tokens
    gen_tokens = model.generate_tokens(input_ids, 5)
    print(f"   Generated IDs: {gen_tokens}")
    print(f"   Decoded: {tokenizer.decode(gen_tokens)}")

    return

    # Skip old single-step
    # print("\n🔬 Running Forward Pass on token [1] (<s>)...")
    # logits = model.forward(1)

    # 5. Analysis
    # t_logits = torch.tensor(logits)
    # probs = F.softmax(t_logits, dim=0)
    # top_k = torch.topk(probs, 10)

    # print("\n📊 Logits Statistics:")
    # print(f"   Mean: {t_logits.mean().item():.4f}")
    # print(f"   Std:  {t_logits.std().item():.4f}")
    # print(f"   Max:  {t_logits.max().item():.4f}")
    # print(f"   Min:  {t_logits.min().item():.4f}")

    # print("\n🏆 Top 10 Predictions:")
    # for i in range(10):
    #     idx = top_k.indices[i].item()
    #     prob = top_k.values[i].item()
    #     token_str = tokenizer.decode([idx])
    #     print(f"   #{i+1}: Token {idx:<5} ({token_str!r}) | Prob: {prob:.4f}")

    # 6. Compare with HuggingFace (Reference)
    print("\n⚖️  Comparing with HuggingFace Implementation...")
    try:
        from transformers import AutoModelForCausalLM
        print("   Loading HF Model (TinyLlama-1.1B)...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="cuda"
        )

        print("   Running HF Forward...")
        hf_input = torch.tensor([[1]], device="cuda") # <s>
        with torch.no_grad():
            hf_out = hf_model(hf_input)

        hf_logits = hf_model(hf_input).logits[0, -1, :] # [Vocab] (Last token)
        hf_probs = torch.softmax(hf_logits, dim=0)
        hf_top_k = torch.topk(hf_probs, 5)

        print("\n🏆 HF Reference Predictions:")
        for i in range(5):
            idx = hf_top_k.indices[i].item()
            prob = hf_top_k.values[i].item()
            token_str = tokenizer.decode([idx])
            print(f"   #{i+1}: Token {idx:<5} ({token_str!r}) | Prob: {prob:.4f}")

    except Exception as e:
        print(f"⚠️  Could not run HF comparison: {e}")

    # 7. Diagnosis
    std = t_logits.std().item()
    if std < 0.1:
        print("\n⚠️  DIAGNOSIS: [RANDOM NOISE]")
        print("   Logits are nearly uniform. Weights likely not applying or zeroed.")
    elif std > 100.0:
        print("\n⚠️  DIAGNOSIS: [EXPLODING]")
        print("   Logits are extremely large. Normalization missing or Scale factor error.")
    elif top_k.indices[0].item() == 0 or top_k.indices[0].item() == 2:
        print("\n⚠️  DIAGNOSIS: [EOS/UNK BIAS]")
        print("   Model predicting special tokens. Check Masking or Padding.")
    else:
        print("\n✅ DIAGNOSIS: [PLAUSIBLE DISTRIBUTION]")
        print("   Distribution looks structurally valid. Check Token Mapping or RoPE details.")

if __name__ == "__main__":
    debug_inference()
