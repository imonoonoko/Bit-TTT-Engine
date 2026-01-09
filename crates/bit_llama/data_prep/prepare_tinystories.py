import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tqdm import tqdm

DATA_DIR = "../data/TinyStories"
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_BIN = os.path.join(DATA_DIR, "train.bin")
TOKENIZER_JSON = os.path.join(DATA_DIR, "tokenizer.json")

def main():
    # 1. Load Dataset
    print("Loading TinyStories (this may take a while to download)...")
    # Use streaming=True if disk space is tight, but TinyStories is small enough (couple GBs).
    # We load full train split.
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # 2. Train Tokenizer
    print("Training Tokenizer...")
    # Initialize BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    vocab_size = 16384
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<|endoftext|>", "<|padding|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Iterator for training tokenizer (use subset to speed up?)
    # 100k samples is enough for 16k vocab.
    def batch_iterator(sample_count=100000, batch_size=1000):
        for i in range(0, min(len(dataset), sample_count), batch_size):
            yield dataset[i : i + batch_size]["text"]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save(TOKENIZER_JSON)
    print(f"Tokenizer saved to {TOKENIZER_JSON}")

    # 3. Tokenize & Save Binary
    print("Tokenizing and Saving to train.bin...")
    
    # Prepare File
    # We use uint16 because vocab < 65536
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    
    with open(OUTPUT_BIN, "wb") as f:
        # Buffer for writing
        batch_size = 10000
        buffer = []
        
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_texts = dataset[i : i + batch_size]["text"]
            encodings = tokenizer.encode_batch(batch_texts)
            
            chunk = []
            for enc in encodings:
                chunk.extend(enc.ids)
                chunk.append(eos_token_id)
            
            # Convert to numpy uint16
            data_np = np.array(chunk, dtype=np.uint16)
            f.write(data_np.tobytes())

    print(f"Data saved to {OUTPUT_BIN}")

if __name__ == "__main__":
    main()
