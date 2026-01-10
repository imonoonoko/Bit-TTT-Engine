import argparse
import os
import struct
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def clean_wiki40b(text):
    """
    Wiki40b format:
    _START_ARTICLE_
    Title
    _START_SECTION_
    Section Title
    _START_PARAGRAPH_
    Content...
    
    We mainly want the content.
    For simplicity, we can replace strictly structural tags with newlines 
    and maybe keep Article titles.
    
    Actually, Wiki40b text usually comes as one big string with these tags.
    We'll treat them as separators.
    """
    # Simple strategy: Replace tags with special tokens or just newlines?
    # For a base model, we want clean text.
    # Let's remove the tags lines.
    
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if line.startswith('_START_'):
            # It's a structure tag.
            # _START_ARTICLE_ or _START_SECTION_ or _START_PARAGRAPH_
            # We can skip the tag itself.
            continue
        cleaned.append(line)
    
    return "\n".join(cleaned)

def train_tokenizer(dataset, vocab_size=32000):
    print("Training Tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|padding|>", "<|unk|>"]
    )
    
    # Iterator for training (take first 20,000 examples)
    def batch_iterator():
        for i, example in enumerate(dataset):
            if i > 20000:
                break
            yield clean_wiki40b(example['text'])
            
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer

def process_and_save(dataset, tokenizer, output_path, limit=None):
    print(f"Processing to {output_path}...")
    with open(output_path, 'wb') as f:
        count = 0
        for i, example in enumerate(dataset):
            if limit and i >= limit:
                break
                
            text = clean_wiki40b(example['text'])
            encoded = tokenizer.encode(text)
            ids = encoded.ids
            
            # Pack as u32 little endian
            for token_id in ids:
                f.write(struct.pack('<I', token_id))
            
            # Add EOS token (assumed to be 0 or check tokenizer)
            eos_id = tokenizer.token_to_id("<|endoftext|>")
            if eos_id is None: eos_id = 0
            f.write(struct.pack('<I', eos_id))
            
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} articles...", end='\r')
                
    print(f"\nFinished processing {count} articles.")

def main():
    parser = argparse.ArgumentParser(description="Prepare Wiki40b-ja dataset")
    parser.add_argument("--output_dir", type=str, default="data/Wiki40b", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of articles (for testing)")
    parser.add_argument("--skip_tokenizer", action="store_true", help="Skip tokenizer training if exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading Wiki40b-ja (Streaming)...")
    ds_train = load_dataset("wiki40b", "ja", split="train", streaming=True)
    ds_val = load_dataset("wiki40b", "ja", split="validation", streaming=True)
    
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    
    if args.skip_tokenizer and os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = train_tokenizer(ds_train, vocab_size=args.vocab_size)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

    # Process Train
    process_and_save(ds_train, tokenizer, os.path.join(args.output_dir, "train.u32"), limit=args.limit)
    
    # Process Val
    # Note: Validation set in Wiki40b is separate.
    process_and_save(ds_val, tokenizer, os.path.join(args.output_dir, "val.u32"), limit=args.limit)

if __name__ == "__main__":
    main()
