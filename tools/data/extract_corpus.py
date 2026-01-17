"""
Extract raw text from Wiki40b-ja dataset to corpus.txt for Rust tokenizer training.
This is a one-time preprocessing step.

Usage:
    python tools/extract_corpus.py --output data/Wiki40b/corpus.txt --limit 50000
"""

import argparse
import os


def clean_wiki40b(text: str) -> str:
    """Remove Wiki40b structural tags."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if line.startswith('_START_'):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def main():
    parser = argparse.ArgumentParser(description="Extract raw text from Wiki40b")
    parser.add_argument("--output", type=str, default="data/Wiki40b/corpus.txt",
                        help="Output file path")
    parser.add_argument("--limit", type=int, default=50000,
                        help="Maximum number of articles to extract")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading Wiki40b-ja (Streaming)...")
    from datasets import load_dataset
    ds = load_dataset("wiki40b", "ja", split="train", streaming=True)

    print(f"Extracting {args.limit} articles to {args.output}...")
    count = 0
    with open(args.output, 'w', encoding='utf-8') as f:
        for example in ds:
            if count >= args.limit:
                break
            
            text = clean_wiki40b(example['text'])
            f.write(text + "\n")
            
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} articles...", end='\r')

    print(f"\n✅ Extracted {count} articles to {args.output}")
    
    # Show file size
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"📁 File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
