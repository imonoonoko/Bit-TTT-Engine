from PIL import Image
import sys
import os

def convert_to_ico(input_path, output_path):
    try:
        img = Image.open(input_path)
        img.save(output_path, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
        print(f"Successfully converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Error converting: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python png2ico.py <input.png> <output.ico>")
        sys.exit(1)

    convert_to_ico(sys.argv[1], sys.argv[2])
