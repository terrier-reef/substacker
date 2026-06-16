"""
Convert HuggingFace models to GGUF format for CPU-only inference.
Requires llama.cpp to be installed.
"""

import os
import subprocess
import json
from pathlib import Path

def convert_to_gguf(model_path, output_path=None, quantization_level=None):
    """
    Convert HuggingFace model to GGUF format.

    Args:
        model_path: Path to HuggingFace model directory
        output_path: Output path for GGUF model (optional)
        quantization_level: Quantization level (q4_0, q4_1, q5_0, q5_1, q8_0)
                           Default: q4_0 (4-bit, most aggressive)
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if output_path is None:
        output_path = os.path.join(model_path, "model.gguf")

    if quantization_level is None:
        quantization_level = "q4_0"

    print(f"Converting model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {quantization_level}")

    try:
        # Try using llama-cpp-python's conversion utilities
        from llama_cpp import llama_cpp

        print("Using llama-cpp-python for conversion...")
        # This is a simplified approach - actual conversion may vary
        # Typically requires the llama.cpp convert.py script

    except ImportError:
        print("llama-cpp-python not found. Using alternative approach...")
        _convert_with_script(model_path, output_path, quantization_level)

    if os.path.exists(output_path):
        size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"\n✓ Conversion successful!")
        print(f"  Model size: {size_gb:.2f} GB")
        return output_path
    else:
        raise RuntimeError("Conversion failed - output file not created")

def _convert_with_script(model_path, output_path, quantization_level):
    """Use llama.cpp's conversion script."""
    print("\nNote: For full GGUF conversion, you may need to:")
    print("  1. Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
    print("  2. Run: python llama.cpp/convert.py <model_path> --outfile <output_path>")
    print("  3. Quantize: ./llama.cpp/quantize <model.gguf> <model-q4.gguf> q4_0")

def quantize_gguf(gguf_path, output_path=None, quantization_type="q4_0"):
    """
    Quantize an existing GGUF model to reduce size further.

    Args:
        gguf_path: Path to GGUF model
        output_path: Output path for quantized model
        quantization_type: Quantization type (q4_0, q4_1, q5_0, q5_1, q8_0)
    """

    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

    if output_path is None:
        base = os.path.splitext(gguf_path)[0]
        output_path = f"{base}-{quantization_type}.gguf"

    print(f"Quantizing GGUF model: {gguf_path}")
    print(f"Type: {quantization_type}")
    print(f"Output: {output_path}")

    try:
        # Try using llama.cpp's quantize tool
        result = subprocess.run(
            ["quantize", gguf_path, output_path, quantization_type],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"Warning: Quantization may have issues")
            print(result.stderr)

    except FileNotFoundError:
        print("quantize tool not found in PATH")
        print("Install llama.cpp and add its directory to PATH")
        return None

    if os.path.exists(output_path):
        original_size = os.path.getsize(gguf_path) / (1024**3)
        quantized_size = os.path.getsize(output_path) / (1024**3)
        ratio = (1 - quantized_size / original_size) * 100

        print(f"\n✓ Quantization complete!")
        print(f"  Original:   {original_size:.2f} GB")
        print(f"  Quantized:  {quantized_size:.2f} GB")
        print(f"  Reduction:  {ratio:.1f}%")
        return output_path

    return None

def get_model_info(model_path):
    """Get model info from config.json."""
    config_path = os.path.join(model_path, "config.json")

    if not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    return {
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HuggingFace models to GGUF format")
    parser.add_argument("model_path", help="Path to HuggingFace model")
    parser.add_argument("--output", help="Output path for GGUF model")
    parser.add_argument("--quantize", default="q4_0", help="Quantization type")
    parser.add_argument("--info", action="store_true", help="Show model info")

    args = parser.parse_args()

    if args.info:
        info = get_model_info(args.model_path)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print("Could not read model info")
    else:
        try:
            convert_to_gguf(args.model_path, args.output, args.quantize)
        except Exception as e:
            print(f"Error: {e}")
