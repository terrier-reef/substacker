import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import platform

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(model_name=None, config=None, device="cpu", use_quantization=True):
    """Load model and tokenizer optimized for CPU inference."""

    if config is None:
        config = load_config()

    if model_name is None:
        model_name = config['model']['name']

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model'].get('trust_remote_code', True),
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")

    # Try 4-bit quantization first, fall back if it fails
    if use_quantization:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            print("Attempting 4-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="cpu",
                trust_remote_code=config['model'].get('trust_remote_code', True),
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("✓ 4-bit quantization successful")

        except (ImportError, RuntimeError) as e:
            print(f"⚠ 4-bit quantization failed: {e}")
            print("Falling back to float16 without quantization...")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                trust_remote_code=config['model'].get('trust_remote_code', True),
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("✓ Model loaded with float16 (no quantization)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            trust_remote_code=config['model'].get('trust_remote_code', True),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✓ Model loaded without quantization")

    # Set model to inference mode
    model.eval()

    print(f"Model loaded on device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Platform: {platform.system()}")

    return model, tokenizer

def get_model_size(model):
    """Get approximate model size in GB."""
    param_size = sum(p.numel() for p in model.parameters()) / 1e9
    buffer_size = sum(b.numel() for b in model.buffers()) / 1e9
    return param_size + buffer_size
