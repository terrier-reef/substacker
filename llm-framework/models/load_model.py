import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(model_name=None, config=None, device="cpu"):
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

    # CPU-only 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cpu",
        trust_remote_code=config['model'].get('trust_remote_code', True),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Set model to inference mode
    model.eval()

    print(f"Model loaded on device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer

def get_model_size(model):
    """Get approximate model size in GB."""
    param_size = sum(p.numel() for p in model.parameters()) / 1e9
    buffer_size = sum(b.numel() for b in model.buffers()) / 1e9
    return param_size + buffer_size
