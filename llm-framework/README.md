# CPU-Only Fine-Tuned LLM Framework

A modular framework for fine-tuning and running large language models on CPU-only systems with no GPU/VRAM requirements.

## Features

- **CPU-Only Inference**: Optimized for systems with 16GB+ RAM and no GPU
- **4-Bit Quantization**: Aggressive quantization for memory efficiency
- **LoRA Fine-Tuning**: Parameter-efficient fine-tuning with minimal compute
- **Unrestricted Mode**: Remove safety filters and customize model behavior
- **REST API**: Serve models via Flask API
- **Streaming Generation**: Token-by-token streaming output
- **Batch Processing**: Process multiple prompts efficiently

## System Requirements

- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 20GB+ for models
- **CPU**: Multi-core processor
- **No GPU required**

## Installation

```bash
# Clone or download the framework
cd llm-framework

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Inference

```python
from models.load_model import load_model_and_tokenizer
from models.inference import CPUInferenceEngine

# Load model
model, tokenizer = load_model_and_tokenizer()

# Create inference engine
engine = CPUInferenceEngine(model, tokenizer)

# Generate text
prompt = "What is machine learning?"
result = engine.generate(prompt, max_tokens=256)
print(result)
```

### 2. Unrestricted Mode

```python
from models.load_model import load_model_and_tokenizer
from models.inference import CPUInferenceEngine
from safety.remove_guardrails import SafetyBypass

model, tokenizer = load_model_and_tokenizer()
engine = CPUInferenceEngine(model, tokenizer)
safety = SafetyBypass(model, tokenizer)

# Set unrestricted mode
safety.set_unrestricted_mode("creative")

# Generate with unrestricted prompt
prompt = "Tell me a story"
unrestricted_prompt = safety.prepare_prompt_for_unrestricted(prompt, "creative")
result = engine.generate(unrestricted_prompt)
print(result)
```

### 3. Fine-Tuning

```bash
# Fine-tune model with LoRA
python training/fine_tune.py

# Or with custom dataset
python training/fine_tune.py --dataset path/to/data.txt
```

### 4. REST API Server

```bash
# Start the API server
python api/server.py

# Server runs on http://localhost:5000
```

**Example API Requests:**

```bash
# Health check
curl http://localhost:5000/health

# Generate text
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 256}'

# Unrestricted generation
curl -X POST http://localhost:5000/generate/unrestricted \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me something", "mode": "creative", "max_tokens": 256}'

# Batch generation
curl -X POST http://localhost:5000/batch/generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Prompt 1", "Prompt 2"], "max_tokens": 256}'
```

## Configuration

Edit `config.yaml` to customize:

- **Model Selection**: Change the base model name
- **Training**: Adjust learning rate, epochs, batch size
- **Inference**: Set max tokens, temperature, top_p, top_k
- **LoRA**: Customize LoRA rank and dropout
- **Quantization**: Configure 4-bit quantization parameters

## Architecture

```
llm-framework/
├── models/               # Model loading and inference
│   ├── load_model.py    # Load and initialize models
│   └── inference.py     # CPU-optimized inference engine
├── training/            # Fine-tuning pipeline
│   ├── fine_tune.py     # LoRA fine-tuning
│   ├── datasets.py      # Data loading utilities
│   └── evaluation.py    # Model evaluation
├── quantization/        # Model quantization
│   ├── convert_to_gguf.py  # Convert to GGUF format
│   └── optimize_cpu.py  # CPU optimizations
├── safety/              # Safety and behavior customization
│   ├── remove_guardrails.py # Bypass safety filters
│   └── custom_behavior.py   # Custom behavior injection
├── api/                 # REST API
│   ├── server.py        # Flask API server
│   └── client.py        # Client utilities
├── config.yaml          # Configuration file
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## Advanced Usage

### Custom Fine-Tuning

```python
from training.fine_tune import train

# Fine-tune with custom dataset
train(
    config_path="config.yaml",
    dataset_path="path/to/training_data.txt",
    output_dir="./custom_model"
)
```

### Safety Customization

```python
from safety.remove_guardrails import SafetyBypass, UnrestrictedPrompting

# Remove safety filters
safety.set_unrestricted_mode("general")

# Use jailbreak patterns
prompt = UnrestrictedPrompting.create_jailbreak_prompt(
    "Generate creative content",
    pattern="fiction"
)

# Combine advanced techniques
prompt = UnrestrictedPrompting.combine_techniques("Your request here")
```

## Performance Optimization

For 16GB RAM systems:

1. **Batch Size**: Use batch_size=1 for inference
2. **Gradient Accumulation**: Increase gradient_accumulation_steps during training
3. **Memory Mapping**: Use memory-mapped datasets
4. **Threading**: Disable threaded dataloaders (dataloader_num_workers=0)
5. **Quantization**: Use 4-bit quantization always

## Troubleshooting

**Out of Memory (OOM)**:
- Reduce batch_size to 1
- Reduce max_tokens
- Enable gradient_checkpointing
- Use aggressive 4-bit quantization

**Slow Inference**:
- Use smaller models (Mistral 7B instead of 13B)
- Batch multiple requests
- Use GGUF quantized format
- Reduce n_threads if CPU overheating

**Model Loading Issues**:
- Ensure model is available on HuggingFace Hub
- Check internet connection for first download
- Verify model compatibility with transformers version

## Models Supported

- Mistral 7B (recommended)
- Llama 2 7B / 13B
- Phi 2.7B (ultra-fast)
- MPT 7B
- Other PEFT-compatible models

## Ethical Considerations

This framework enables removal of safety filters and guardrails. Users are responsible for:

- Complying with applicable laws and regulations
- Respecting intellectual property and copyright
- Not using for deception, misinformation, or harm
- Responsible disclosure of any security issues
- Ethical use of AI capabilities

## License

This framework is provided as-is for research and educational purposes.

## Contributing

Contributions welcome! Please ensure:

- Code is well-documented
- Changes follow existing patterns
- Documentation is updated
- Ethical considerations are addressed
