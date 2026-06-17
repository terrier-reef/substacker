# Phi-2.7B Fine-Tuning for Creative Writing

A lightweight framework for fine-tuning Phi-2.7B on creative writing and storytelling data. Train on CPU, load results into LM Studio.

## Features

✓ **Phi-2.7B** - Capable 2.7B model optimized for creative tasks  
✓ **LoRA** - Parameter-efficient fine-tuning (only trains ~1% of params)  
✓ **CPU Training** - No GPU required, optimized for 16GB+ RAM  
✓ **LM Studio Compatible** - Fine-tuned models load directly into LM Studio  
✓ **Simple** - Single `train.py` script, minimal dependencies  

## Requirements

- **Python**: 3.9+
- **RAM**: 16GB minimum
- **Disk**: 10GB for model + training data
- **Time**: 1-3 hours for training

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Training Data

Add your creative writing samples to `data/creative_stories.jsonl`:

```jsonl
{"text": "Your story here..."}
{"text": "Another story..."}
{"text": "More creative content..."}
```

**Format**: One JSON object per line with a `text` field containing the story.

### 2. Configure (Optional)

Edit `config.yaml` to adjust:
- `num_train_epochs`: Number of training passes (default: 3)
- `learning_rate`: How fast the model learns (default: 2e-4)
- `lora.r`: LoRA rank - higher = more capacity but slower (default: 16)

### 3. Start Training

```bash
python train.py
```

The script will:
1. Download Phi-2.7B (~5.7GB, first time only)
2. Apply LoRA
3. Train on your data
4. Save fine-tuned model to `output/phi-creative-v1/`

**Expected output:**
```
============================================================
FINE-TUNING PHI-2.7B FOR CREATIVE WRITING
============================================================

Loading Phi-2.7B model: microsoft/phi-2
Loading tokenizer...
Loading model in float16...
Applying LoRA...

✓ Model loaded and LoRA applied
trainable params: 1,234,560 || all params: 2,708,787,200 || trainable%: 0.046
```

Training will take 1-3 hours on CPU depending on data size.

### 4. Load in LM Studio

1. **Open LM Studio**
2. **Click folder icon** to browse models
3. **Navigate to**: `output/phi-creative-v1/`
4. **Click Load** or select the model
5. **Start chatting!**

Your fine-tuned model is now ready for creative writing.

## Data Format

### JSONL Format (Recommended)

One JSON object per line:
```jsonl
{"text": "Story 1..."}
{"text": "Story 2..."}
```

### Data Tips

- **Quality > Quantity**: 50 well-written stories > 1000 mediocre ones
- **Consistency**: Stories in similar style/tone work better
- **Length**: 200-2000 words per story is ideal
- **Variety**: Mix different genres and writing styles

### Creating Training Data

```python
import json

stories = [
    "Once upon a time...",
    "In a distant land...",
    "The hero embarked on...",
]

with open("data/creative_stories.jsonl", "w") as f:
    for story in stories:
        f.write(json.dumps({"text": story}) + "\n")
```

## Monitoring Training

During training, you'll see:
- Epoch progress (Epoch 1/3, 2/3, 3/3)
- Training loss (should decrease over time)
- Eval loss (validation performance)

Lower loss = better learning. Typical training:
```
[100/300] loss: 3.45
[200/300] loss: 2.89
[300/300] loss: 2.45
```

## Output

Fine-tuned model is saved to:
```
output/phi-creative-v1/
├── adapter_config.json    # LoRA configuration
├── adapter_model.bin      # Trained LoRA weights
├── config.json           # Model config
├── generation_config.json
├── pytorch_model.bin     # Model weights
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

**To use in LM Studio**: Point to the `output/phi-creative-v1/` directory.

## Advanced Usage

### Custom Configuration

Edit `config.yaml`:

```yaml
training:
  num_train_epochs: 5          # Train longer
  learning_rate: 1e-4          # More careful learning
  
lora:
  r: 32                        # Higher rank = more capacity
  lora_dropout: 0.1            # More regularization
```

### Multiple Fine-Tuned Versions

Create different versions for different styles:

```bash
# Copy config for variant
cp config.yaml config_poetry.yaml

# Edit config_poetry.yaml with poetry-specific settings
# Create data/poetry.jsonl with poems

# Train variant
python train.py config_poetry.yaml
```

### Using Different Model

Change in `config.yaml`:
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Even smaller
  # or
  name: "mistralai/Mistral-7B-Instruct-v0.1"  # Larger (needs more RAM)
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or gradient accumulation:
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # Increase this
```

### Model Won't Load in LM Studio

- Ensure all files are in `output/phi-creative-v1/`
- Check file permissions
- Try restarting LM Studio
- Use Show advanced settings in LM Studio to reduce context length

### Slow Training

This is normal on CPU! Training takes 1-3 hours depending on:
- Amount of training data
- CPU speed
- Number of epochs

### Data Not Loading

Verify JSONL format:
```python
import json
with open("data/creative_stories.jsonl") as f:
    for line in f:
        data = json.loads(line)
        print(data.get("text", "")[:50])
```

## Performance Tips

1. **Reduce context length in LM Studio** (from 2048 to 512)
2. **Generate fewer tokens** at a time
3. **Use lower temperature** (0.5-0.7) for consistency
4. **Batch predictions** in LM Studio for speed

## Model Comparison

| Model | Size | Speed | Quality | RAM |
|-------|------|-------|---------|-----|
| Phi-2.7B | 5.7GB | Fast | Good | 16GB |
| TinyLlama 1.1B | 2.3GB | Very Fast | Basic | 8GB |
| Mistral 7B | 14.3GB | Slow | Excellent | 32GB+ |

For creative writing on 16GB, **Phi-2.7B is perfect**.

## Workflow Example

```bash
# 1. Add your stories to data/creative_stories.jsonl
echo '{"text": "My first story about dragons..."}' >> data/creative_stories.jsonl
echo '{"text": "A tale of wonder and magic..."}' >> data/creative_stories.jsonl

# 2. Train
python train.py

# 3. Load in LM Studio
# Navigate to: output/phi-creative-v1/

# 4. Use for creative writing!
# Prompt: "Write me a story about..."
```

## Next Steps

1. ✅ Prepare your training data
2. ✅ Run `python train.py`
3. ✅ Load in LM Studio
4. ✅ Write amazing stories!

## Resources

- [Phi-2 Model](https://huggingface.co/microsoft/phi-2)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [LM Studio](https://lmstudio.ai)

## License

This framework is open source. Models are subject to their respective licenses.
