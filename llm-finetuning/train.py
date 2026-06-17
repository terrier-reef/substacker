#!/usr/bin/env python3
"""
Fine-tune Phi-2.7B on creative writing data using LoRA.
"""

import torch
import yaml
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_lora_config(config):
    """Create LoRA configuration for efficient fine-tuning."""
    lora_config = config['lora']
    return LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        task_type=TaskType.CAUSAL_LM,
    )

def load_model_for_training(config):
    """Load Phi-2.7B model with LoRA for CPU training."""
    model_name = config['model']['name']

    print(f"\n{'='*60}")
    print(f"Loading Phi-2.7B model: {model_name}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model'].get('trust_remote_code', True),
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in float16 for CPU efficiency
    print("Loading model in float16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=config['model'].get('trust_remote_code', True),
        low_cpu_mem_usage=True,
    )

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)

    print("\n" + "="*60)
    print("✓ Model loaded and LoRA applied")
    model.print_trainable_parameters()
    print("="*60 + "\n")

    return model, tokenizer

def load_training_data(config):
    """Load training data from JSONL file."""
    data_path = config['data']['train_file']
    text_column = config['data']['text_column']

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print(f"Loading training data from: {data_path}")

    # Load JSONL dataset
    dataset = load_dataset("json", data_files=data_path, split="train")

    print(f"✓ Loaded {len(dataset)} training examples\n")

    return dataset, text_column

def preprocess_data(dataset, tokenizer, text_column, max_length=512):
    """Tokenize training data."""

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    print("Tokenizing data...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
    )

    print(f"✓ Data tokenized\n")

    return tokenized_dataset

def train(config_path="config.yaml"):
    """Fine-tune Phi-2.7B on creative writing data."""

    print("\n" + "="*60)
    print("FINE-TUNING PHI-2.7B FOR CREATIVE WRITING")
    print("="*60 + "\n")

    # Load configuration
    config = load_config(config_path)

    # Load model and tokenizer
    model, tokenizer = load_model_for_training(config)

    # Load training data
    dataset, text_column = load_training_data(config)
    tokenized_dataset = preprocess_data(
        dataset,
        tokenizer,
        text_column,
        max_length=config['data']['max_length']
    )

    # Split into train/eval (80/20)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        save_strategy=config['training']['save_strategy'],
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy="epoch",
        report_to=[],  # Disable wandb for CPU training
        remove_unused_columns=False,
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Save memory on CPU
        seed=42,
    )

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save final model
    print("\n" + "="*60)
    print("SAVING FINE-TUNED MODEL")
    print("="*60 + "\n")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Fine-tuning complete!")
    print(f"✓ Model saved to: {output_dir}")
    print(f"\nNext step: Load this model in LM Studio")
    print(f"Model path: {os.path.abspath(output_dir)}\n")

if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    try:
        train(config_path)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
