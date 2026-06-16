import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_lora_config(config):
    """Setup LoRA configuration."""
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
    """Load model with 4-bit quantization for training."""
    model_name = config['model']['name']

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model'].get('trust_remote_code', True),
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model for training: {model_name}")

    # 4-bit quantization config
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
        low_cpu_mem_usage=True,
    )

    # Apply LoRA
    lora_config = setup_lora_config(config)
    model = get_peft_model(model, lora_config)

    print(f"Model dtype: {model.dtype}")
    print(f"LoRA applied. Trainable params:")
    model.print_trainable_parameters()

    return model, tokenizer

def load_training_data(dataset_path=None, config=None):
    """Load training dataset."""
    if dataset_path is None:
        # Use a small example dataset for testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    else:
        dataset = load_dataset("text", data_files={"train": dataset_path})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def train(config_path="config.yaml", dataset_path=None, output_dir=None):
    """Fine-tune model with LoRA."""

    config = load_config(config_path)
    model, tokenizer = load_model_for_training(config)

    if output_dir is None:
        output_dir = config['training']['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    train_dataset = load_training_data(dataset_path, config)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],  # Disable wandb for CPU
        remove_unused_columns=False,
        dataloader_num_workers=0,  # CPU training
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Save memory
    )

    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    print(f"Training complete. Model saved to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer

if __name__ == "__main__":
    train()
