import torch
import yaml
from transformers import TextIteratorStreamer
from threading import Thread
import time

class CPUInferenceEngine:
    """CPU-optimized inference engine for LLM."""

    def __init__(self, model, tokenizer, config_path="config.yaml"):
        self.model = model
        self.tokenizer = tokenizer

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.inference_config = self.config.get('inference', {})
        self.device = "cpu"

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None,
                 top_k=None, repetition_penalty=None, stream=False):
        """Generate text from prompt with CPU optimization."""

        if max_tokens is None:
            max_tokens = self.inference_config.get('max_tokens', 512)
        if temperature is None:
            temperature = self.inference_config.get('temperature', 0.7)
        if top_p is None:
            top_p = self.inference_config.get('top_p', 0.95)
        if top_k is None:
            top_k = self.inference_config.get('top_k', 40)
        if repetition_penalty is None:
            repetition_penalty = self.inference_config.get('repetition_penalty', 1.1)

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        print(f"Input tokens: {inputs['input_ids'].shape[1]}")

        if stream:
            return self._generate_streaming(
                inputs, max_tokens, temperature, top_p,
                top_k, repetition_penalty
            )
        else:
            return self._generate_batch(
                inputs, max_tokens, temperature, top_p,
                top_k, repetition_penalty
            )

    def _generate_batch(self, inputs, max_tokens, temperature, top_p, top_k, repetition_penalty):
        """Batch generation (non-streaming)."""

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text

    def _generate_streaming(self, inputs, max_tokens, temperature, top_p, top_k, repetition_penalty):
        """Streaming generation with token-by-token output."""

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run generation in background thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer, thread

    def batch_generate(self, prompts, max_tokens=None, temperature=None):
        """Generate for multiple prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Generating [{i+1}/{len(prompts)}]...")
            result = self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            results.append(result)
        return results
