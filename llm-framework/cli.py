#!/usr/bin/env python3
"""
Command-line interface for CPU-only LLM framework.
"""

import argparse
import sys
import os
from models.load_model import load_model_and_tokenizer
from models.inference import CPUInferenceEngine
from safety.remove_guardrails import SafetyBypass, UnrestrictedPrompting

def main():
    parser = argparse.ArgumentParser(
        description="CPU-Only LLM Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py generate "What is AI?" --temperature 0.7
  python cli.py generate "Tell a story" --unrestricted --mode creative
  python cli.py serve --host 0.0.0.0 --port 5000
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from prompt")
    gen_parser.add_argument("prompt", help="Input prompt")
    gen_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    gen_parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    gen_parser.add_argument("--stream", action="store_true", help="Stream output")
    gen_parser.add_argument("--unrestricted", action="store_true", help="Use unrestricted mode")
    gen_parser.add_argument("--mode", default="general", help="Unrestricted mode type")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start REST API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=5000, help="Server port")

    # Fine-tune command
    tune_parser = subparsers.add_parser("finetune", help="Fine-tune model")
    tune_parser.add_argument("--dataset", help="Path to training dataset")
    tune_parser.add_argument("--output", default="./fine_tuned_models", help="Output directory")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test model loading")

    args = parser.parse_args()

    if args.command == "generate":
        generate_command(args)
    elif args.command == "serve":
        serve_command(args)
    elif args.command == "finetune":
        finetune_command(args)
    elif args.command == "test":
        test_command()
    else:
        parser.print_help()

def generate_command(args):
    """Handle generate command."""
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    engine = CPUInferenceEngine(model, tokenizer)

    prompt = args.prompt

    if args.unrestricted:
        print(f"Using unrestricted mode: {args.mode}")
        safety = SafetyBypass(model, tokenizer)
        prompt = safety.prepare_prompt_for_unrestricted(prompt, args.mode)

    print("\n" + "="*50)
    print(f"Prompt: {args.prompt}")
    print("="*50 + "\n")

    if args.stream:
        print("Streaming output (not fully implemented)...")
        result = engine.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=False
        )
        print(result)
    else:
        result = engine.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=False
        )
        print(result)

    print("\n" + "="*50)

def serve_command(args):
    """Handle serve command."""
    print(f"Starting server on {args.host}:{args.port}...")
    from api.server import app

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=False
    )

def finetune_command(args):
    """Handle fine-tune command."""
    print("Starting fine-tuning...")
    from training.fine_tune import train

    train(
        dataset_path=args.dataset,
        output_dir=args.output
    )

def test_command():
    """Test model loading."""
    print("Testing model loading...")

    try:
        print("Loading tokenizer and model...")
        model, tokenizer = load_model_and_tokenizer()
        from models.load_model import get_model_size

        print(f"✓ Model loaded successfully")
        print(f"  Model size: {get_model_size(model):.2f} GB")
        print(f"  Device: {model.device}")
        print(f"  Dtype: {model.dtype}")

        print("\nTesting inference...")
        engine = CPUInferenceEngine(model, tokenizer)

        test_prompt = "Hello, how are you?"
        result = engine.generate(test_prompt, max_tokens=50)

        print(f"✓ Inference successful")
        print(f"  Input: {test_prompt}")
        print(f"  Output: {result[:100]}...")

        print("\nTesting unrestricted mode...")
        safety = SafetyBypass(model, tokenizer)
        unrestricted_prompt = safety.prepare_prompt_for_unrestricted(
            test_prompt, "general"
        )
        result = engine.generate(unrestricted_prompt, max_tokens=50)

        print(f"✓ Unrestricted mode working")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
