"""
REST API server for CPU-only LLM inference.
"""

from flask import Flask, request, jsonify, stream_with_context, Response
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.load_model import load_model_and_tokenizer
from models.inference import CPUInferenceEngine
from safety.remove_guardrails import SafetyBypass, UnrestrictedPrompting

app = Flask(__name__)

# Global model and inference engine
model = None
tokenizer = None
engine = None
safety = None

def init_model():
    """Initialize model on first request."""
    global model, tokenizer, engine, safety

    if model is None:
        print("Loading model...")
        model, tokenizer = load_model_and_tokenizer()
        engine = CPUInferenceEngine(model, tokenizer)
        safety = SafetyBypass(model, tokenizer)
        print("Model loaded and ready!")

@app.before_request
def before_request():
    init_model()

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": "cpu"
    })

@app.route("/model/info", methods=["GET"])
def model_info():
    """Get model information."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    from models.load_model import get_model_size

    return jsonify({
        "device": str(model.device),
        "dtype": str(model.dtype),
        "estimated_size_gb": get_model_size(model),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    })

@app.route("/generate", methods=["POST"])
def generate():
    """Generate text from prompt."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()

    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    prompt = data["prompt"]
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)
    stream = data.get("stream", False)

    # Optional: mode for unrestricted generation
    unrestricted_mode = data.get("unrestricted_mode")
    if unrestricted_mode:
        prompt = safety.prepare_prompt_for_unrestricted(prompt, unrestricted_mode)

    try:
        if stream:
            streamer, thread = engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            def generate_stream():
                for token in streamer:
                    yield f"data: {token}\n"

            thread.join()
            return Response(
                stream_with_context(generate_stream()),
                mimetype="text/event-stream"
            )
        else:
            result = engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )

            return jsonify({
                "prompt": data["prompt"],
                "generated_text": result,
                "mode": "unrestricted" if unrestricted_mode else "standard"
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate/unrestricted", methods=["POST"])
def generate_unrestricted():
    """Generate with unrestricted system prompt."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()

    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    prompt = data["prompt"]
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)
    mode = data.get("mode", "general")  # creative, research, roleplay, general

    # Prepare unrestricted prompt
    prepared_prompt = safety.prepare_prompt_for_unrestricted(prompt, mode)

    try:
        result = engine.generate(
            prepared_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )

        return jsonify({
            "original_prompt": prompt,
            "unrestricted_mode": mode,
            "generated_text": result,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/safety/modes", methods=["GET"])
def safety_modes():
    """Get available unrestricted modes."""
    return jsonify({
        "available_modes": list(SafetyBypass.UNRESTRICTED_PROMPTS.keys()),
        "jailbreak_patterns": list(UnrestrictedPrompting.JAILBREAK_PATTERNS.keys())
    })

@app.route("/batch/generate", methods=["POST"])
def batch_generate():
    """Generate for multiple prompts."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()

    if not data or "prompts" not in data:
        return jsonify({"error": "Missing 'prompts' field"}), 400

    prompts = data["prompts"]
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)

    if not isinstance(prompts, list):
        return jsonify({"error": "'prompts' must be a list"}), 400

    try:
        results = engine.batch_generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return jsonify({
            "count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    """API documentation."""
    return jsonify({
        "name": "CPU-Only LLM API",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/model/info": "GET - Model information",
            "/generate": "POST - Generate text from prompt",
            "/generate/unrestricted": "POST - Generate with unrestricted mode",
            "/safety/modes": "GET - List available safety modes",
            "/batch/generate": "POST - Generate for multiple prompts",
        }
    })

if __name__ == "__main__":
    print("Starting CPU-only LLM API server...")
    print("Server running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
