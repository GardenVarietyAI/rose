"""Model loading and subprocess worker for inference."""

import asyncio
import json
import logging
import sys
from typing import Any, Callable, Dict

from rose_core.models import get_tokenizer, load_hf_model
from rose_inference.generation.backends.hf_generator import generate_stream

logger = logging.getLogger(__name__)


async def load_model(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer for inference."""
    logger.info(f"Loading model: {model_name}")

    loader: Callable[..., Any] = model_config.get("loader", load_hf_model)

    # Use model_path if available (for custom/fine-tuned models), otherwise use model_name
    model_id = model_config.get("model_path") or model_config.get("model_name", model_name)

    # Load the model based on loader type
    if loader.__name__ == "load_peft_model":
        # PEFT models require model_path
        model = loader(
            model_id=model_id,
            model_path=model_config.get("model_path"),
            torch_dtype=model_config.get("torch_dtype"),
        )
    else:
        # Regular HF models only need model_id
        model = loader(
            model_id=model_id,
            torch_dtype=model_config.get("torch_dtype"),
        )

    # Load tokenizer
    tokenizer = get_tokenizer(model_id)

    # Return model info
    return {
        "name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "config": model_config,
        "device": str(model.device) if hasattr(model, "device") else "cpu",
        "dtype": str(next(model.parameters()).dtype) if hasattr(model, "parameters") else "unknown",
    }


# Export for compatibility
def cleanup_models() -> None:
    """No-op for compatibility."""
    pass


# Subprocess worker functionality
class StdoutAdapter:
    """Adapter to send JSON lines to stdout."""

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Send data as JSON line to stdout."""
        try:
            print(json.dumps(data), flush=True)
        except (BrokenPipeError, IOError) as e:
            logger.error(f"Failed to write to stdout: {e}")
            raise


async def run_subprocess_inference() -> None:
    """Run inference as subprocess worker - reads from stdin, writes to stdout."""
    # Configure logging to stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    try:
        # Read request from stdin
        logger.info("Reading request from stdin...")
        try:
            stdin_data = sys.stdin.read()
            if not stdin_data:
                logger.error("No input data received from stdin")
                print(json.dumps({"type": "error", "error": "No input data received"}), flush=True)
                sys.exit(1)
            request = json.loads(stdin_data)
        except (IOError, OSError) as e:
            logger.error(f"Failed to read from stdin: {e}")
            print(json.dumps({"type": "error", "error": "Failed to read input"}), flush=True)
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            print(json.dumps({"type": "error", "error": "Invalid request format"}), flush=True)
            sys.exit(1)

        stream_id = request.get("stream_id", "unknown")

        logger.info(f"[{stream_id}] Starting inference in subprocess")

        # Extract request data
        model_name = request["model_name"]
        model_config = request["config"]
        generation_kwargs = request["generation_kwargs"]
        messages = request.get("messages")
        prompt = request.get("prompt", "")

        logger.info(f"[{stream_id}] Model: {model_name}, Generation kwargs: {generation_kwargs}")

        # Load model
        logger.info(f"[{stream_id}] Loading model: {model_name}")
        model_info = await load_model(model_name, model_config)

        # Create adapter
        adapter = StdoutAdapter()

        # Run generation
        logger.info(f"[{stream_id}] Starting generation")
        token_counts = await generate_stream(
            model=model_info["model"],
            tokenizer=model_info["tokenizer"],
            prompt=prompt,
            messages=messages,
            generation_kwargs=generation_kwargs,
            websocket=adapter,
            stream_id=stream_id,
        )

        # Send completion
        await adapter.send_json(
            {
                "type": "complete",
                "input_tokens": token_counts["input_tokens"],
                "output_tokens": token_counts["output_tokens"],
                "total_tokens": token_counts["input_tokens"] + token_counts["output_tokens"],
            }
        )

        logger.info(f"[{stream_id}] Inference completed successfully")
        # Explicitly exit with success
        sys.exit(0)

    except Exception as e:
        logger.error(f"Inference error: {e}")
        try:
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)
        except Exception:
            pass  # Best effort error reporting
        sys.exit(1)


# Entry point for subprocess
if __name__ == "__main__":
    asyncio.run(run_subprocess_inference())
