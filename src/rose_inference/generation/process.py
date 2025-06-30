"""Process inference requests - simple WebSocket-based inference."""

import asyncio
import atexit
import json
import logging
import threading
from typing import Any, Dict

from transformers import TextIteratorStreamer

from rose_core.config.service import MAX_CONCURRENT_INFERENCE
from rose_core.models import cleanup_model_memory, get_tokenizer, load_hf_model

logger = logging.getLogger(__name__)

# Global model cache
_models: Dict[str, Dict[str, Any]] = {}

# Semaphore for controlling concurrent inferences (default: 1 = sequential)
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)


def cleanup_on_exit():
    """Clean up models on exit."""
    logger.info("Shutting down inference server, cleaning up models...")
    _models.clear()
    cleanup_model_memory()
    logger.info("Cleanup complete")


# Register cleanup handler
atexit.register(cleanup_on_exit)


async def load_model(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer (cached)."""
    if model_name in _models:
        logger.info(f"Using cached model: {model_name}")
        return _models[model_name]

    logger.info(f"Loading model: {model_name}")
    try:
        # Use model_path if available (for custom/fine-tuned models), otherwise use model_name
        model_id = model_config.get("model_path") or model_config.get("model_name", model_name)
        model = load_hf_model(
            model_id=model_id,
            torch_dtype=model_config.get("torch_dtype"),
        )

        # Load tokenizer
        tokenizer = get_tokenizer(model_config.get("model_path") or model_config.get("model_name", model_name))

        _models[model_name] = {"model": model, "tokenizer": tokenizer, "config": model_config}

        logger.info(f"Successfully loaded model: {model_name}")
        return _models[model_name]

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


async def process_inference_request(websocket, request_data: Dict[str, Any]) -> None:
    """Process a single inference request and stream results back."""
    async with inference_semaphore:  # Simple concurrency control
        try:
            model_name = request_data["model_name"]
            model_config = request_data["config"]
            prompt = request_data["prompt"]
            generation_kwargs = request_data["generation_kwargs"]
            messages = request_data.get("messages")  # Optional messages for chat formatting

            # Load model
            model_info = await load_model(model_name, model_config)
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]

            # Use chat template if messages provided and tokenizer supports it
            if messages and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                logger.info(f"Using chat template for {len(messages)} messages")
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize input
            logger.info(f"Tokenizing prompt (length: {len(prompt)}): {prompt[:100]}...")
            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            logger.info(f"Input shape: {inputs['input_ids'].shape}")

            # Create streamer for token-by-token output
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            # Add streamer to generation kwargs
            generation_kwargs = {
                **generation_kwargs,
                **inputs,
                "streamer": streamer,
                "do_sample": generation_kwargs.get("temperature", 1.0) > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # Log generation parameters
            logger.info(
                f"Generation kwargs: max_new_tokens={generation_kwargs.get('max_new_tokens')}, "
                f"temperature={generation_kwargs.get('temperature')}, "
                f"do_sample={generation_kwargs.get('do_sample')}"
            )

            # Start generation in background thread (required for streaming)
            generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            generation_thread.start()

            # Stream tokens back
            position = 0
            try:
                logger.info("Starting to stream tokens...")
                for token in streamer:
                    if token:  # Skip empty tokens
                        logger.debug(f"Sending token {position}: {repr(token)}")
                        await websocket.send(json.dumps({"type": "token", "token": token, "position": position}))
                        position += 1
                logger.info(f"Finished streaming {position} tokens")
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                await websocket.send(json.dumps({"type": "error", "error": str(e)}))

            # Wait for generation to complete
            generation_thread.join()

            # Send completion
            await websocket.send(json.dumps({"type": "complete", "total_tokens": position}))

        except Exception as e:
            logger.error(f"Inference error: {e}")
            error_msg = {"type": "error", "error": str(e)}
            try:
                await websocket.send(json.dumps(error_msg))
            except Exception:
                pass  # Connection might be closed

        finally:
            # Cleanup memory after each inference
            cleanup_model_memory()
            logger.info("Inference completed")
