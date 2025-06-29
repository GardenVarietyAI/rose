"""Process inference requests - the actual model inference logic."""

import json
import logging
import traceback
from threading import Thread
from typing import Any, Dict

from transformers import TextIteratorStreamer

from rose_core.models import cleanup_model_memory, get_tokenizer, load_hf_model

logger = logging.getLogger(__name__)

# Global model cache
_models: Dict[str, Dict[str, Any]] = {}


async def load_model(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer."""
    if model_name in _models:
        logger.info(f"Using cached model: {model_name}")
        return _models[model_name]

    logger.info(f"Loading model: {model_name}")
    try:
        # Load model with config
        # Use model_path if available (for custom models), otherwise use model_name
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
    try:
        model_name = request_data["model_name"]
        model_config = request_data["config"]
        prompt = request_data["prompt"]
        generation_kwargs = request_data["generation_kwargs"]

        # Load model
        model_info = await load_model(model_name, model_config)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]

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

        # Start generation in background thread
        generation_thread = Thread(target=model.generate, kwargs=generation_kwargs)
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
        generation_thread.join(timeout=30)

        # Send completion
        await websocket.send(json.dumps({"type": "complete", "total_tokens": position}))

    except Exception as e:
        logger.error(f"Inference error: {e}")
        logger.error(traceback.format_exc())
        error_msg = {"type": "error", "error": str(e), "traceback": traceback.format_exc()}
        try:
            await websocket.send(json.dumps(error_msg))
        except Exception:
            pass  # Connection might be closed

    finally:
        # Cleanup memory after each inference
        cleanup_model_memory()
        logger.info("Inference completed")
