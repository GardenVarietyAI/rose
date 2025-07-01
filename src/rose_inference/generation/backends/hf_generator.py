"""HuggingFace model generation logic."""

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from transformers import TextIteratorStreamer

from rose_core.config.service import INFERENCE_TIMEOUT

logger = logging.getLogger(__name__)


async def generate_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    messages: Optional[List[Dict[str, Any]]],
    generation_kwargs: Dict[str, Any],
    websocket: Any,
    stream_id: str,
) -> int:
    """Run model generation and stream tokens to websocket.

    Returns:
        Total number of tokens generated.
    """
    # Use chat template if messages provided and tokenizer supports it
    if messages and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        logger.info(f"[{stream_id}] Using chat template for {len(messages)} messages")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif messages and not prompt:
        # Fallback: format messages if no chat template available
        logger.info(f"[{stream_id}] No chat template, using fallback formatting for {len(messages)} messages")
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")  # Add generation prompt
        prompt = "\n\n".join(prompt_parts)

    # Tokenize input
    logger.info(f"[{stream_id}] Tokenizing prompt (length: {len(prompt)}): {prompt[:100]}...")
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    logger.info(f"[{stream_id}] Input shape: {inputs['input_ids'].shape}")

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
        f"[{stream_id}] Generation kwargs: max_new_tokens={generation_kwargs.get('max_new_tokens')}, "
        f"temperature={generation_kwargs.get('temperature')}, "
        f"do_sample={generation_kwargs.get('do_sample')}"
    )

    # Start generation in background thread (required for streaming)
    generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    generation_thread.start()

    # Stream tokens back
    position = 0
    try:
        logger.info(f"[{stream_id}] Starting to stream tokens...")
        for token in streamer:
            if token:  # Skip empty tokens
                logger.debug(f"[{stream_id}] Sending token {position}: {repr(token)}")
                await websocket.send(json.dumps({"type": "token", "token": token, "position": position}))
                position += 1
        logger.info(f"[{stream_id}] Finished streaming {position} tokens")
    except Exception as e:
        logger.error(f"[{stream_id}] Error during streaming: {e}")
        await websocket.send(json.dumps({"type": "error", "error": str(e)}))
        raise

    # Wait for generation to complete with timeout
    generation_thread.join(timeout=INFERENCE_TIMEOUT)
    if generation_thread.is_alive():
        logger.error(f"[{stream_id}] Generation thread still alive after {INFERENCE_TIMEOUT}s timeout")

    return position
