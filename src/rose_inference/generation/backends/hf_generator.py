"""HuggingFace model generation logic."""

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from transformers import TextIteratorStreamer

from rose_core.config.service import INFERENCE_TIMEOUT

logger = logging.getLogger(__name__)


def _format_messages_fallback(messages: List[Dict[str, Any]]) -> str:
    """Fallback formatting when tokenizer doesn't have a chat template."""
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
    return "\n\n".join(prompt_parts)


async def generate_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    messages: Optional[List[Dict[str, Any]]],
    generation_kwargs: Dict[str, Any],
    websocket: Any,
    stream_id: str,
) -> Dict[str, int]:
    """Run model generation and stream tokens to websocket.

    Returns:
        Dictionary with input_tokens and output_tokens counts.
    """
    # Handle different input combinations
    if messages:
        # Format messages using chat template or fallback
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            logger.info(f"[{stream_id}] Using chat template for {len(messages)} messages")
            formatted_messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            logger.info(f"[{stream_id}] No chat template, using fallback formatting for {len(messages)} messages")
            formatted_messages = _format_messages_fallback(messages)

        # If we have both messages and prompt, prepend messages as context
        if prompt:
            logger.info(f"[{stream_id}] Combining message history with prompt")
            prompt = f"{formatted_messages}\n\n{prompt}"
        else:
            prompt = formatted_messages

    # Tokenize input
    logger.info(f"[{stream_id}] Tokenizing prompt (length: {len(prompt)}): {prompt[:100]}...")
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get input token count
    input_token_count = inputs["input_ids"].shape[1]
    logger.info(f"[{stream_id}] Input shape: {inputs['input_ids'].shape}, tokens: {input_token_count}")

    # Send input token count event immediately after tokenization
    await websocket.send(json.dumps({"type": "input_tokens_counted", "input_tokens": input_token_count}))

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

    return {"input_tokens": input_token_count, "output_tokens": position}
