"""Simplified model generation with streaming."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from transformers.generation import TextIteratorStreamer

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
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


async def generate_stream(
    model: Any,
    tokenizer: Any,
    messages: Optional[List[Dict[str, Any]]] = None,
    prompt: str = "",
    generation_kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream generation events from a model.

    Yields:
        Dict events with types: input_tokens_counted, token, complete, error
    """
    generation_kwargs = generation_kwargs or {}

    # Format input
    if messages:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = _format_messages_fallback(messages)

        if prompt:
            formatted_prompt = f"{formatted_prompt}\n\n{prompt}"
    else:
        formatted_prompt = prompt

    if not formatted_prompt:
        yield {"type": "complete", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_token_count = inputs["input_ids"].shape[1]
    yield {"type": "input_tokens_counted", "input_tokens": input_token_count}

    # Setup streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **generation_kwargs,
        **inputs,
        "streamer": streamer,
        "do_sample": generation_kwargs.get("temperature", 1.0) > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Start generation
    generation_task = asyncio.create_task(asyncio.to_thread(model.generate, **gen_kwargs))

    # Stream tokens
    position = 0
    try:

        async def stream_tokens() -> AsyncIterator[str]:
            loop = asyncio.get_event_loop()
            while True:
                token = await loop.run_in_executor(None, lambda: next(streamer, None))
                if token is None:
                    break
                yield token

        async for token in stream_tokens():
            if token:
                yield {"type": "token", "token": token, "position": position}
                position += 1

    except Exception as e:
        logger.error(f"Generation error: {e}")
        yield {"type": "error", "error": str(e)}
        generation_task.cancel()
        return
    finally:
        timeout = config.get("inference_timeout", 120) if config else 120
        try:
            await asyncio.wait_for(generation_task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Generation timeout after {timeout}s")
            generation_task.cancel()

    yield {
        "type": "complete",
        "input_tokens": input_token_count,
        "output_tokens": position,
        "total_tokens": input_token_count + position,
    }
