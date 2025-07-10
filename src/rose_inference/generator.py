"""Simplified model generation with streaming."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import torch
from transformers.generation.streamers import TextIteratorStreamer

logger = logging.getLogger(__name__)


def _format_messages(
    messages: Optional[List[Dict[str, Any]]], tokenizer: Any, prompt: Optional[str] = None
) -> Optional[str]:
    """Format messages and/or prompt into a single string, or return None if nothing to format."""
    if messages:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            formatted: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback formatting
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
            formatted = "\n\n".join(prompt_parts)

        if prompt:
            return f"{formatted}\n\n{prompt}"

        return formatted

    return prompt


async def _prepare_inputs(
    formatted_prompt: str,
    tokenizer: Any,
    model: Any,
) -> Tuple[Dict[str, Any], int]:
    """Tokenize and prepare inputs for model generation.

    Returns:
        Tuple of (inputs_dict, input_token_count)
    """
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_token_count = inputs["input_ids"].shape[1]
    return inputs, input_token_count


def _process_logprob(logprob: float) -> float:
    """Handle -inf values for JSON compatibility."""
    if logprob == float("-inf"):
        return -100.0
    return logprob


def _extract_top_logprobs(
    log_probs: torch.Tensor,
    top_k: int,
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Extract top k tokens with their log probabilities."""
    if top_k <= 0:
        return []

    top_k = min(top_k, log_probs.size(-1))
    top_log_probs, top_indices = torch.topk(log_probs, k=top_k)

    top_tokens_data = []
    for idx, logprob in zip(top_indices, top_log_probs):
        logprob_value = _process_logprob(logprob.item())
        top_token_text = tokenizer.decode(idx.item(), skip_special_tokens=False)
        top_tokens_data.append(
            {
                "token": top_token_text,
                "logprob": logprob_value,
                "bytes": list(top_token_text.encode("utf-8")),
            }
        )

    return top_tokens_data


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
    formatted_prompt = _format_messages(messages, tokenizer, prompt)

    if formatted_prompt is None:
        yield {"type": "complete", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return

    # Tokenize and prepare inputs
    inputs, input_token_count = await _prepare_inputs(formatted_prompt, tokenizer, model)
    yield {"type": "input_tokens_counted", "input_tokens": input_token_count}

    # Setup streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Build generation kwargs
    gen_kwargs = {
        **generation_kwargs,
        **inputs,
        "do_sample": generation_kwargs.get("temperature", 1.0) > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
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


async def generate_with_logprobs(
    model: Any,
    tokenizer: Any,
    messages: Optional[List[Dict[str, Any]]] = None,
    prompt: str = "",
    generation_kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Generate with logprobs using output_scores=True.

    Non-streaming generation that returns logprobs for each token.
    """
    generation_kwargs = generation_kwargs or {}

    # Extract logprobs parameters (remove them from generation_kwargs)
    generation_kwargs.pop("logprobs", False)
    top_logprobs = generation_kwargs.pop("top_logprobs", 0)

    # Format input
    formatted_prompt = _format_messages(messages, tokenizer, prompt)

    if formatted_prompt is None:
        yield {"type": "complete", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return

    # Tokenize and prepare inputs
    inputs, input_token_count = await _prepare_inputs(formatted_prompt, tokenizer, model)
    yield {"type": "input_tokens_counted", "input_tokens": input_token_count}

    # Build generation kwargs with extra parameters for logprobs
    gen_kwargs = {
        **generation_kwargs,
        **inputs,
        "do_sample": generation_kwargs.get("temperature", 1.0) > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True,
    }

    try:
        # Generate all at once
        with torch.no_grad():
            outputs = await asyncio.to_thread(model.generate, **gen_kwargs)

        # Extract generated tokens (skip input tokens)
        generated_ids = outputs.sequences[0][input_token_count:]

        # Process each token with its score
        for position, (token_id, score) in enumerate(zip(generated_ids, outputs.scores)):
            # Convert token_id tensor to integer
            token_id_int = token_id.item()

            # Decode token
            token_text = tokenizer.decode(token_id_int, skip_special_tokens=False)

            # Convert logits to log probabilities
            log_probs = torch.nn.functional.log_softmax(score[0], dim=-1)

            # Get log probability of the selected token
            token_logprob = _process_logprob(log_probs[token_id_int].item())

            # Get top alternatives if requested
            top_tokens_data = _extract_top_logprobs(log_probs, top_logprobs, tokenizer)

            yield {
                "type": "token",
                "token": token_text,
                "position": position,
                "logprob": token_logprob,
                "top_logprobs": top_tokens_data,
            }

        output_token_count = len(generated_ids)
        yield {
            "type": "complete",
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "total_tokens": input_token_count + output_token_count,
        }

    except Exception as e:
        logger.error(f"Generation with logprobs error: {e}")
        yield {"type": "error", "error": str(e)}
