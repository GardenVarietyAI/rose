import json
import logging
from typing import Any, AsyncIterator, Dict, Union

import llama_cpp
from fastapi import APIRouter, HTTPException, Request
from llama_cpp.server.types import CreateChatCompletionRequest
from sse_starlette import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


async def chat_completion_stream(
    llama: llama_cpp.Llama,
    messages: list[Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> AsyncIterator[str]:
    """Stream chat completion chunks as SSE events."""
    try:
        stream = llama.create_chat_completion(messages=messages, stream=True, **kwargs)  # type: ignore[arg-type]
        for chunk in stream:
            yield json.dumps(chunk)
        yield "[DONE]"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"error": str(e)}
        yield json.dumps(error_chunk)


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequest,
) -> Union[llama_cpp.ChatCompletion, EventSourceResponse]:
    llama: llama_cpp.Llama = request.app.state.chat_model

    kwargs: Dict[str, Any] = {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "top_k": body.top_k,
        "min_p": body.min_p,
        "max_tokens": body.max_tokens,
        "presence_penalty": body.presence_penalty,
        "frequency_penalty": body.frequency_penalty,
        "repeat_penalty": body.repeat_penalty,
        "stop": body.stop,
        "seed": body.seed,
    }

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if body.response_format is not None:
        kwargs["response_format"] = body.response_format

    if body.tools is not None:
        kwargs["tools"] = body.tools

    if body.tool_choice is not None:
        kwargs["tool_choice"] = body.tool_choice

    messages = [dict(msg) for msg in body.messages]

    if body.stream:
        return EventSourceResponse(
            chat_completion_stream(llama, messages, kwargs),
            media_type="text/event-stream",
        )

    try:
        response: llama_cpp.ChatCompletion = llama.create_chat_completion(messages=messages, **kwargs)  # type: ignore[arg-type,assignment]
        return response
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
