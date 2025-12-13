import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, Optional, Union

import llama_cpp
from fastapi import APIRouter, Request
from pydantic import BaseModel
from rose_server.models.messages import Message
from sse_starlette import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    model: str
    messages: list[dict[str, Any]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repeat_penalty: Optional[float] = None
    stop: Optional[Union[str, list[str]]] = None
    seed: Optional[int] = None
    response_format: Optional[dict[str, Any]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    stream: bool = False


async def stream_chat_completion(
    llama: llama_cpp.Llama,
    messages: list[Dict[str, Any]],
    kwargs: Dict[str, Any],
    thread_id: str,
    model: str,
    get_db_session: Any,
    input_messages: list[dict[str, Any]],
) -> AsyncIterator[str]:
    """Stream chat completion chunks and save to database."""
    accumulated_content = ""
    finish_reason = None

    try:
        stream = llama.create_chat_completion(messages=messages, stream=True, **kwargs)  # type: ignore[arg-type]
        for chunk in stream:
            if chunk["choices"][0]["delta"].get("content"):
                accumulated_content += chunk["choices"][0]["delta"]["content"]
            if chunk["choices"][0].get("finish_reason"):
                finish_reason = chunk["choices"][0]["finish_reason"]
            yield json.dumps(chunk)
        yield "[DONE]"

        async with get_db_session() as session:
            for msg in input_messages:
                message = Message(
                    thread_id=thread_id,
                    role=msg["role"],
                    content=[{"type": "text", "text": msg["content"]}],
                    meta={"model": model},
                )
                session.add(message)

            assistant_msg = Message(
                thread_id=thread_id,
                role="assistant",
                content=[{"type": "text", "text": accumulated_content}],
                meta={"model": model, "finish_reason": finish_reason},
            )
            session.add(assistant_msg)

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield json.dumps({"error": str(e)})


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: Request,
    body: ChatRequest,
) -> Union[dict[str, Any], EventSourceResponse]:
    llama: llama_cpp.Llama = request.app.state.chat_model

    thread_id = body.thread_id or str(uuid.uuid4())

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

    messages = body.messages

    if body.stream:
        return EventSourceResponse(
            stream_chat_completion(
                llama=llama,
                messages=messages,
                kwargs=kwargs,
                thread_id=thread_id,
                model=body.model,
                get_db_session=request.app.state.get_db_session,
                input_messages=body.messages,
            ),
            media_type="text/event-stream",
        )

    response: llama_cpp.ChatCompletion = llama.create_chat_completion(messages=messages, **kwargs)  # type: ignore[arg-type,assignment]

    async with request.app.state.get_db_session() as session:
        for msg in body.messages:
            message = Message(
                thread_id=thread_id,
                role=msg["role"],
                content=[{"type": "text", "text": msg["content"]}],
                meta={"model": body.model},
            )
            session.add(message)

        assistant_content = response["choices"][0]["message"]["content"]
        assistant_msg = Message(
            thread_id=thread_id,
            role="assistant",
            content=[{"type": "text", "text": assistant_content}],
            meta={"model": body.model, "finish_reason": response["choices"][0]["finish_reason"]},
        )
        session.add(assistant_msg)

    return response
