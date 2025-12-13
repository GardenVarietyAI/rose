import json
import logging
import re
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, cast

import llama_cpp
from fastapi import APIRouter, Request
from llama_cpp.llama_types import ChatCompletionRequestMessage
from pydantic import BaseModel
from rose_server.models.messages import Message
from sse_starlette import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


def extract_reasoning(text: str) -> Optional[str]:
    """Extract <think>...</think> content and return reasoning."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def strip_reasoning_tags(text: str) -> str:
    """Remove <think>...</think> tags from text."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


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
    messages: List[ChatCompletionRequestMessage],
    kwargs: Dict[str, Any],
    thread_id: str,
    model: str,
    get_db_session: Any,
) -> AsyncIterator[str]:
    """Stream chat completion chunks and save to database."""
    accumulated_content = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    completion_id: Optional[str] = None

    try:
        stream = cast(Iterator[Dict[str, Any]], llama.create_chat_completion(messages=messages, stream=True, **kwargs))
        for chunk in stream:
            if chunk.get("id"):
                completion_id = chunk["id"]
            if chunk["choices"][0]["delta"].get("content"):
                accumulated_content += chunk["choices"][0]["delta"]["content"]
            if chunk["choices"][0].get("finish_reason"):
                finish_reason = chunk["choices"][0]["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]
            yield json.dumps(chunk)

        async with get_db_session() as session:
            user_msg = messages[-1]
            if user_msg["role"] == "user":
                user_message = Message(
                    thread_id=thread_id,
                    role="user",
                    content=user_msg["content"],
                    model=model,
                )
                session.add(user_message)

            reasoning = extract_reasoning(accumulated_content)
            cleaned_content = strip_reasoning_tags(accumulated_content)

            assistant_meta: Dict[str, Any] = {"finish_reason": finish_reason}
            if completion_id:
                assistant_meta["completion_id"] = completion_id
            if usage:
                assistant_meta["usage"] = usage

            assistant_msg = Message(
                thread_id=thread_id,
                role="assistant",
                content=cleaned_content,
                reasoning=reasoning,
                model=model,
                meta=assistant_meta,
            )
            session.add(assistant_msg)

        yield "[DONE]"

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

    messages = cast(List[ChatCompletionRequestMessage], body.messages)

    if body.stream:
        return EventSourceResponse(
            stream_chat_completion(
                llama=llama,
                messages=messages,
                kwargs=kwargs,
                thread_id=thread_id,
                model=body.model,
                get_db_session=request.app.state.get_db_session,
            ),
            media_type="text/event-stream",
        )

    response = cast(Dict[str, Any], llama.create_chat_completion(messages=messages, **kwargs))

    async with request.app.state.get_db_session() as session:
        user_msg = body.messages[-1]
        if user_msg["role"] == "user":
            user_message = Message(
                thread_id=thread_id,
                role="user",
                content=user_msg["content"],
                model=body.model,
            )
            session.add(user_message)

        assistant_content = response["choices"][0]["message"]["content"]
        reasoning = extract_reasoning(assistant_content)
        cleaned_content = strip_reasoning_tags(assistant_content)

        assistant_meta: Dict[str, Any] = {
            "completion_id": response["id"],
            "finish_reason": response["choices"][0]["finish_reason"],
        }
        if response.get("usage"):
            assistant_meta["usage"] = response["usage"]

        assistant_msg = Message(
            thread_id=thread_id,
            role="assistant",
            content=cleaned_content,
            reasoning=reasoning,
            model=body.model,
            meta=assistant_meta,
        )
        session.add(assistant_msg)

    return response
