import json
import logging
import re
import uuid
from typing import Any, Dict, Optional, Union, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from rose_server.dependencies import get_db_session, get_llama_client
from rose_server.models.messages import Message
from rose_server.models.search_events import SearchEvent
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


async def _thread_exists(session: AsyncSession, thread_id: str) -> bool:
    stmt = select(Message).where(col(Message.thread_id) == thread_id).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


def extract_reasoning(text: str) -> Optional[str]:
    """Extract <think>...</think> content and return reasoning."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def strip_reasoning_tags(text: str) -> Optional[str]:
    """Remove <think>...</think> tags from text."""
    result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return result if result else None


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    model: Optional[str] = None
    messages: list[dict[str, Any]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[Union[str, list[str]]] = None
    seed: Optional[int] = None
    response_format: Optional[dict[str, Any]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("messages must contain at least one entry")
        return value


def serialize_message_content(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    body: ChatRequest,
    llama_client: httpx.AsyncClient = Depends(get_llama_client),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    if body.stream:
        raise HTTPException(status_code=400, detail="Streaming unavailable.")

    if body.thread_id:
        thread_id = body.thread_id
        if not await _thread_exists(session, thread_id):
            raise HTTPException(status_code=404, detail="Thread not found")
    else:
        thread_id = str(uuid.uuid4())

    kwargs = body.model_dump(
        include={
            "temperature",
            "top_p",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "seed",
            "response_format",
            "tools",
            "tool_choice",
        },
        exclude_none=True,
    )

    messages = body.messages

    try:
        upstream = await llama_client.post(
            "chat/completions",
            json={"model": body.model, "messages": messages, "stream": False, **kwargs},
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"LLM server unavailable: {str(e)}")

    if upstream.status_code >= 400:
        raise HTTPException(status_code=upstream.status_code, detail=upstream.text)

    response = cast(Dict[str, Any], upstream.json())
    model_used = response.get("model")

    user_msg = body.messages[-1]
    prompt = serialize_message_content(user_msg.get("content"))
    if user_msg["role"] == "user":
        session.add(
            Message(
                thread_id=thread_id,
                role="user",
                content=prompt,
                model=model_used,
            )
        )
        await session.commit()

    choices = response.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="Upstream completion missing choices")

    assistant_raw_content = (choices[0].get("message") or {}).get("content")
    if isinstance(assistant_raw_content, str):
        reasoning = extract_reasoning(assistant_raw_content)
        cleaned_content = strip_reasoning_tags(assistant_raw_content)
    else:
        reasoning = None
        cleaned_content = serialize_message_content(assistant_raw_content)

    assistant_meta: Dict[str, Any] = {
        "completion_id": response.get("id"),
        "finish_reason": choices[0].get("finish_reason"),
    }
    usage = response.get("usage")
    if usage is not None:
        assistant_meta["usage"] = usage

    assistant_message = Message(
        thread_id=thread_id,
        role="assistant",
        content=cleaned_content,
        reasoning=reasoning,
        model=model_used,
        meta=assistant_meta,
    )
    session.add(assistant_message)

    if prompt is not None and not body.thread_id:
        if len(body.messages) == 1 and body.messages[0]["role"] == "user":
            session.add(
                SearchEvent(
                    event_type="ask",
                    search_mode="llm",
                    query=prompt,
                    result_count=0,
                    thread_id=thread_id,
                )
            )

    response_dict: Dict[str, Any] = {
        **response,
        "message_uuid": assistant_message.uuid,
        "thread_id": thread_id,
    }
    return response_dict
