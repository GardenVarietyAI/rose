import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Union, cast

from fastapi import APIRouter, Depends, HTTPException
from openai import APIConnectionError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, field_validator
from rose_server.dependencies import get_db_session, get_openai_client
from rose_server.llms import MODELS
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
    openai_client: AsyncOpenAI = Depends(get_openai_client),
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

    model = body.model if body.model and body.model != "default" else MODELS["chat"]["id"]

    kwargs: Dict[str, Any] = {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_tokens,
        "presence_penalty": body.presence_penalty,
        "frequency_penalty": body.frequency_penalty,
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

    messages = cast(List[ChatCompletionMessageParam], body.messages)

    try:
        response = await openai_client.chat.completions.create(messages=messages, model=model, **kwargs)
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail=f"LLM server unavailable: {str(e)}")

    user_msg = body.messages[-1]
    prompt = serialize_message_content(user_msg.get("content"))
    if user_msg["role"] == "user":
        session.add(
            Message(
                thread_id=thread_id,
                role="user",
                content=prompt,
                model=model,
            )
        )

    assistant_raw_content = response.choices[0].message.content
    if isinstance(assistant_raw_content, str):
        reasoning = extract_reasoning(assistant_raw_content)
        cleaned_content = strip_reasoning_tags(assistant_raw_content)
    else:
        reasoning = None
        cleaned_content = serialize_message_content(assistant_raw_content)

    assistant_meta: Dict[str, Any] = {
        "completion_id": response.id,
        "finish_reason": response.choices[0].finish_reason,
    }
    if response.usage:
        assistant_meta["usage"] = response.usage.model_dump()

    session.add(
        Message(
            thread_id=thread_id,
            role="assistant",
            content=cleaned_content,
            reasoning=reasoning,
            model=model,
            meta=assistant_meta,
        )
    )

    if not body.thread_id and len(body.messages) == 1:
        if body.messages[0]["role"] == "user" and prompt is not None:
            session.add(
                SearchEvent(
                    event_type="ask",
                    search_mode="llm",
                    query=prompt,
                    result_count=0,
                    thread_id=thread_id,
                )
            )

    response_dict: Dict[str, Any] = response.model_dump()
    response_dict["thread_id"] = thread_id
    return response_dict
