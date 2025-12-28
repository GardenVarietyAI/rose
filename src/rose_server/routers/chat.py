import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException

from rose_server.dependencies import get_llama_client
from rose_server.schemas.chat import ChatRequest
from rose_server.services.llama import (
    LlamaStatusError,
    LlamaUnavailableError,
    request_chat_completion_json,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    body: ChatRequest,
    llama_client: httpx.AsyncClient = Depends(get_llama_client),
) -> dict[str, Any]:
    if body.stream:
        raise HTTPException(status_code=400, detail="Streaming unavailable.")

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

    payload: dict[str, Any] = {
        "messages": body.messages,
        "stream": False,
        **kwargs,
    }

    requested_model = body.model.strip() if body.model else ""
    if requested_model and requested_model != "default":
        payload["model"] = requested_model

    try:
        return await request_chat_completion_json(llama_client, payload)
    except LlamaUnavailableError as e:
        raise HTTPException(status_code=503, detail=f"LLM server unavailable: {e}") from e
    except LlamaStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=e.text) from e
