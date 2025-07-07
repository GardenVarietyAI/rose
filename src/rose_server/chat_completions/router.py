"""Event-based router for OpenAI-compatible chat completions API.
This replaces the existing router with our event-native system while
maintaining 100% API compatibility.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from rose_core.config.settings import settings
from rose_server.events.event_types import LLMEvent
from rose_server.events.formatters import ChatCompletionsFormatter
from rose_server.events.generator import EventGenerator
from rose_server.models.store import get as get_language_model
from rose_server.schemas.chat import ChatMessage, ChatRequest

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


def _prepare_tool_params(request: ChatRequest) -> Dict[str, Any]:
    """Extract tool parameters from request and log them."""
    enable_tools = bool(request.tools)
    return {"enable_tools": enable_tools, "tools": request.tools, "tool_choice": request.tool_choice}


@router.post("/chat/completions", response_model=None)
async def event_based_chat_completions(
    request: ChatRequest = Body(...),
) -> JSONResponse | EventSourceResponse:
    """Event-based endpoint for chat completions."""
    model = await get_language_model(request.model)
    if not model:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Model '{request.model}' not found",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    logger.info(f"[EVENT] Using model: {request.model}")
    messages = request.messages
    logger.info(f"[EVENT] Message count: {len(messages)}")

    # Validate tools if provided
    if request.tools:
        for tool in request.tools:
            tool_type = tool.get("type") if isinstance(tool, dict) else getattr(tool, "type", None)
            if tool_type in ["code_interpreter", "web_search"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": f"Tool type '{tool_type}' is not supported. Please use function tools instead.",
                            "type": "invalid_request_error",
                            "param": "tools",
                            "code": "unsupported_tool_type",
                        }
                    },
                )

    try:
        # Build config from model
        config = {
            "model_name": model.model_name,
            "model_type": model.model_type,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "memory_gb": model.memory_gb,
        }

        if model.is_fine_tuned and model.path:
            config["model_path"] = str(Path(settings.data_dir) / model.path)
            config["base_model"] = model.parent
            config["is_fine_tuned"] = True

        if model.get_lora_modules():
            config["lora_target_modules"] = model.get_lora_modules()
        generator = EventGenerator(request.model, config)
        formatter = ChatCompletionsFormatter()
        logger.info("[EVENT] Using ChatCompletionsGenerator for chat completions")
        if request.stream:
            return await create_event_streaming_response(generator, messages, formatter, request)
        else:
            return await create_event_complete_response(generator, messages, formatter, request)
    except Exception as e:
        logger.error(f"[EVENT] Error in chat completions: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})


async def create_event_streaming_response(
    generator: EventGenerator,
    messages: list[ChatMessage],
    formatter: ChatCompletionsFormatter,
    request: ChatRequest,
) -> EventSourceResponse:
    """Create streaming response using events and sse-starlette."""

    async def generate() -> AsyncGenerator[Dict[str, Any], None]:
        """Generate SSE events from LLM events."""
        try:
            tool_params = _prepare_tool_params(request)
            async for event in generator.generate_events(
                messages, temperature=request.temperature, max_tokens=request.max_tokens, **tool_params
            ):
                chunk = formatter.format_event(event)
                if chunk:
                    yield {"data": chunk.json()}
            yield {"data": "[DONE]"}
        except Exception as e:
            logger.error(f"[EVENT] Streaming error: {str(e)}")
            error_chunk = {
                "id": f"chatcmpl-error-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": f"Error: {str(e)}"}, "finish_reason": "stop"}],
            }
            yield {"data": json.dumps(error_chunk)}

    return EventSourceResponse(generate(), media_type="text/plain")


async def create_event_complete_response(
    generator: EventGenerator,
    messages: list[ChatMessage],
    formatter: ChatCompletionsFormatter,
    request: ChatRequest,
) -> JSONResponse:
    """Create complete (non-streaming) response from events."""
    try:
        tool_params = _prepare_tool_params(request)
        all_events: list[LLMEvent] = []
        async for event in generator.generate_events(
            messages, temperature=request.temperature, max_tokens=request.max_tokens, **tool_params
        ):
            all_events.append(event)
        complete_response = formatter.format_complete_response(all_events)
        complete_response["model"] = request.model
        content_length = 0
        if complete_response.get("choices") and len(complete_response["choices"]) > 0:
            content = complete_response["choices"][0].get("message", {}).get("content")
            content_length = len(content) if content else 0
        logger.info(f"[EVENT] Generated {len(all_events)} events, response length: {content_length}")
        return JSONResponse(content=complete_response)
    except Exception as e:
        logger.error(f"[EVENT] Complete response error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error generating response: {str(e)}"})
