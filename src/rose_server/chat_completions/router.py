"""Event-based router for OpenAI-compatible chat completions API.
This replaces the existing router with our event-native system while
maintaining 100% API compatibility.
"""

import json
import logging
import time
import uuid

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from rose_server.events import LLMEvent
from rose_server.events.formatters import ChatCompletionsFormatter
from rose_server.events.generators import ChatCompletionsGenerator
from rose_server.llms.huggingface_llm import HuggingFaceLLM
from rose_server.schemas.chat import ChatMessage, ChatRequest
from rose_server.services import get_model_registry

router = APIRouter()
logger = logging.getLogger(__name__)


def _prepare_tool_params(request: ChatRequest, context: str = "") -> dict:
    """Extract tool parameters from request and log them."""
    enable_tools = bool(request.tools)
    tool_count = len(request.tools) if request.tools else 0
    logger.info(f"Chat completions{context}: enable_tools={enable_tools}, tools={tool_count}")
    return {"enable_tools": enable_tools, "tools": request.tools, "tool_choice": request.tool_choice}


@router.post("/v1/chat/completions", response_model=None)
async def event_based_chat_completions(
    request: ChatRequest = Body(...),
) -> JSONResponse | EventSourceResponse:
    """Event-based endpoint for chat completions."""
    registry = get_model_registry()
    if request.model not in registry.list_models():
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
    try:
        registry = get_model_registry()
        config = registry.get_model_config(request.model)
        if not config:
            return JSONResponse(status_code=400, content={"error": f"Model {request.model} not available"})
        try:
            base_llm = HuggingFaceLLM(config)
        except Exception as e:
            logger.error(f"Failed to create model {request.model}: {e}")
            return JSONResponse(status_code=500, content={"error": f"Failed to load model: {str(e)}"})
        generator = ChatCompletionsGenerator(base_llm)
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
    generator: ChatCompletionsGenerator,
    messages: list[ChatMessage],
    formatter: ChatCompletionsFormatter,
    request: ChatRequest,
) -> EventSourceResponse:
    """Create streaming response using events and sse-starlette."""

    async def generate():
        """Generate SSE events from LLM events."""
        try:
            tool_params = _prepare_tool_params(request, " streaming")
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
    generator: ChatCompletionsGenerator,
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
