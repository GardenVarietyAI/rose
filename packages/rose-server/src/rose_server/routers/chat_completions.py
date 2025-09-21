import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse
from rose_server.events.event_types import LLMEvent
from rose_server.events.formatters import ChatCompletionsFormatter
from rose_server.events.generator import EventGenerator
from rose_server.metrics import MetricsCollector
from rose_server.schemas.chat import ChatMessage, ChatRequest
from rose_server.schemas.models import ModelConfig
from rose_server.settings import settings
from rose_server.stores.models import get as get_language_model
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/v1/chat/completions")
logger = logging.getLogger(__name__)


def _prepare_tool_params(request: ChatRequest) -> Dict[str, Any]:
    enable_tools = bool(request.tools)
    return {"enable_tools": enable_tools, "tools": request.tools, "tool_choice": request.tool_choice}


@router.post("", response_model=None)
async def event_based_chat_completions(
    req: Request,
    request: ChatRequest = Body(...),
) -> JSONResponse | EventSourceResponse:
    inference_server = req.app.state.inference_server
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

    config = ModelConfig.from_language_model(
        model, inference_timeout=settings.inference_timeout, data_dir=settings.data_dir, models_dir=settings.models_dir
    )
    logger.info(f"[EVENT] Using model: {request.model}")
    messages = request.messages
    logger.info(f"[EVENT] Message count: {len(messages)}")

    try:
        generator = EventGenerator(config, inference_server, settings.max_concurrent_inference)
        formatter = ChatCompletionsFormatter()
        metrics = MetricsCollector(model=request.model)
        logger.info("[EVENT] Using ChatCompletionsGenerator for chat completions")
        if request.stream:
            return await create_event_streaming_response(generator, messages, formatter, request, metrics)
        else:
            return await create_event_complete_response(generator, messages, formatter, request, metrics)
    except Exception as e:
        logger.error(f"[EVENT] Error in chat completions: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})


async def create_event_streaming_response(
    generator: EventGenerator,
    messages: list[ChatMessage],
    formatter: ChatCompletionsFormatter,
    request: ChatRequest,
    metrics: MetricsCollector,
) -> EventSourceResponse:
    async def generate() -> AsyncGenerator[Dict[str, Any], None]:
        """Generate SSE events from LLM events."""
        try:
            tool_params = _prepare_tool_params(request)
            # Set seed in formatter for fingerprint generation
            formatter.set_request_seed(request.seed)
            async for event in generator.generate_events(
                messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                seed=request.seed,
                **tool_params,
            ):
                metrics.process_event(event)
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
    metrics: MetricsCollector,
) -> JSONResponse:
    try:
        tool_params = _prepare_tool_params(request)
        # Set seed in formatter for fingerprint generation
        formatter.set_request_seed(request.seed)
        all_events: list[LLMEvent] = []
        async for event in generator.generate_events(
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            seed=request.seed,
            **tool_params,
        ):
            # Process event for metrics
            metrics.process_event(event)
            all_events.append(event)

        # Get performance metrics and add to response
        performance_metrics = metrics.get_metrics()
        complete_response = formatter.format_complete_response(all_events)
        if performance_metrics:
            complete_response["performance"] = performance_metrics.to_dict()
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
