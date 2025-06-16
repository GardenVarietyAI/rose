"""OpenAI-compatible completions API router."""

import logging
import time
import uuid
from typing import List, Union

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from rose_server.events import TokenGenerated
from rose_server.events.generators import CompletionsGenerator
from rose_server.llms.huggingface_llm import HuggingFaceLLM
from rose_server.schemas.completions import (
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)
from rose_server.services import get_model_registry

router = APIRouter()
logger = logging.getLogger(__name__)


def ensure_list(prompt: Union[str, List[str]]) -> List[str]:
    """Ensure prompt is a list."""
    if isinstance(prompt, str):
        return [prompt]
    return prompt


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest = Body(...),
    http_request: Request = None,
) -> Union[JSONResponse, EventSourceResponse]:
    """OpenAI-compatible endpoint for text completions.
    This endpoint is designed for base models that complete prompts
    rather than following chat/instruction formats.
    """
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
    logger.info(f"Completion request for model: {request.model}")
    config = registry.get_model_config(request.model)
    if not config:
        return JSONResponse(status_code=400, content={"error": f"Model {request.model} not available"})
    try:
        llm = HuggingFaceLLM(config)
    except Exception as e:
        logger.error(f"Failed to create model {request.model}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to load model: {str(e)}"})
    prompts = ensure_list(request.prompt)
    if request.stream:
        return await create_streaming_completion(llm, prompts, request)
    else:
        return await create_standard_completion_from_stream(llm, prompts, request)


async def create_standard_completion_from_stream(llm, prompts: List[str], request: CompletionRequest) -> JSONResponse:
    """Create a standard (non-streaming) completion by collecting stream events."""

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for idx, prompt in enumerate(prompts[: request.n]):
        try:
            generator = CompletionsGenerator(llm)
            text = ""
            if request.echo:
                text = prompt
            async for event in generator.generate_prompt_events(
                prompt, temperature=request.temperature, max_tokens=request.max_tokens, echo=False
            ):
                if isinstance(event, TokenGenerated):
                    text += event.token
            choice = CompletionChoice(text=text, index=idx, finish_reason="stop")
            choices.append(choice)
            total_prompt_tokens += 10
            total_completion_tokens += len(text.split())
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            choice = CompletionChoice(text="", index=idx, finish_reason="error")
            choices.append(choice)
    response = CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )
    return JSONResponse(content=response.dict())


async def create_streaming_completion(llm, prompts: List[str], request: CompletionRequest) -> EventSourceResponse:
    """Create a streaming completion."""

    async def generate():
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        prompt = prompts[0]
        try:
            logger.info(f"Streaming completion for prompt: {prompt[:50]}...")
            if request.echo:
                chunk = CompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[CompletionChoice(text=prompt, index=0, finish_reason=None)],
                )
                yield {"data": chunk.json()}
            generator = CompletionsGenerator(llm)
            async for event in generator.generate_prompt_events(
                prompt, temperature=request.temperature, max_tokens=request.max_tokens, echo=False
            ):
                if isinstance(event, TokenGenerated):
                    chunk = CompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[CompletionChoice(text=event.token, index=0, finish_reason=None)],
                    )
                    yield {"data": chunk.json()}
            final_chunk = CompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[CompletionChoice(text="", index=0, finish_reason="stop")],
            )
            yield {"data": final_chunk.json()}
            yield {"data": "[DONE]"}
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_chunk = CompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[CompletionChoice(text=f"\nError: {str(e)}", index=0, finish_reason="error")],
            )
            yield {"data": error_chunk.json()}

    return EventSourceResponse(generate(), media_type="text/event-stream")
