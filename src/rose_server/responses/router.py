import json
import logging
from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from sse_starlette.sse import EventSourceResponse

from rose_server.events.formatters import ResponsesFormatter
from rose_server.events.generator import EventGenerator
from rose_server.llms import model_cache
from rose_server.llms.deps import ModelRegistryDep
from rose_server.responses.store import get_response, store_response_messages
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.responses import (
    ResponsesContentItem,
    ResponsesOutputItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["responses"])


async def _convert_input_to_messages(request: ResponsesRequest) -> list[ChatMessage]:
    """Convert request to messages, loading history if needed."""
    messages = []

    # Load conversation history if continuing from previous response
    if request.previous_response_id:
        from .store import get_conversation_messages

        messages = await get_conversation_messages(request.previous_response_id)

    # Add system instructions
    if request.instructions:
        messages.append(ChatMessage(role="system", content=request.instructions))

    # Add current user input
    if isinstance(request.input, str):
        # Handle string input directly
        messages.append(ChatMessage(role="user", content=request.input))
    elif isinstance(request.input, list):
        # Handle list of MessageCreateRequest objects
        for msg in request.input:
            # Map 'developer' role to 'system' for ChatMessage
            role = "system" if msg.role == "developer" else msg.role
            # Handle content - if it's a list, convert to string
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            messages.append(ChatMessage(role=role, content=content))

    return messages


async def _generate_streaming_response(request: ResponsesRequest, llm, messages: list[ChatMessage]):
    async def generate():
        try:
            generator = EventGenerator(llm)
            formatter = ResponsesFormatter()

            async for event in generator.generate_events(
                messages, enable_tools=bool(request.tools), tools=request.tools
            ):
                formatted = formatter.format_event(event)
                if formatted:
                    yield {"data": json.dumps(formatted)}
            yield {"data": "[DONE]"}
        except Exception as e:
            error_event = {"type": "response.error", "error": str(e)}
            yield {"data": json.dumps(error_event)}
            yield {"data": "[DONE]"}

    return EventSourceResponse(generate())


async def _generate_complete_response(
    request: ResponsesRequest, llm, messages: list[ChatMessage], chain_id: Optional[str] = None
):
    generator = EventGenerator(llm)
    formatter = ResponsesFormatter()
    all_events = []

    async for event in generator.generate_events(messages, enable_tools=bool(request.tools), tools=request.tools):
        all_events.append(event)

    complete_response = formatter.format_complete_response(all_events)
    complete_response["model"] = request.model

    if request.store:
        await _store_response(complete_response, messages, request.model, chain_id)

    return complete_response


async def _store_response(
    complete_response: dict, messages: list[ChatMessage], model: str, chain_id: Optional[str] = None
):
    reply_text = ""
    for output_item in complete_response.get("output", []):
        if output_item.get("type") == "message":
            content_list = output_item.get("content", [])
            for content_item in content_list:
                if content_item.get("type") == "output_text":
                    reply_text = content_item.get("text", "")
                    break

    message_id = await store_response_messages(
        messages=messages,
        reply_text=reply_text,
        model=model,
        input_tokens=complete_response["usage"]["input_tokens"],
        output_tokens=complete_response["usage"]["output_tokens"],
        created_at=complete_response["created_at"],
        chain_id=chain_id,
    )

    complete_response["id"] = message_id


@router.get("/responses/{response_id}", response_model=ResponsesResponse)
async def retrieve_response(response_id: str):
    try:
        response_msg = await get_response(response_id)
        if not response_msg:
            raise HTTPException(status_code=404, detail=f"Response {response_id} not found")

        text_content = ""

        if isinstance(response_msg.content, list):
            for item in response_msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content = item.get("text", "")
                    break
        else:
            logger.warning(f"Unexpected content format for response {response_id}: {type(response_msg.content)}")
            text_content = str(response_msg.content) if response_msg.content else ""

        model_name = response_msg.meta.get("model", "unknown") if response_msg.meta else "unknown"

        content_item = ResponsesContentItem(type="output_text", text=text_content)
        output_item = ResponsesOutputItem(
            id=response_msg.id,
            type="message",
            status="completed",
            role="assistant",
            content=[content_item],
        )

        # Get token counts from meta
        meta = response_msg.meta or {}
        usage = ResponsesUsage(
            input_tokens=meta.get("input_tokens", 0),
            output_tokens=meta.get("output_tokens", meta.get("token_count", 0)),  # Fallback to old token_count
            total_tokens=meta.get("total_tokens", meta.get("token_count", 0)),
        )

        response = ResponsesResponse(
            id=response_msg.id,
            created_at=response_msg.created_at,
            model=model_name,
            status="completed",
            output=[output_item],
            usage=usage,
        )

        return response.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving response {response_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Internal server error: {str(e)}",
                    "type": "server_error",
                    "code": None,
                }
            },
        )


@router.post("/responses", response_model=None)
async def create_response(request: ResponsesRequest = Body(...), registry: ModelRegistryDep = None):
    try:
        logger.info(f"RESPONSES API - Input type: {type(request.input)}, Input: {request.input}")
        logger.info(f"RESPONSES API - Instructions: {request.instructions}")

        # Validate previous_response_id if provided
        previous_response = None
        if request.previous_response_id:
            previous_response = await get_response(request.previous_response_id)
            if not previous_response:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": f"Previous response '{request.previous_response_id}' not found",
                            "type": "invalid_request_error",
                            "code": "response_not_found",
                        }
                    },
                )

        messages = await _convert_input_to_messages(request)
        logger.info(f"RESPONSES API - Converted messages: {messages}")

        if not messages:
            logger.error("No messages extracted from request")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "No valid messages found in request",
                        "type": "invalid_request_error",
                        "code": None,
                    }
                },
            )

        config = await registry.get_model_config(request.model)
        if not config:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"No configuration found for model '{request.model}'",
                        "type": "invalid_request_error",
                        "code": None,
                    }
                },
            )

        llm = await model_cache.get_model(request.model, config)

        if request.stream:
            return await _generate_streaming_response(request, llm, messages)
        else:
            return await _generate_complete_response(
                request, llm, messages, previous_response.response_chain_id if previous_response else None
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Responses API error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Internal server error: {str(e)}",
                    "type": "server_error",
                    "code": None,
                }
            },
        )
