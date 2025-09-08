import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import APIRouter, Body, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from rose_server.dependencies import InferenceServer, get_inference_server, get_model_config
from rose_server.events.formatters import ResponsesFormatter
from rose_server.events.generator import EventGenerator
from rose_server.responses.store import get_chain_ids, get_conversation_messages, get_response, store_response_messages
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.models import ModelConfig
from rose_server.schemas.responses import (
    ResponsesContentItem,
    ResponsesOutputItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from rose_server.tools import format_tools_for_prompt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["responses"])


@router.get("/responses/chains")
async def chains() -> List[str]:
    chains: List[str] = await get_chain_ids()
    return chains


async def _convert_input_to_messages(request: ResponsesRequest) -> list[ChatMessage]:
    """Convert request to messages, loading history if needed."""
    messages = []

    # Load conversation history if continuing from previous response
    if request.previous_response_id:
        chain_messages = await get_conversation_messages(request.previous_response_id)

        # Convert to ChatMessage format
        for msg in chain_messages:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        messages.append(ChatMessage(role=msg.role, content=item.get("text", "")))
                        break
            elif isinstance(msg.content, str):
                messages.append(ChatMessage(role=msg.role, content=msg.content))

    # Add system instructions
    system_content_parts = []

    if request.instructions:
        system_content_parts.append(request.instructions)

    # Auto-inject tool instructions if tools are provided
    if request.tools:
        tool_instructions = format_tools_for_prompt(request.tools)
        if tool_instructions:
            system_content_parts.append(tool_instructions)

    # Combine system instructions
    if system_content_parts:
        combined_instructions = "\n\n".join(system_content_parts)
        messages.append(ChatMessage(role="system", content=combined_instructions))

    # Add current user input
    if isinstance(request.input, str):
        # Handle string input directly
        messages.append(ChatMessage(role="user", content=request.input))
    elif isinstance(request.input, list):
        # Handle list of ResponsesInput objects
        for msg in request.input:
            # Check message type
            if hasattr(msg, "type"):
                if msg.type == "function_call":
                    # Function call from assistant - add as a message showing the call
                    content = f"[Function call: {msg.name}({msg.arguments})]"
                    messages.append(ChatMessage(role="assistant", content=content))
                elif msg.type == "function_call_output":
                    # Function output - add as a system message with instructions to use the result
                    content = (
                        f"The function returned the following result:\n\n{msg.output}\n\n"
                        "Please provide a natural language response incorporating this information."
                    )
                    messages.append(ChatMessage(role="system", content=content))
            else:
                # Standard message format
                # Map 'developer' role to 'system' for ChatMessage
                role = "system" if msg.role == "developer" else msg.role
                # Handle content - if it's a list, convert to string
                content = msg.content if isinstance(msg.content, str) else str(msg.content) if msg.content else ""
                messages.append(ChatMessage(role=role, content=content))

    return messages


async def _generate_streaming_response(
    config: ModelConfig,
    messages: list[ChatMessage],
    inference_server: InferenceServer,
    tools: Optional[list] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tool_choice: Optional[str] = None,
    chain_id: Optional[str] = None,
) -> EventSourceResponse:
    async def generate() -> AsyncIterator[Dict[str, Any]]:
        try:
            generator = EventGenerator(config, inference_server)
            formatter = ResponsesFormatter()

            async for event in generator.generate_events(
                messages,
                enable_tools=bool(tools),
                tools=tools,
                max_tokens=max_output_tokens,
                temperature=temperature,
                tool_choice=tool_choice,
                chain_id=chain_id,
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
    config: ModelConfig,
    messages: list[ChatMessage],
    inference_server: InferenceServer,
    tools: Optional[list] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tool_choice: Optional[str] = None,
    store: bool = False,
    chain_id: Optional[str] = None,
) -> ResponsesResponse:
    generator = EventGenerator(config, inference_server)
    formatter = ResponsesFormatter()
    all_events = []

    async for event in generator.generate_events(
        messages,
        enable_tools=bool(tools),
        tools=tools,
        max_tokens=max_output_tokens,
        temperature=temperature,
        tool_choice=tool_choice,
        chain_id=chain_id,
    ):
        all_events.append(event)

    complete_response = formatter.format_complete_response(all_events)
    complete_response.model = config.model_name

    if store:
        await _store_response(complete_response, messages, config.model_name, chain_id)

    return complete_response


async def _store_response(
    complete_response: ResponsesResponse, messages: list[ChatMessage], model: str, chain_id: Optional[str] = None
) -> None:
    reply_text = ""
    for output_item in complete_response.output:
        if output_item.type == "message":
            content_list = output_item.content
            for content_item in content_list:
                if content_item.type == "output_text":
                    reply_text = content_item.text
                    break

    message_id = await store_response_messages(
        messages=messages,
        reply_text=reply_text,
        model=model,
        input_tokens=complete_response.usage.input_tokens,
        output_tokens=complete_response.usage.output_tokens,
        created_at=complete_response.created_at,
        chain_id=chain_id,
    )

    complete_response.id = message_id


@router.get("/responses/{response_id}", response_model=ResponsesResponse)
async def retrieve_response(response_id: str) -> ResponsesResponse:
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
        return ResponsesResponse(
            id=response_msg.id,
            created_at=response_msg.created_at,
            model=model_name,
            status="completed",
            output=[output_item],
            usage=ResponsesUsage(
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", meta.get("token_count", 0)),  # Fallback to old token_count
                total_tokens=meta.get("total_tokens", meta.get("token_count", 0)),
            ),
        )
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
async def create_response(
    request: ResponsesRequest = Body(...),
    inference_server: InferenceServer = Depends(get_inference_server),
) -> Union[EventSourceResponse, ResponsesResponse]:
    try:
        logger.info(f"RESPONSES API - Input type: {type(request.input)}, Input: {request.input}")
        logger.info(f"RESPONSES API - Instructions: {request.instructions}")
        logger.info(f"RESPONSES API - max_output_tokens: {request.max_output_tokens}")

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

        config = await get_model_config(request.model)
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

        if request.stream:
            return await _generate_streaming_response(
                config=config,
                messages=messages,
                inference_server=inference_server,
                tools=request.tools,
                max_output_tokens=request.max_output_tokens,
                temperature=request.temperature,
                tool_choice=request.tool_choice,
                chain_id=previous_response.response_chain_id if previous_response else None,
            )
        else:
            chain_id = previous_response.response_chain_id if previous_response else None
            return await _generate_complete_response(
                config=config,
                messages=messages,
                inference_server=inference_server,
                tools=request.tools,
                max_output_tokens=request.max_output_tokens,
                temperature=request.temperature,
                tool_choice=request.tool_choice,
                store=request.store,
                chain_id=chain_id,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Responses API error: {str(e)}")
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
