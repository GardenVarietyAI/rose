import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from rose_server.dependencies import InferenceServer, get_inference_server, get_model_config
from rose_server.events.formatters import ResponsesFormatter
from rose_server.events.generator import EventGenerator
from rose_server.metrics import MetricsCollector
from rose_server.models.qwen_configs import get_qwen_config
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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["responses"])


@router.get("/responses/chains")
async def chains() -> List[str]:
    chains: List[str] = await get_chain_ids()
    return chains


# TODO: Naive token counting as a stop-gap for now
def _smart_truncate_messages(messages: List[ChatMessage], max_tokens: int) -> List[ChatMessage]:
    """Smart truncation that preserves system messages using character-based estimation."""
    if not messages or len(messages) <= 2:
        return messages

    # Rough estimation: 4 characters per token
    chars_per_token = 4
    max_chars = max_tokens * chars_per_token

    # Separate system and other messages
    system_msgs = []
    other_msgs = []

    for msg in messages:
        if msg.role == "system":
            system_msgs.append(msg)
        else:
            other_msgs.append(msg)

    # Count system message characters
    system_chars = sum(len(msg.content or "") for msg in system_msgs)
    system_chars += len(system_msgs) * 50  # Overhead for formatting

    if system_chars >= max_chars - 800:
        # System too large, keep last few messages
        logger.warning(f"System prompt too large (~{system_chars // chars_per_token} tokens), keeping recent messages")
        return messages[-5:]

    # Build result with system messages first
    result = system_msgs.copy()
    remaining_chars = max_chars - system_chars - 400

    # Add messages from the end backwards
    for msg in reversed(other_msgs):
        msg_chars = len(msg.content or "") + 50
        if msg_chars > remaining_chars:
            break
        result.append(msg)
        remaining_chars -= msg_chars

    # Restore chronological order
    result = system_msgs + list(reversed(result[len(system_msgs) :]))

    return result


async def _convert_input_to_messages(request: ResponsesRequest) -> List[ChatMessage]:
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
            if hasattr(msg, "type"):
                if msg.type == "function_call":
                    # Function calls are handled by the client
                    continue
                elif msg.type == "function_call_output":
                    # Format tool output in Hermes/Qwen3 format with <tool_response> tags
                    messages.append(
                        ChatMessage(role="tool", content=f"<tool_response>\n{msg.output}\n</tool_response>")
                    )
            else:
                messages.append(
                    ChatMessage(
                        role="system" if msg.role == "developer" else msg.role,
                        content=msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                        if msg.content
                        else "",
                    )
                )

    return messages


async def _generate_streaming_response(
    config: ModelConfig,
    messages: List[ChatMessage],
    inference_server: InferenceServer,
    tools: Optional[List[Any]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tool_choice: Optional[str] = None,
    chain_id: Optional[str] = None,
    use_codex_format: bool = False,
) -> EventSourceResponse:
    async def generate() -> AsyncIterator[Dict[str, Any]]:
        try:
            generator = EventGenerator(config, inference_server)
            formatter = ResponsesFormatter()
            metrics = MetricsCollector(model=config.model_name)

            # Create the event stream generator so we can optionally drain it in the background
            ev_gen = generator.generate_events(
                messages,
                enable_tools=bool(tools),
                tools=tools,
                max_tokens=max_output_tokens,
                temperature=temperature,
                tool_choice=tool_choice,
                chain_id=chain_id,
            )

            async for event in ev_gen:
                metrics.process_event(event)
                formatted = formatter.format_event(event)
                if formatted:
                    # Include the SSE event name for maximum compatibility with clients
                    ev_name = formatted.get("type", "")
                    yield {"data": json.dumps(formatted), "event": ev_name}

                    # No Codex-specific early turn closure; standard Responses stream.

            # Emit completion events for Responses format
            if hasattr(formatter, "get_completion_events"):
                for completion_event in formatter.get_completion_events():
                    yield {"data": json.dumps(completion_event), "event": completion_event.get("type", "")}

            yield {"data": "[DONE]", "event": "done"}
        except Exception as e:
            error_event = {"type": "response.error", "error": str(e)}
            yield {"data": json.dumps(error_event)}
            yield {"data": "[DONE]"}

    return EventSourceResponse(generate())


async def _generate_complete_response(
    config: ModelConfig,
    messages: List[ChatMessage],
    inference_server: InferenceServer,
    tools: Optional[List[Any]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tool_choice: Optional[str] = None,
    store: bool = False,
    chain_id: Optional[str] = None,
) -> ResponsesResponse:
    generator = EventGenerator(config, inference_server)
    formatter = ResponsesFormatter()
    metrics = MetricsCollector(model=config.model_name)
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
        metrics.process_event(event)
        all_events.append(event)

    complete_response = formatter.format_complete_response(all_events)
    complete_response.model = config.model_name

    # Log performance metrics
    performance_metrics = metrics.get_metrics()
    if performance_metrics:
        logger.info(f"[METRICS] Response generation complete for {config.model_name}")

    if store:
        await _store_response(complete_response, messages, config.model_name, chain_id)

    return complete_response


async def _store_response(
    complete_response: ResponsesResponse, messages: List[ChatMessage], model: str, chain_id: Optional[str] = None
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
    req: Request,
    request: ResponsesRequest = Body(...),
    inference_server: InferenceServer = Depends(get_inference_server),
) -> Union[EventSourceResponse, ResponsesResponse]:
    try:
        # Detect if this is a Codex request
        user_agent = req.headers.get("user-agent", "") if req else ""
        use_codex_format = user_agent.startswith("codex_cli_rs/")

        logger.info(f"RESPONSES API - max_output_tokens: {request.max_output_tokens}")
        logger.info(f"RESPONSES API - User-Agent: {user_agent}, use_codex_format: {use_codex_format}")

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

        # Smart truncation if needed
        qwen_config = get_qwen_config(config.model_id)

        # Quick estimation check - 4 chars per token average
        total_chars = sum(len(msg.content or "") for msg in messages)
        estimated_tokens = total_chars // 4 + len(messages) * 15  # Content + formatting overhead

        if estimated_tokens > qwen_config.max_context_length - 200:
            logger.warning(f"Estimated {estimated_tokens} tokens exceeds context limit, truncating")
            messages = _smart_truncate_messages(messages, qwen_config.max_context_length - 200)

        if request.stream:
            return await _generate_streaming_response(
                config=config,
                messages=messages,
                inference_server=inference_server,
                tools=request.tools,
                max_output_tokens=request.max_output_tokens,
                temperature=request.temperature,
                tool_choice=request.tool_choice,
                chain_id=request.prompt_cache_key
                or (previous_response.response_chain_id if previous_response else None),
                use_codex_format=use_codex_format,
            )
        else:
            chain_id = request.prompt_cache_key or (previous_response.response_chain_id if previous_response else None)
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
