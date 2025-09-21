import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import APIRouter, Body, HTTPException, Request
from rose_server._inference import InferenceServer
from rose_server.database import get_session
from rose_server.entities.messages import Message
from rose_server.entities.models import LanguageModel
from rose_server.events.formatters import ResponsesFormatter
from rose_server.events.generator import EventGenerator
from rose_server.metrics import MetricsCollector
from rose_server.models.qwen_configs import get_qwen_config
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.models import ModelConfig
from rose_server.schemas.responses import (
    ResponsesContentItem,
    ResponsesOutputItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from rose_server.settings import settings
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["responses"])


@router.get("/responses/chains")
async def chains() -> List[str]:
    """Get all conversation chain IDs."""
    async with get_session(read_only=True) as session:
        query = (
            select(Message.response_chain_id)
            .where(Message.response_chain_id.is_not(None))
            .distinct()
            .order_by(Message.created_at)
        )
        result = await session.execute(query)
        chain_ids: List[str] = result.scalars().all()
        return chain_ids


@dataclass
class TokenizedMessage:
    message: ChatMessage
    token_count: int


def _smart_truncate_messages(tokenized_messages: List[TokenizedMessage], max_tokens: int) -> List[ChatMessage]:
    """Pure function for smart message truncation preserving system messages."""
    if not tokenized_messages or len(tokenized_messages) <= 2:
        return [tm.message for tm in tokenized_messages]

    system_msgs: List[TokenizedMessage] = []
    other_msgs: List[TokenizedMessage] = []

    for tm in tokenized_messages:
        if tm.message.role == "system":
            system_msgs.append(tm)
        else:
            other_msgs.append(tm)

    # Count system message tokens (add ~15 per message for formatting)
    system_tokens = sum(tm.token_count + 15 for tm in system_msgs)

    if system_tokens >= max_tokens - 200:
        logger.warning(f"System prompt too large ({system_tokens} tokens), keeping recent messages")
        return [tm.message for tm in tokenized_messages[-5:]]

    result: List[ChatMessage] = [tm.message for tm in system_msgs]
    remaining_tokens = max_tokens - system_tokens - 100  # Buffer for response

    # Add messages from the end backwards until we hit token limit
    for tm in reversed(other_msgs):
        msg_tokens = tm.token_count + 15  # Add formatting overhead
        if msg_tokens > remaining_tokens:
            break
        result.append(tm.message)
        remaining_tokens -= msg_tokens

    # Restore chronological order
    final_result = [tm.message for tm in system_msgs]
    final_result.extend(reversed(result[len(system_msgs) :]))

    if len(final_result) < len(tokenized_messages):
        used_tokens = max_tokens - remaining_tokens
        logger.info(f"Truncated: {len(tokenized_messages)} -> {len(final_result)} messages (~{used_tokens} tokens)")

    return final_result


async def _convert_input_to_messages(request: ResponsesRequest) -> List[ChatMessage]:
    """Convert request to messages, loading history if needed."""
    messages = []

    if request.previous_response_id:
        # Load all messages in the conversation chain
        async with get_session(read_only=True) as session:
            # Get the response message
            response_msg = await session.get(Message, request.previous_response_id)
            if response_msg:
                # Load all messages in the chain
                query = (
                    select(Message)
                    .where(Message.response_chain_id == response_msg.response_chain_id)
                    .order_by(Message.created_at)
                )
                result = await session.execute(query)
                chain_messages: List[Message] = result.scalars().all()
            else:
                chain_messages = []

        for msg in chain_messages:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        messages.append(ChatMessage(role=msg.role, content=item.get("text", "")))
                        break
            elif isinstance(msg.content, str):
                messages.append(ChatMessage(role=msg.role, content=msg.content))

    # Combine system instructions
    system_content_parts = []
    if request.instructions:
        system_content_parts.append(request.instructions)
    if system_content_parts:
        combined_instructions = "\n\n".join(system_content_parts)
        messages.append(ChatMessage(role="system", content=combined_instructions))

    # Add current user input
    if isinstance(request.input, str):
        messages.append(ChatMessage(role="user", content=request.input))
    elif isinstance(request.input, list):
        # Handle list of ResponsesInput objects
        for msg in request.input:
            if hasattr(msg, "type"):
                if msg.type == "function_call":
                    # Preserve function calls in conversation history for context
                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                {
                                    "id": msg.call_id or msg.id,
                                    "type": "function",
                                    "function": {"name": msg.name, "arguments": msg.arguments},
                                }
                            ],
                        )
                    )
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
    max_concurrent_inference: int = 2,
) -> EventSourceResponse:
    async def generate() -> AsyncIterator[Dict[str, Any]]:
        try:
            generator = EventGenerator(config, inference_server, max_concurrent_inference)
            formatter = ResponsesFormatter()
            metrics = MetricsCollector(model=config.model_name)

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
                formatted = formatter.format_event(event)
                if formatted:
                    ev_name = formatted.get("type", "")
                    yield {"data": json.dumps(formatted), "event": ev_name}

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
    max_concurrent_inference: int = 2,
) -> ResponsesResponse:
    generator = EventGenerator(config, inference_server, max_concurrent_inference)
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

    # Store response messages
    end_time = time.time()
    # Generate new chain_id if not provided
    if not chain_id:
        chain_id = f"chain_{uuid.uuid4().hex[:16]}"

    async with get_session() as session:
        current_user_message = next((msg for msg in reversed(messages) if msg.role == "user"), None)
        if current_user_message:
            user_message = Message(
                role="user",
                content=[{"type": "text", "text": current_user_message.content}],
                created_at=complete_response.created_at,
                response_chain_id=chain_id,
                meta={"model": model},
            )
            session.add(user_message)

        assistant_message = Message(
            role="assistant",
            content=[{"type": "text", "text": reply_text}],
            created_at=complete_response.created_at,
            response_chain_id=chain_id,
            meta={
                "model": model,
                "input_tokens": complete_response.usage.input_tokens,
                "output_tokens": complete_response.usage.output_tokens,
                "total_tokens": complete_response.usage.input_tokens + complete_response.usage.output_tokens,
                "response_time_ms": int((end_time - complete_response.created_at) * 1000),
            },
        )
        session.add(assistant_message)
        await session.commit()
        message_id: str = assistant_message.id

    complete_response.id = message_id


@router.get("/responses/{response_id}", response_model=ResponsesResponse)
async def retrieve_response(response_id: str) -> ResponsesResponse:
    try:
        async with get_session(read_only=True) as session:
            response_msg = await session.get(Message, response_id)

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
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/responses", response_model=None)
async def create_response(
    req: Request,
    request: ResponsesRequest = Body(...),
) -> Union[EventSourceResponse, ResponsesResponse]:
    try:
        inference_server = req.app.state.inference_server
        use_codex_format = req.headers.get("user-agent", "").startswith("codex_cli_rs/")

        logger.info(f"RESPONSES API - max_output_tokens: {request.max_output_tokens}")
        logger.info(f"RESPONSES API - use_codex_format: {use_codex_format}")

        previous_response = None
        if request.previous_response_id:
            async with get_session(read_only=True) as session:
                previous_response = await session.get(Message, request.previous_response_id)

        if request.previous_response_id and not previous_response:
            raise HTTPException(status_code=400, detail=f"Previous response '{request.previous_response_id}' not found")

        messages = await _convert_input_to_messages(request)
        if not messages:
            logger.error("No messages extracted from request")
            raise HTTPException(status_code=400, detail="No valid messages found in request")

        # Get the language model
        async with get_session(read_only=True) as session:
            result = await session.execute(select(LanguageModel).where(LanguageModel.id == request.model))
            model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=400, detail=f"No configuration found for model '{request.model}'")
        config = ModelConfig.from_language_model(
            model,
            inference_timeout=settings.inference_timeout,
            data_dir=settings.data_dir,
            models_dir=settings.models_dir,
        )

        if not req.app.state.tokenizer:
            raise HTTPException(status_code=500, detail="Tokenizer not initialized")

        tokenized_messages = [
            TokenizedMessage(message=msg, token_count=len(req.app.state.tokenizer.encode(msg.content).ids))
            for msg in messages
            if msg.content
        ]

        total_tokens = sum(tm.token_count for tm in tokenized_messages)
        qwen_config = get_qwen_config(config.model_id)
        if total_tokens > qwen_config.max_context_length:
            logger.warning(f"Messages use {total_tokens} tokens, exceeds {qwen_config.max_context_length}, truncating")
            messages = _smart_truncate_messages(tokenized_messages, qwen_config.max_context_length)

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
                max_concurrent_inference=settings.max_concurrent_inference,
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
                max_concurrent_inference=settings.max_concurrent_inference,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Responses API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
