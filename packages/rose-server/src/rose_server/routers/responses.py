import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Request
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
from sqlmodel import col, select
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["responses"])


@router.get("/responses/chains")
async def chains(req: Request) -> List[str]:
    async with req.app.state.get_db_session(read_only=True) as session:
        query = (
            select(Message.response_chain_id)
            .where(Message.response_chain_id.isnot(None))  # type: ignore
            .distinct()
            .order_by(Message.created_at)  # type: ignore
        )
        result = await session.execute(query)
        chain_ids: List[str] = list(result.scalars().all())
        return chain_ids


@dataclass
class TokenizedMessage:
    message: ChatMessage
    token_count: int


def _smart_truncate_messages(tokenized_messages: List[TokenizedMessage], max_tokens: int) -> List[ChatMessage]:
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


async def _save_response_messages(
    formatter: ResponsesFormatter,
    messages: List[ChatMessage],
    config: ModelConfig,
    chain_id: str,
    get_db_session: Any,
) -> None:
    response_text = formatter.accumulated_content
    response_id = formatter.response_id

    if not response_text or not response_id:
        logger.warning("No response text or ID captured, skipping save")
        return

    try:
        async with get_db_session() as session:
            current_user_message = next((msg for msg in reversed(messages) if msg.role == "user"), None)
            if current_user_message:
                user_message = Message(
                    role="user",
                    content=[{"type": "text", "text": current_user_message.content}],
                    created_at=formatter.created_at or int(time.time()),
                    response_chain_id=chain_id,
                    meta={"model": config.model_name},
                )
                session.add(user_message)

            assistant_message = Message(
                id=response_id,
                role="assistant",
                content=[{"type": "text", "text": response_text}],
                created_at=formatter.created_at or int(time.time()),
                response_chain_id=chain_id,
                meta={"model": config.model_name},
            )
            session.add(assistant_message)
            await session.commit()
            logger.info(f"Saved response {response_id} to chain {chain_id}")
    except Exception as e:
        logger.error(f"Failed to save response messages: {e}", exc_info=True)


@router.get("/responses/{response_id}", response_model=ResponsesResponse)
async def retrieve_response(req: Request, response_id: str) -> ResponsesResponse:
    try:
        async with req.app.state.get_db_session(read_only=True) as session:
            response_msg = await session.get(Message, response_id)

        if not response_msg:
            raise HTTPException(status_code=404, detail=f"Response {response_id} not found")

        text_content = ""

        for item in response_msg.content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_content = item.get("text", "")
                break

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
                output_tokens=meta.get("output_tokens", 0),
                total_tokens=meta.get("total_tokens", 0),
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
    background_tasks: BackgroundTasks,
    request: ResponsesRequest = Body(...),
) -> EventSourceResponse | ResponsesResponse:
    try:
        inference_server = req.app.state.inference_server
        use_codex_format = req.headers.get("user-agent", "").startswith("codex_cli_rs/")

        logger.info(f"RESPONSES API - max_output_tokens: {request.max_output_tokens}")
        logger.info(f"RESPONSES API - use_codex_format: {use_codex_format}")

        previous_response = None
        messages = []

        if request.previous_response_id:
            async with req.app.state.get_db_session(read_only=True) as session:
                previous_response = await session.get(Message, request.previous_response_id)

                if not previous_response:
                    raise HTTPException(
                        status_code=400, detail=f"Previous response '{request.previous_response_id}' not found"
                    )

                query = (
                    select(Message)
                    .where(Message.response_chain_id == previous_response.response_chain_id)
                    .order_by(col(Message.created_at))
                )
                result = await session.execute(query)
                chain_messages: List[Message] = list(result.scalars().all())

                for msg in chain_messages:
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            messages.append(ChatMessage(role=msg.role, content=item.get("text", "")))  # type: ignore[arg-type]
                            break

        system_content_parts = []
        if request.instructions:
            system_content_parts.append(request.instructions)

        combined_instructions = "\n\n".join(system_content_parts)
        messages.append(ChatMessage(role="system", content=combined_instructions))

        # Add current user input
        if isinstance(request.input, str):
            messages.append(ChatMessage(role="user", content=request.input))

        if isinstance(request.input, list):
            # Handle list of ResponsesInput objects
            for msg in request.input:  # type: ignore[assignment]
                if hasattr(msg, "type"):
                    if getattr(msg, "type", None) == "function_call":
                        # Preserve function calls in conversation history for context
                        messages.append(
                            ChatMessage(
                                role="assistant",
                                content=None,
                                tool_calls=[
                                    {
                                        "id": getattr(msg, "call_id", None) or getattr(msg, "id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": getattr(msg, "name", ""),
                                            "arguments": getattr(msg, "arguments", ""),
                                        },
                                    }
                                ],
                            )
                        )
                    elif getattr(msg, "type", None) == "function_call_output":
                        # Format tool output with <tool_response> tags for Hermes/Qwen3
                        messages.append(
                            ChatMessage(
                                role="tool",
                                content=f"<tool_response>\n{getattr(msg, 'output', '')}\n</tool_response>",
                            )
                        )
                else:
                    msg_role = getattr(msg, "role", "user")
                    msg_content = getattr(msg, "content", "")
                    messages.append(
                        ChatMessage(
                            role="system" if msg_role == "developer" else msg_role,  # type: ignore[arg-type]
                            content=msg_content
                            if isinstance(msg_content, str)
                            else str(msg_content)
                            if msg_content
                            else "",
                        )
                    )

        if not messages:
            logger.error("No messages extracted from request")
            raise HTTPException(status_code=400, detail="No valid messages found in request")

        async with req.app.state.get_db_session(read_only=True) as session:
            result = await session.execute(select(LanguageModel).where(LanguageModel.id == request.model))
            model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=400, detail=f"No configuration found for model '{request.model}'")

        config = ModelConfig.from_language_model(
            model,
            inference_timeout=req.app.state.settings.inference_timeout,
            data_dir=req.app.state.settings.data_dir,
            models_dir=req.app.state.settings.models_dir,
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
            logger.warning(f"Total tokens {total_tokens} exceeds max {qwen_config.max_context_length}, truncating")
            messages = _smart_truncate_messages(tokenized_messages, qwen_config.max_context_length)

        chain_id = request.prompt_cache_key or (previous_response.response_chain_id if previous_response else None)
        if not chain_id:
            chain_id = str(uuid.uuid4())

        formatter = ResponsesFormatter()
        generator = EventGenerator(config, inference_server, req.app.state.settings.max_concurrent_inference)
        metrics = MetricsCollector(model=config.model_name)

        if request.stream:

            async def generate() -> AsyncIterator[Dict[str, Any]]:
                try:
                    async for event in generator.generate_events(
                        messages,
                        enable_tools=bool(request.tools),
                        tools=request.tools,
                        max_tokens=request.max_output_tokens,
                        temperature=request.temperature,
                        tool_choice=request.tool_choice,
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

            if request.store:
                background_tasks.add_task(
                    _save_response_messages,
                    formatter=formatter,
                    messages=messages,
                    config=config,
                    chain_id=chain_id,
                    get_db_session=req.app.state.get_db_session,
                )

            return EventSourceResponse(generate())
        else:
            all_events = []

            async for event in generator.generate_events(
                messages,
                enable_tools=bool(request.tools),
                tools=request.tools,
                max_tokens=request.max_output_tokens,
                temperature=request.temperature,
                tool_choice=request.tool_choice,
                chain_id=chain_id,
            ):
                metrics.process_event(event)
                formatter.format_event(event)
                all_events.append(event)

            complete_response = formatter.format_complete_response(all_events)
            complete_response.model = config.model_name

            performance_metrics = metrics.get_metrics()
            if performance_metrics:
                logger.info(f"[METRICS] Response generation complete for {config.model_name}")

            if request.store:
                background_tasks.add_task(
                    _save_response_messages,
                    formatter=formatter,
                    messages=messages,
                    config=config,
                    chain_id=chain_id,
                    get_db_session=req.app.state.get_db_session,
                )

            return complete_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Responses API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
