"""Run execution logic - refactored for clarity and mypy compliance."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import (
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from rose_core.config.service import DATA_DIR
from rose_server.events import ResponseCompleted, ResponseStarted, TokenGenerated
from rose_server.events.generators import RunsGenerator
from rose_server.language_models import model_cache
from rose_server.language_models.store import get as get_language_model
from rose_server.messages.store import create_message, get_messages
from rose_server.runs.store import RunsStore
from rose_server.runs.streaming import (
    stream_message_chunk,
    stream_message_completed,
    stream_message_created,
    stream_message_in_progress,
    stream_run_status,
    stream_run_step_event,
)
from rose_server.schemas.assistants import AssistantResponse
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.runs import RunResponse, RunStep, RunStepType
from rose_server.tools import format_tools_for_prompt, parse_xml_tool_call

logger = logging.getLogger(__name__)


@dataclass
class ResponseUsage:
    """Token-usage bookkeeping returned to the caller."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:  # pragma: no cover - trivial
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def find_latest_user_message(messages: Sequence[ChatMessage]) -> Optional[str]:
    """Return the most-recent user text message (or None)."""
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        for item in msg.content:
            if item.type == "text" and item.text.value:
                return str(item.text.value)
    return None


def _build_conversation_context(messages: Sequence[ChatMessage], *, limit: int = 5) -> str:
    """Join the last *limit* messages into a compact context string."""
    parts: List[str] = []
    for msg in messages[-limit:]:
        text = "".join(item.text.value for item in msg.content if item.type == "text" and item.text.value)
        if text:
            parts.append(f"{msg.role}: {text}")
    return "\n".join(parts)


async def _build_prompt(
    run: RunResponse,
    assistant: AssistantResponse,
    messages: Sequence[ChatMessage],
    latest_user_message: str,
) -> str:
    """Compose the final prompt shown to the language model."""
    prompt_parts: List[str] = []

    instructions = run.instructions or assistant.instructions
    if instructions:
        prompt_parts.append(f"Instructions: {instructions}")

    context = _build_conversation_context(messages)
    if context:
        prompt_parts.append(f"Recent conversation:\n{context}")

    if assistant.tools:
        tool_prompt = format_tools_for_prompt(assistant.tools, assistant_id=assistant.id)
        if tool_prompt:
            prompt_parts.append(tool_prompt)

    prompt_parts.append(f"\nUser: {latest_user_message}")
    return "\n\n".join(prompt_parts)


async def _fail_run(  # noqa: D401  (imperative helper)
    *,
    run_id: str,
    step: Optional[RunStep],
    runs_store: RunsStore,
    code: str,
    message: str,
) -> Tuple[str, ...]:
    """
    Centralised failure handling.

    Returns the events that need to be yielded in the generator in the order they
    should be sent to the client.
    """
    err = {"code": code, "message": message}
    if step:
        await runs_store.update_run_step(step.id, status="failed", last_error=err)
    await runs_store.update_run_status(run_id, "failed", last_error=err)
    status_evt = await stream_run_status(run_id, "failed", last_error=err)
    step_evt = await stream_run_step_event("failed", step) if step else ""
    # Empty strings are ignored by the caller
    return tuple(filter(bool, (step_evt, status_evt)))


async def _handle_tool_calls(
    *,
    run: RunResponse,
    assistant: AssistantResponse,
    response_text: str,
    step: RunStep,
    runs_store: RunsStore,
) -> Optional[Tuple[str, ...]]:
    """
    Detect an XML tool call in *response_text* and update state/streams.

    Returns a tuple of events to yield if a call is present,
    otherwise None so the caller continues the normal flow.
    """
    if not assistant.tools:
        return None

    parsed_call, _ = parse_xml_tool_call(response_text)
    if not parsed_call:
        return None

    call_id = f"call_{uuid.uuid4().hex[:8]}"
    tool_calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": parsed_call["tool"],
                "arguments": json.dumps(parsed_call["arguments"]),
            },
        }
    ]

    # Close MESSAGE_CREATION step
    await runs_store.update_run_step(
        step.id,
        status="completed",
        step_details={"message_creation": {"message_id": "pending_tools"}},
    )
    completed_evt = await stream_run_step_event("completed", step)

    # New TOOL_CALLS step
    tool_step = await runs_store.create_run_step(
        run_id=run.id,
        assistant_id=assistant.id,
        thread_id=run.thread_id,
        step_type=RunStepType.TOOL_CALLS,
        step_details={"tool_calls": tool_calls},
    )
    created_evt = await stream_run_step_event("created", tool_step)

    # Transition run to requires_action
    required_action = {
        "type": "submit_tool_outputs",
        "submit_tool_outputs": {"tool_calls": tool_calls},
    }
    await runs_store.update_run_status(run.id, "requires_action", required_action=required_action)
    status_evt = await stream_run_status(run.id, "requires_action", required_action=required_action)

    return completed_evt, created_evt, status_evt


async def execute_assistant_run_streaming(
    run: RunResponse,
    assistant: AssistantResponse,
) -> AsyncGenerator[str, None]:
    """
    Execute *run* and stream events as a server-sent event (SSE) compatible
    async generator.

    Behaviour is unchanged from the original implementation; the internals are
    merely decomposed for readability and testability.
    """
    runs_store = RunsStore()
    step: Optional[RunStep] = None

    # start run
    yield await stream_run_status(run.id, "in_progress")
    await runs_store.update_run_status(run.id, "in_progress")

    # create MESSAGE_CREATION step
    step = await runs_store.create_run_step(
        run_id=run.id,
        assistant_id=assistant.id,
        thread_id=run.thread_id,
        step_type=RunStepType.MESSAGE_CREATION,
        step_details={"message_creation": {"message_id": None}},
    )
    yield await stream_run_step_event("created", step)

    # validate thread
    messages = await get_messages(run.thread_id, order="asc")
    latest_user_msg = find_latest_user_message(messages)
    if latest_user_msg is None:
        for evt in await _fail_run(
            run_id=run.id,
            step=step,
            runs_store=runs_store,
            code="no_user_message",
            message="No user message found",
        ):
            yield evt
        return

    # create prompt
    full_prompt = await _build_prompt(run, assistant, messages, latest_user_msg)

    # locate model
    requested_model = run.model or assistant.model
    model = await get_language_model(requested_model)
    if not model:
        for evt in await _fail_run(
            run_id=run.id,
            step=step,
            runs_store=runs_store,
            code="model_not_found",
            message=f"Model '{requested_model}' not found",
        ):
            yield evt
        return

    # Build config from model
    config = {
        "model_name": model.model_name,
        "model_type": model.model_type,
        "temperature": model.temperature,
        "top_p": model.top_p,
        "memory_gb": model.memory_gb,
    }

    if model.is_fine_tuned and model.path:
        config["model_path"] = str(Path(DATA_DIR) / model.path)
        config["base_model"] = model.parent
        config["is_fine_tuned"] = True

    if model.get_lora_modules():
        config["lora_target_modules"] = model.get_lora_modules()

    if not config.get("model_name"):
        for evt in await _fail_run(
            run_id=run.id,
            step=step,
            runs_store=runs_store,
            code="model_error",
            message=f"No configuration found for model: {requested_model}",
        ):
            yield evt
        return

    # load / cache model
    try:
        llm = await model_cache.get_model(requested_model, config)
    except Exception as exc:  # pragma: no cover
        for evt in await _fail_run(
            run_id=run.id,
            step=step,
            runs_store=runs_store,
            code="model_error",
            message=f"Failed to create model: {exc}",
        ):
            yield evt
        return

    # stream message creation events
    message_id = f"msg_{uuid.uuid4().hex}"
    yield await stream_message_created(message_id, run.thread_id, assistant.id, run.id)
    yield await stream_message_in_progress(message_id, run.thread_id, assistant.id, run.id)

    # model inference with incremental streaming
    generator = RunsGenerator(llm)
    chat_prompt = [ChatMessage(role="user", content=full_prompt)]
    enable_tools = bool(assistant.tools)

    usage = ResponseUsage()
    response_text = ""

    try:
        async for event in generator.generate_events(
            chat_prompt,
            temperature=run.temperature or assistant.temperature,
            top_p=run.top_p or assistant.top_p,
            enable_tools=enable_tools,
            tools=assistant.tools if enable_tools else None,
        ):
            if isinstance(event, ResponseStarted):
                usage.prompt_tokens = getattr(event, "prompt_tokens", 0)
            elif isinstance(event, TokenGenerated):
                response_text += event.token
                usage.completion_tokens += 1
                yield await stream_message_chunk(run.id, message_id, event.token)
            elif isinstance(event, ResponseCompleted):
                if event.output_tokens:
                    usage.completion_tokens = event.output_tokens
    except Exception as exc:  # pragma: no cover
        logger.exception("Inference error")
        for evt in await _fail_run(
            run_id=run.id,
            step=step,
            runs_store=runs_store,
            code="execution_error",
            message=str(exc),
        ):
            yield evt
        return

    # detect tool calls (may early-return)
    tool_events = await _handle_tool_calls(
        run=run,
        assistant=assistant,
        response_text=response_text,
        step=step,
        runs_store=runs_store,
    )
    if tool_events:
        for evt in tool_events:
            yield evt
        return

    # normal assistant text reply
    yield await stream_message_completed(
        message_id,
        response_text,
        run.thread_id,
        assistant.id,
        run.id,
    )
    message = await create_message(
        thread_id=run.thread_id,
        role="assistant",
        content=[{"type": "text", "text": response_text}],
        metadata={"run_id": run.id, "assistant_id": assistant.id},
    )

    # close step & run
    await runs_store.update_run_step(
        step.id,
        status="completed",
        step_details={"message_creation": {"message_id": message.id}},
        usage=usage.to_dict(),
    )
    yield await stream_run_step_event("completed", step)

    await runs_store.update_run_status(run.id, "completed", usage=usage.to_dict())
    yield await stream_run_status(run.id, "completed", usage=usage.to_dict())
