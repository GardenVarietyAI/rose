import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
)

from sse_starlette import ServerSentEvent

from rose_core.config.service import DATA_DIR
from rose_server.database import current_timestamp
from rose_server.entities.messages import Message
from rose_server.entities.run_steps import RunStep
from rose_server.events import ResponseCompleted, ResponseStarted, TokenGenerated
from rose_server.events.generators import RunsGenerator
from rose_server.language_models import model_cache
from rose_server.language_models.store import get as get_language_model
from rose_server.messages.store import create_message, get_messages
from rose_server.runs.steps.store import create_run_step, update_run_step
from rose_server.runs.store import update_run
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
from rose_server.schemas.runs import RunResponse, RunStepResponse
from rose_server.tools import format_tools_for_prompt, parse_xml_tool_call

logger = logging.getLogger(__name__)


@dataclass
class ResponseUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def find_latest_user_message(messages: List[Message]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        for item in msg.content:
            if item["type"] == "text":
                return str(item["text"]["value"])
    return None


def build_conversation_context(messages: List[Message], *, limit: int = 5) -> str:
    parts: List[str] = []
    for msg in messages[-limit:]:
        text_parts = []
        for item in msg.content:
            if item["type"] == "text":
                text_parts.append(item["text"]["value"])
        text = "".join(text_parts)
        if text:
            parts.append(f"{msg.role}: {text}")
    return "\n".join(parts)


async def build_prompt(
    run: RunResponse,
    assistant: AssistantResponse,
    messages: List[Message],
    latest_user_message: str,
) -> str:
    prompt_parts: List[str] = []

    instructions = run.instructions or assistant.instructions
    if instructions:
        prompt_parts.append(f"Instructions: {instructions}")

    context = build_conversation_context(messages)
    if context:
        prompt_parts.append(f"Recent conversation:\n{context}")

    if assistant.tools:
        tool_prompt = format_tools_for_prompt(assistant.tools, assistant_id=assistant.id)
        if tool_prompt:
            prompt_parts.append(tool_prompt)

    prompt_parts.append(f"\nUser: {latest_user_message}")
    return "\n\n".join(prompt_parts)


async def get_model_for_run(run: RunResponse, assistant: AssistantResponse) -> Any:
    requested_model = run.model or assistant.model
    model = await get_language_model(requested_model)
    if not model:
        raise ValueError(f"Model '{requested_model}' not found")
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
        raise ValueError(f"No configuration found for model: {requested_model}")
    return await model_cache.get_model(requested_model, config)


async def create_message_creation_step(run: RunResponse, assistant: AssistantResponse) -> RunStepResponse:
    step_entity = RunStep(
        id=f"step_{uuid.uuid4().hex}",
        created_at=current_timestamp(),
        run_id=run.id,
        assistant_id=assistant.id,
        thread_id=run.thread_id,
        type="message_creation",
        step_details={"message_creation": {"message_id": None}},
        status="in_progress",
    )
    await create_run_step(step_entity)
    return RunStepResponse(**step_entity.model_dump())


async def close_and_update_step(
    step: RunStepResponse, status: str, details: Dict[str, Any], usage: Optional[Dict[str, int]] = None
) -> None:
    await update_run_step(
        step.id,
        status=status,
        step_details=details,
        usage=usage,
    )


async def fail_run(
    run_id: str,
    step: Optional[RunStepResponse],
    code: str,
    message: str,
) -> AsyncGenerator[ServerSentEvent, None]:
    err = {"code": code, "message": message}
    if step:
        await update_run_step(step.id, status="failed", last_error=err)
    await update_run(run_id, status="failed", last_error=err)
    status_evt = await stream_run_status(run_id, "failed", last_error=err)
    step_evt = await stream_run_step_event("failed", step) if step else ""
    if step:
        yield step_evt
    yield status_evt


async def handle_tool_calls(
    run: RunResponse,
    assistant: AssistantResponse,
    response_text: str,
    step: RunStepResponse,
) -> Optional[Tuple[str, ...]]:
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

    await update_run_step(
        step.id,
        status="completed",
        step_details={"message_creation": {"message_id": "pending_tools"}},
    )
    completed_evt = await stream_run_step_event("completed", step)

    tool_step_entity = RunStep(
        id=f"step_{uuid.uuid4().hex}",
        created_at=current_timestamp(),
        run_id=run.id,
        assistant_id=assistant.id,
        thread_id=run.thread_id,
        type="tool_calls",
        step_details={"tool_calls": tool_calls},
        status="in_progress",
    )
    await create_run_step(tool_step_entity)
    tool_step = RunStepResponse(**tool_step_entity.model_dump())
    created_evt = await stream_run_step_event("created", tool_step)

    required_action = {
        "type": "submit_tool_outputs",
        "submit_tool_outputs": {"tool_calls": tool_calls},
    }
    await update_run(run.id, status="requires_action", required_action=required_action)
    status_evt = await stream_run_status(run.id, "requires_action", required_action=required_action)

    return completed_evt, created_evt, status_evt


async def execute_assistant_run_streaming(
    run: RunResponse,
    assistant: AssistantResponse,
) -> AsyncGenerator[ServerSentEvent, None]:
    """
    Execute *run* and stream events as a server-sent event (SSE) compatible
    async generator. Modular, testable, readable.
    """
    # Start run
    await update_run(run.id, status="in_progress")
    yield await stream_run_status(run.id, "in_progress")
    step = await create_message_creation_step(run, assistant)
    yield await stream_run_step_event("created", step)

    # Validate thread
    messages = await get_messages(run.thread_id, order="asc")
    latest_user_msg = find_latest_user_message(messages)
    if latest_user_msg is None:
        async for evt in fail_run(run.id, step, "no_user_message", "No user message found"):
            yield evt
        return

    # Build prompt
    full_prompt = await build_prompt(run, assistant, messages, latest_user_msg)

    # Get model
    try:
        llm = await get_model_for_run(run, assistant)
    except Exception as exc:
        async for evt in fail_run(run.id, step, "model_error", str(exc)):
            yield evt
        return

    # Stream message creation events
    message_id = f"msg_{uuid.uuid4().hex}"
    yield await stream_message_created(message_id, run.thread_id, assistant.id, run.id)
    yield await stream_message_in_progress(message_id, run.thread_id, assistant.id, run.id)

    # Model inference & streaming
    generator = RunsGenerator(llm)
    chat_prompt = [ChatMessage(role="user", content=full_prompt)]
    usage = ResponseUsage()
    response_text = ""

    try:
        async for event in generator.generate_events(
            chat_prompt,
            temperature=run.temperature or assistant.temperature,
            top_p=run.top_p or assistant.top_p,
            enable_tools=bool(assistant.tools),
            tools=assistant.tools if assistant.tools else None,
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
    except Exception as exc:
        logger.exception("Inference error")
        async for evt in fail_run(run.id, step, "execution_error", str(exc)):
            yield evt
        return

    # Tool call detection/handling
    tool_events = await handle_tool_calls(
        run=run,
        assistant=assistant,
        response_text=response_text,
        step=step,
    )
    if tool_events:
        for evt in tool_events:
            yield evt
        return

    # Normal assistant text reply
    yield await stream_message_completed(
        message_id,
        response_text,
        run.thread_id,
        assistant.id,
        run.id,
    )
    message = Message(
        id=f"msg_{uuid.uuid4().hex}",
        created_at=current_timestamp(),
        thread_id=run.thread_id,
        role="assistant",
        content=[{"type": "text", "text": {"value": response_text, "annotations": []}}],
        assistant_id=assistant.id,
        run_id=run.id,
        meta={},
    )
    message = await create_message(message)

    await close_and_update_step(
        step,
        "completed",
        {"message_creation": {"message_id": message.id}},
        usage=usage.to_dict(),
    )
    yield await stream_run_step_event("completed", step)

    await update_run(run.id, status="completed", usage=usage.to_dict())
    yield await stream_run_status(run.id, "completed", usage=usage.to_dict())
