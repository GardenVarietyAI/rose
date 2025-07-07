import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from sse_starlette import ServerSentEvent

from rose_core.config.settings import settings
from rose_server.database import current_timestamp
from rose_server.entities.messages import Message
from rose_server.entities.run_steps import RunStep
from rose_server.events.event_types import ResponseCompleted, ResponseStarted, TokenGenerated
from rose_server.events.formatters.runs import RunsFormatter
from rose_server.events.generator import EventGenerator
from rose_server.models.store import get as get_language_model
from rose_server.runs.builtin_tools import execute_builtin_tool
from rose_server.runs.prompt_builder import find_latest_user_message
from rose_server.runs.steps.store import create_run_step, update_run_step
from rose_server.runs.store import update_run
from rose_server.schemas.assistants import AssistantResponse
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.runs import RunResponse, RunStepResponse
from rose_server.threads.messages.store import create_message, get_messages
from rose_server.tools import parse_xml_tool_call
from rose_server.types.runs import ResponseUsage

logger = logging.getLogger(__name__)


async def stream_run_step_event(event_type: str, step: RunStepResponse) -> ServerSentEvent:
    """Create a streaming event for run step updates."""
    event_data = step.dict()
    return ServerSentEvent(data=json.dumps(event_data), event=f"thread.run.step.{event_type}")


async def stream_run_status(run_id: str, status: str, **kwargs: Dict[str, Any]) -> ServerSentEvent:
    """Create a streaming event for run status updates."""
    event_data = {
        "id": run_id,
        "object": f"thread.run.{status}",
        "delta": {"status": status, **kwargs},
    }
    return ServerSentEvent(data=json.dumps(event_data), event=f"thread.run.{status}")


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


async def get_model_for_run(model_name: str) -> Dict[str, Any]:
    model = await get_language_model(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found")

    config = model.model_dump()

    if model.is_fine_tuned and model.path:
        config["model_path"] = str(Path(settings.data_dir) / model.path)
        config["base_model"] = model.parent

    if model.get_lora_modules():
        config["lora_target_modules"] = model.get_lora_modules()

    if not config.get("model_name"):
        raise ValueError(f"No configuration found for model: {model_name}")

    return config


async def handle_tool_calls(
    *,
    run_id: str,
    assistant_id: str,
    thread_id: str,
    response_text: str,
    step: RunStepResponse,
    tools: Optional[List[Any]] = None,
) -> Optional[Tuple[ServerSentEvent, ...]]:
    if not tools:
        return None
    parsed_call, _ = parse_xml_tool_call(response_text)
    if not parsed_call:
        return None

    # First check if this is a built-in tool
    builtin_result = await execute_builtin_tool(
        tool_call=parsed_call,
        run_id=run_id,
        assistant_id=assistant_id,
        thread_id=thread_id,
    )

    if builtin_result:
        # Built-in tool was executed
        tool_step, output = builtin_result

        # Update the message creation step
        await update_run_step(
            step.id,
            status="completed",
            step_details={"message_creation": {"message_id": "tool_executed"}},
        )
        completed_evt = await stream_run_step_event("completed", step)

        # Stream the tool step events
        created_evt = await stream_run_step_event("created", tool_step)
        completed_tool_evt = await stream_run_step_event("completed", tool_step)

        # Create a message with the tool output
        message = Message(
            created_at=current_timestamp(),
            thread_id=thread_id,
            role="assistant",
            content=[{"type": "text", "text": {"value": output, "annotations": []}}],
            assistant_id=assistant_id,
            run_id=run_id,
            meta={"tool_output": True},
        )
        await create_message(message)

        # Complete the run
        await update_run(run_id, status="completed")
        status_evt = await stream_run_status(run_id, "completed")

        return completed_evt, created_evt, completed_tool_evt, status_evt

    # Not a built-in tool, require client to submit outputs
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
        created_at=current_timestamp(),
        run_id=run_id,
        assistant_id=assistant_id,
        thread_id=thread_id,
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
    await update_run(run_id, status="requires_action", required_action=required_action)
    status_evt = await stream_run_status(run_id, "requires_action", required_action=required_action)

    return completed_evt, created_evt, status_evt


async def execute_assistant_run_streaming(
    run: RunResponse,
    assistant: AssistantResponse,
) -> AsyncGenerator[ServerSentEvent, None]:
    """
    Execute run events as a server-sent event (SSE) compatible async generator.
    """
    # Start run
    await update_run(run.id, status="in_progress")
    yield await stream_run_status(run.id, "in_progress")
    # Create message creation step inline
    step_entity = RunStep(
        created_at=current_timestamp(),
        run_id=run.id,
        assistant_id=assistant.id,
        thread_id=run.thread_id,
        type="message_creation",
        step_details={"message_creation": {"message_id": None}},
        status="in_progress",
    )
    await create_run_step(step_entity)
    step = RunStepResponse(**step_entity.model_dump())
    yield await stream_run_step_event("created", step)

    # Validate thread
    messages = await get_messages(run.thread_id, order="asc")
    latest_user_msg = find_latest_user_message(messages)
    if latest_user_msg is None:
        async for evt in fail_run(run.id, step, "no_user_message", "No user message found"):
            yield evt
        return

    # Convert thread messages to ChatMessage format
    chat_messages: List[ChatMessage] = []

    # Add assistant instructions as system message if present
    instructions = run.instructions or assistant.instructions
    if instructions:
        chat_messages.append(ChatMessage(role="system", content=instructions))

    # Convert thread messages
    for msg in messages:
        # Extract text content from message
        text_parts = []
        for item in msg.content:
            if item["type"] == "text":
                text_parts.append(item["text"]["value"])
        if text_parts:
            chat_messages.append(ChatMessage(role=msg.role, content="".join(text_parts)))

    # Get model
    try:
        config = await get_model_for_run(run.model or assistant.model)
    except Exception as exc:
        async for evt in fail_run(run.id, step, "model_error", str(exc)):
            yield evt
        return

    # Model inference & streaming with RunsFormatter
    generator = EventGenerator(config)
    message_id = f"msg_{uuid.uuid4().hex}"
    formatter = RunsFormatter(run.id, run.thread_id, assistant.id, message_id)

    # Track usage for final update
    usage = ResponseUsage()
    response_text = ""

    try:
        async for event in generator.generate_events(
            chat_messages,
            temperature=run.temperature or assistant.temperature,
            top_p=run.top_p or assistant.top_p,
            enable_tools=bool(assistant.tools),
            tools=assistant.tools if assistant.tools else None,
        ):
            # Track usage and response text
            if isinstance(event, ResponseStarted):
                usage.prompt_tokens = getattr(event, "prompt_tokens", 0)
            elif isinstance(event, TokenGenerated):
                response_text += event.token
                usage.completion_tokens += 1
            elif isinstance(event, ResponseCompleted):
                if event.output_tokens:
                    usage.completion_tokens = event.output_tokens

            # Format and yield message events
            formatted_event = formatter.format_event(event)
            if formatted_event:
                yield formatted_event
    except Exception as exc:
        logger.exception("Inference error")
        async for evt in fail_run(run.id, step, "execution_error", str(exc)):
            yield evt
        return

    # Tool call detection/handling
    tool_events = await handle_tool_calls(
        run_id=run.id,
        assistant_id=assistant.id,
        thread_id=run.thread_id,
        response_text=response_text,
        step=step,
        tools=assistant.tools,
    )
    if tool_events:
        for evt in tool_events:
            yield evt
        return

    message = Message(
        created_at=current_timestamp(),
        thread_id=run.thread_id,
        role="assistant",
        content=[{"type": "text", "text": {"value": response_text, "annotations": []}}],
        assistant_id=assistant.id,
        run_id=run.id,
        meta={},
    )
    message = await create_message(message)

    # Update step inline
    await update_run_step(
        step.id,
        status="completed",
        step_details={"message_creation": {"message_id": message.id}},
        usage=usage.to_dict(),
    )
    yield await stream_run_step_event("completed", step)

    await update_run(run.id, status="completed", usage=usage.to_dict())
    yield await stream_run_status(run.id, "completed", usage=usage.to_dict())
