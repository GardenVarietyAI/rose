import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from sse_starlette import ServerSentEvent
from tokenizers import Tokenizer

from rose_server.entities.messages import Message
from rose_server.entities.run_steps import RunStep
from rose_server.events.event_types import TokenGenerated
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
from rose_server.types.models import ModelConfig
from rose_server.vector_stores.chroma import Chroma

logger = logging.getLogger(__name__)


@dataclass
class ResponseUsage:
    """Track token usage for responses."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def create_with_tokenizer(
        cls, tokenizer: Tokenizer, prompt_text: str = "", completion_text: str = ""
    ) -> "ResponseUsage":
        """Create usage with accurate token counting."""
        return cls(
            prompt_tokens=len(tokenizer.encode(prompt_text).ids) if prompt_text else 0,
            completion_tokens=len(tokenizer.encode(completion_text).ids) if completion_text else 0,
        )


async def stream_run_step_event(event_type: str, step: RunStepResponse) -> ServerSentEvent:
    """Create a streaming event for run step updates."""
    return ServerSentEvent(data=json.dumps(step.model_dump()), event=f"thread.run.step.{event_type}")


async def stream_run_status(run_id: str, status: str, **kwargs: Dict[str, Any]) -> ServerSentEvent:
    """Create a streaming event for run status updates."""
    return ServerSentEvent(
        data=json.dumps(
            {
                "id": run_id,
                "object": f"thread.run.{status}",
                "delta": {"status": status, **kwargs},
            }
        ),
        event=f"thread.run.{status}",
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
    if step:
        step_evt = await stream_run_step_event("failed", step)
        yield step_evt
    yield status_evt


async def handle_tool_calls(
    *,
    run_id: str,
    assistant_id: str,
    thread_id: str,
    response_text: str,
    step: RunStepResponse,
    tools: Optional[List[Any]] = None,
    chroma: Chroma = None,
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
        chroma=chroma,
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
            created_at=int(time.time()),
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
        created_at=int(time.time()),
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
    chroma: Chroma = None,
) -> AsyncGenerator[ServerSentEvent, None]:
    """
    Execute run events as a server-sent event (SSE) compatible async generator.
    """
    # Start run
    await update_run(run.id, status="in_progress")
    yield await stream_run_status(run.id, "in_progress")
    # Create message creation step inline
    step_entity = RunStep(
        created_at=int(time.time()),
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

    # Get model and tokenizer
    try:
        model_name = run.model or assistant.model
        model = await get_language_model(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")
        config = ModelConfig.from_language_model(model)

        # Initialize tokenizer for accurate token counting
        tokenizer = Tokenizer.from_pretrained(model_name)
    except Exception as exc:
        async for evt in fail_run(run.id, step, "model_error", str(exc)):
            yield evt
        return

    # Model inference & streaming with RunsFormatter
    generator = EventGenerator(config)
    message_id = f"msg_{uuid.uuid4().hex}"
    formatter = RunsFormatter(run.id, run.thread_id, assistant.id, message_id)

    # Build prompt text for accurate token counting
    prompt_text = ""
    for msg in chat_messages:
        prompt_text += f"{msg.role}: {msg.content}\n"

    # Track usage with accurate initial counting
    usage = ResponseUsage.create_with_tokenizer(tokenizer, prompt_text=prompt_text)
    response_text = ""

    try:
        async for event in generator.generate_events(
            chat_messages,
            temperature=run.temperature or assistant.temperature,
            top_p=run.top_p or assistant.top_p,
            enable_tools=bool(assistant.tools),
            tools=assistant.tools if assistant.tools else None,
        ):
            # Track response text and handle events
            if isinstance(event, TokenGenerated):
                response_text += event.token

            # Format and yield message events
            formatted_event = formatter.format_event(event)
            if formatted_event:
                yield formatted_event
    except Exception as exc:
        logger.exception("Inference error")
        async for evt in fail_run(run.id, step, "execution_error", str(exc)):
            yield evt
        return

    # Update completion tokens with accurate count
    usage.completion_tokens = len(tokenizer.encode(response_text).ids) if response_text else 0

    # Tool call detection/handling
    tool_events = await handle_tool_calls(
        run_id=run.id,
        assistant_id=assistant.id,
        thread_id=run.thread_id,
        response_text=response_text,
        step=step,
        tools=assistant.tools,
        chroma=chroma,
    )
    if tool_events:
        for evt in tool_events:
            yield evt
        return

    message = Message(
        created_at=int(time.time()),
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
