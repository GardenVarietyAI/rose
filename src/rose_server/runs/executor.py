"""Run execution logic."""

import json
import logging
import uuid
from typing import AsyncGenerator

from rose_server.events import ResponseCompleted, ResponseStarted, TokenGenerated
from rose_server.events.generators import RunsGenerator
from rose_server.language_models import model_cache
from rose_server.runs.store import RunsStore
from rose_server.runs.streaming import (
    stream_message_chunk,
    stream_message_completed,
    stream_message_created,
    stream_message_in_progress,
    stream_run_status,
    stream_run_step_event,
)
from rose_server.runs.utils import find_latest_user_message
from rose_server.schemas.assistants import Run
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.runs import RunStepType
from rose_server.services import get_model_registry
from rose_server.tools import format_tools_for_prompt, parse_xml_tool_call

logger = logging.getLogger(__name__)


def _build_conversation_context(messages, limit=5):
    """Build conversation context from recent messages."""
    context_messages = []
    for msg in messages[-limit:]:
        text_content = ""
        for content_item in msg.content:
            if content_item.type == "text":
                text_content += content_item.text.value
        if text_content:
            context_messages.append(f"{msg.role}: {text_content}")
    return "\n".join(context_messages) if context_messages else None


async def _build_prompt(run, assistant, messages, latest_user_message):
    """Build the full prompt including instructions, context, and tools."""
    prompt_parts = []
    instructions = run.instructions or assistant.instructions
    if instructions:
        prompt_parts.append(f"Instructions: {instructions}")
    conversation_context = _build_conversation_context(messages)
    if conversation_context:
        prompt_parts.append(f"Recent conversation:\n{conversation_context}")
    if assistant.tools:
        tool_prompt = format_tools_for_prompt(assistant.tools, assistant_id=assistant.id)
        if tool_prompt:
            prompt_parts.append(tool_prompt)
    prompt_parts.append(f"\nUser: {latest_user_message}")
    return "\n\n".join(prompt_parts)


async def _handle_tool_calls(run, assistant, response_text, step, runs_store):
    """Check for tool calls and handle them if present."""
    if not assistant.tools:
        return False
    parsed_call, cleaned_text = parse_xml_tool_call(response_text)
    if not parsed_call:
        return False
    tool_calls = []
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": parsed_call["tool"], "arguments": json.dumps(parsed_call["arguments"])},
    }
    tool_calls.append(tool_call)
    await runs_store.update_run_step(
        step.id,
        status="completed",
        step_details={"message_creation": {"message_id": "pending_tools"}},
    )
    await runs_store.create_run_step(
        run_id=run.id,
        assistant_id=assistant.id,
        thread_id=run.thread_id,
        step_type=RunStepType.TOOL_CALLS,
        step_details={"tool_calls": tool_calls},
    )
    await runs_store.update_run_status(
        run.id,
        "requires_action",
        required_action={"type": "submit_tool_outputs", "submit_tool_outputs": {"tool_calls": tool_calls}},
    )
    logger.info(f"Run {run.id} requires action for tool calls")
    return True


async def execute_assistant_run_streaming(run: Run, thread_store, assistant) -> AsyncGenerator[str, None]:
    """Execute an assistant run with streaming output."""
    runs_store = RunsStore()
    step = None
    try:
        yield await stream_run_status(run.id, "in_progress")
        await runs_store.update_run_status(run.id, "in_progress")
        step = await runs_store.create_run_step(
            run_id=run.id,
            assistant_id=assistant.id,
            thread_id=run.thread_id,
            step_type=RunStepType.MESSAGE_CREATION,
            step_details={"message_creation": {"message_id": None}},
        )
        yield await stream_run_step_event("created", step)
        messages = await thread_store.get_messages(run.thread_id, order="asc")
        latest_user_message = find_latest_user_message(messages)
        if not latest_user_message:
            error = {"code": "no_user_message", "message": "No user message found"}
            await runs_store.update_run_status(run.id, "failed", last_error=error)
            await runs_store.update_run_step(step.id, status="failed", last_error=error)
            yield await stream_run_status(run.id, "failed", last_error=error)
            return
        full_prompt = await _build_prompt(run, assistant, messages, latest_user_message)
        requested_model = run.model or assistant.model
        temperature = run.temperature or assistant.temperature
        top_p = run.top_p or assistant.top_p
        registry = get_model_registry()
        if requested_model not in registry.list_models():
            error = {"code": "model_not_found", "message": f"Model '{requested_model}' not found"}
            await runs_store.update_run_status(run.id, "failed", last_error=error)
            await runs_store.update_run_step(step.id, status="failed", last_error=error)
            yield await stream_run_status(run.id, "failed", last_error=error)
            return
        config = registry.get_model_config(requested_model)
        if not config:
            error = {"code": "model_error", "message": f"No configuration found for model: {requested_model}"}
            await runs_store.update_run_status(run.id, "failed", last_error=error)
            await runs_store.update_run_step(step.id, status="failed", last_error=error)
            yield await stream_run_status(run.id, "failed", last_error=error)
            return
        try:
            llm = await model_cache.get_model(requested_model, config)
        except Exception as e:
            error = {"code": "model_error", "message": f"Failed to create model: {str(e)}"}
            await runs_store.update_run_status(run.id, "failed", last_error=error)
            await runs_store.update_run_step(step.id, status="failed", last_error=error)
            yield await stream_run_status(run.id, "failed", last_error=error)
            return
        message_id = f"msg_{uuid.uuid4().hex}"
        generator = RunsGenerator(llm)
        yield await stream_message_created(message_id, run.thread_id, assistant.id, run.id)
        yield await stream_message_in_progress(message_id, run.thread_id, assistant.id, run.id)
        messages = [ChatMessage(role="user", content=full_prompt)]
        enable_tools = bool(assistant.tools)
        tools = assistant.tools if enable_tools else None
        response_text = ""
        usage = {}
        prompt_tokens = 0
        completion_tokens = 0
        async for event in generator.generate_events(
            messages, temperature=temperature, top_p=top_p, enable_tools=enable_tools, tools=tools
        ):
            if isinstance(event, ResponseStarted):
                prompt_tokens = getattr(event, "prompt_tokens", 0)
            elif isinstance(event, TokenGenerated):
                yield await stream_message_chunk(run.id, message_id, event.token)
                response_text += event.token
                completion_tokens += 1
            elif isinstance(event, ResponseCompleted):
                if event.output_tokens:
                    completion_tokens = event.output_tokens
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
        if assistant.tools:
            parsed_call, cleaned_text = parse_xml_tool_call(response_text)
            if parsed_call:
                tool_calls = []
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": parsed_call["tool"], "arguments": json.dumps(parsed_call["arguments"])},
                }
                tool_calls.append(tool_call)
                await runs_store.update_run_step(
                    step.id,
                    status="completed",
                    step_details={"message_creation": {"message_id": "pending_tools"}},
                )
                yield await stream_run_step_event("completed", step)
                tool_step = await runs_store.create_run_step(
                    run_id=run.id,
                    assistant_id=assistant.id,
                    thread_id=run.thread_id,
                    step_type=RunStepType.TOOL_CALLS,
                    step_details={"tool_calls": tool_calls},
                )
                yield await stream_run_step_event("created", tool_step)
                required_action = {"type": "submit_tool_outputs", "submit_tool_outputs": {"tool_calls": tool_calls}}
                await runs_store.update_run_status(run.id, "requires_action", required_action=required_action)
                yield await stream_run_status(run.id, "requires_action", required_action=required_action)
                return
        yield await stream_message_completed(message_id, response_text, run.thread_id, assistant.id, run.id)
        message = await thread_store.create_message(
            thread_id=run.thread_id,
            role="assistant",
            content=[{"type": "text", "text": response_text}],
            metadata={"run_id": run.id, "assistant_id": run.assistant_id},
        )
        await runs_store.update_run_step(
            step.id,
            status="completed",
            step_details={"message_creation": {"message_id": message.id}},
            usage=usage,
        )
        yield await stream_run_step_event("completed", step)
        yield await stream_run_status(run.id, "completed", usage=usage)
        await runs_store.update_run_status(run.id, "completed", usage=usage)
    except Exception as e:
        logger.error(f"Error in streaming run {run.id}: {str(e)}", exc_info=True)
        if step:
            await runs_store.update_run_step(
                step.id, status="failed", last_error={"code": "execution_error", "message": str(e)}
            )
            yield await stream_run_step_event("failed", step)
        yield await stream_run_status(run.id, "failed", last_error={"code": "execution_error", "message": str(e)})
        await runs_store.update_run_status(run.id, "failed", last_error={"code": "execution_error", "message": str(e)})
