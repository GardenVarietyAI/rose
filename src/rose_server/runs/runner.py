import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from sse_starlette import ServerSentEvent
from tokenizers import Tokenizer

from rose_server.entities.messages import Message
from rose_server.entities.run_steps import RunStep
from rose_server.events.event_types import ResponseCompleted, ResponseStarted, TokenGenerated
from rose_server.events.formatters.runs import RunsFormatter
from rose_server.events.generator import EventGenerator
from rose_server.models.registry import ModelRegistry
from rose_server.models.store import get as get_language_model
from rose_server.runs.steps.store import create_run_step, update_run_step
from rose_server.runs.store import update_run
from rose_server.schemas.assistants import AssistantResponse
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.runs import RunResponse, RunStepResponse
from rose_server.threads.messages.store import create_message, get_messages
from rose_server.tools import parse_xml_tool_call
from rose_server.tools.handlers.file_search import intercept_file_search_tool_call
from rose_server.tools.toolbox import BUILTIN_TOOLS
from rose_server.types.models import ModelConfig
from rose_server.vector_stores.chroma import Chroma

logger = logging.getLogger(__name__)


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


def _format_messages(messages: List[Message], instructions: Optional[str]) -> List[ChatMessage]:
    chat_messages = []
    if instructions:
        chat_messages.append(ChatMessage(role="system", content=instructions))
    for msg in messages:
        text_parts = [item["text"]["value"] for item in msg.content if item["type"] == "text"]
        if text_parts:
            chat_messages.append(ChatMessage(role=msg.role, content="".join(text_parts)))
    return chat_messages


def find_latest_user_message(messages: List[Message]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        for item in msg.content:
            if item["type"] == "text":
                return str(item["text"]["value"])
    return None


async def process_tool_outputs(
    model: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
    tool_outputs: List[Dict[str, Any]],
    registry: ModelRegistry,
) -> Dict[str, Union[str, int]]:
    """Process tool outputs and generate continuation response."""

    tool_results = []
    for output in tool_outputs:
        tool_call_id = output.get("tool_call_id", "unknown")
        tool_output = output.get("output", "")
        tool_results.append(f"Tool {tool_call_id} result: {tool_output}")

    tool_results_text = "\n".join(tool_results)
    continuation_prompt = f"Based on the tool results:\n{tool_results_text}\n\nPlease provide a response to the user."
    usage: Dict[str, Union[str, int]] = {"response_text": "I've processed the tool results."}

    try:
        config = await registry.get_model_config(model)
        if config:
            generator = EventGenerator(config)
            messages = [ChatMessage(role="user", content=continuation_prompt)]
            usage["response_text"] = ""
            prompt_tokens = 0
            completion_tokens = 0

            async for event in generator.generate_events(
                messages, temperature=temperature, top_p=top_p, enable_tools=False
            ):
                if isinstance(event, ResponseStarted):
                    prompt_tokens = getattr(event, "prompt_tokens", 0)
                elif isinstance(event, TokenGenerated):
                    usage["response_text"] += event.token
                    completion_tokens += 1
                elif isinstance(event, ResponseCompleted):
                    if event.output_tokens:
                        completion_tokens = event.output_tokens

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        return usage
    except Exception as e:
        logger.error(f"Error generating continuation response: {str(e)}")
        raise e


async def execute_assistant_run_streaming(
    run: RunResponse,
    assistant: AssistantResponse,
    chroma: Chroma = None,
) -> AsyncGenerator[ServerSentEvent, None]:
    """
    Execute run events as a server-sent event (SSE) compatible async generator.
    """
    await update_run(run.id, status="in_progress")
    yield await stream_run_status(run.id, "in_progress")

    step_entity = RunStep(
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

    messages = await get_messages(run.thread_id, order="asc")
    latest_user_msg = find_latest_user_message(messages)
    if latest_user_msg is None:
        async for evt in fail_run(run.id, step, "no_user_message", "No user message found"):
            yield evt
        return

    chat_messages = _format_messages(messages=messages, instructions=run.instructions or assistant.instructions)

    try:
        model_name = run.model or assistant.model
        model = await get_language_model(model_name)
        config = ModelConfig.from_language_model(model)
        tokenizer = Tokenizer.from_pretrained(model_name)
    except Exception as e:
        async for evt in fail_run(run.id, step, "model_error", str(e)):
            yield evt
        return

    generator = EventGenerator(config)
    formatter = RunsFormatter(run.id, run.thread_id, assistant.id)

    prompt_text = "\n".join(f"{msg.role}: {msg.content}" for msg in chat_messages)
    prompt_tokens = len(tokenizer.encode(prompt_text).ids) if prompt_text else 0
    completion_tokens = 0
    total_tokens = prompt_tokens + completion_tokens

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    response_text = ""

    try:
        async for event in generator.generate_events(
            chat_messages,
            temperature=run.temperature or assistant.temperature,
            top_p=run.top_p or assistant.top_p,
            enable_tools=bool(assistant.tools),
            tools=assistant.tools if assistant.tools else None,
        ):
            if isinstance(event, TokenGenerated):
                response_text += event.token
            formatted_event = formatter.format_event(event)
            if formatted_event:
                yield formatted_event
    except Exception as e:
        logger.exception("Inference error")
        async for evt in fail_run(run.id, step, "execution_error", str(e)):
            yield evt
        return

    usage["completion_tokens"] = len(tokenizer.encode(response_text).ids) if response_text else 0

    tool_call, _ = parse_xml_tool_call(response_text)
    if tool_call:
        tool_name = tool_call.get("tool", "")
        if tool_name in BUILTIN_TOOLS:
            tool_config = BUILTIN_TOOLS[tool_name]
            step_entity = RunStep(
                run_id=run.id,
                assistant_id=assistant.id,
                thread_id=run.thread_id,
                type="tool_calls",
                status="in_progress",
                step_details={"tool_calls": []},
            )

            if not tool_config.get("supported", False):
                step_entity.status = "failed"
                step_entity.last_error = {
                    "code": "unsupported_tool",
                    "message": f"The '{tool_name}' tool is not currently supported. Please use a function tool instead.",
                }
                await create_run_step(step_entity)
                await update_run_step(
                    step.id, status="completed", step_details={"message_creation": {"message_id": "tool_executed"}}
                )
                await update_run(run.id, status="completed")
                yield await stream_run_step_event("completed", step)
                yield await stream_run_step_event("created", RunStepResponse(**step_entity.model_dump()))
                yield await stream_run_step_event("completed", RunStepResponse(**step_entity.model_dump()))
                yield await stream_run_status(run.id, "completed")
                return

            await create_run_step(step_entity)
            try:
                if tool_name == "file_search":
                    result = await intercept_file_search_tool_call(chroma, tool_call, assistant.id)
                    if result:
                        _, output = result
                        tool_call_detail = {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "file_search",
                            "file_search": {
                                "query": tool_call.get("arguments", {}).get("query", ""),
                                "results": output,
                            },
                        }
                    else:
                        raise Exception("Search failed")
                else:
                    raise Exception(f"Tool '{tool_name}' execution not implemented")

                step_entity.step_details = {"tool_calls": [tool_call_detail]}
                await update_run_step(step_entity.id, status="completed", step_details=step_entity.step_details)

                await update_run_step(
                    step.id, status="completed", step_details={"message_creation": {"message_id": "tool_executed"}}
                )
                yield await stream_run_step_event("completed", step)

                tool_step = RunStepResponse(**step_entity.model_dump())
                yield await stream_run_step_event("created", tool_step)
                yield await stream_run_step_event("completed", tool_step)

                message = Message(
                    thread_id=run.thread_id,
                    role="assistant",
                    content=[{"type": "text", "text": {"value": output, "annotations": []}}],
                    assistant_id=assistant.id,
                    run_id=run.id,
                    meta={"tool_output": True},
                )
                await create_message(message)

                await update_run(run.id, status="completed")
                yield await stream_run_status(run.id, "completed")
                return

            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                await update_run_step(
                    step_entity.id,
                    status="failed",
                    last_error={"code": "tool_execution_error", "message": str(e)},
                )
                await update_run_step(
                    step.id, status="completed", step_details={"message_creation": {"message_id": "tool_executed"}}
                )
                yield await stream_run_step_event("completed", step)
                yield await stream_run_step_event("created", RunStepResponse(**step_entity.model_dump()))
                yield await stream_run_step_event("completed", RunStepResponse(**step_entity.model_dump()))
                await update_run(run.id, status="completed")
                yield await stream_run_status(run.id, "completed")
                return

        # Non-built-in tool - trigger client tool output
        tool_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_call["tool"],
                    "arguments": json.dumps(tool_call["arguments"]),
                },
            }
        ]
        await update_run_step(
            step.id, status="completed", step_details={"message_creation": {"message_id": "pending_tools"}}
        )
        yield await stream_run_step_event("completed", step)

        tool_step = RunStep(
            run_id=run.id,
            assistant_id=assistant.id,
            thread_id=run.thread_id,
            type="tool_calls",
            step_details={"tool_calls": tool_calls},
            status="in_progress",
        )
        await create_run_step(tool_step)
        yield await stream_run_step_event("created", RunStepResponse(**tool_step.model_dump()))

        required_action = {
            "type": "submit_tool_outputs",
            "submit_tool_outputs": {"tool_calls": tool_calls},
        }
        await update_run(run.id, status="requires_action", required_action=required_action)
        yield await stream_run_status(run.id, "requires_action", required_action=required_action)
        return

    # No tool calls - proceed with assistant message
    message = Message(
        thread_id=run.thread_id,
        role="assistant",
        content=[{"type": "text", "text": {"value": response_text, "annotations": []}}],
        assistant_id=assistant.id,
        run_id=run.id,
        meta={},
    )
    message = await create_message(message)

    await update_run_step(
        step.id,
        status="completed",
        step_details={"message_creation": {"message_id": message.id}},
        usage=usage,
    )
    yield await stream_run_step_event("completed", step)

    await update_run(run.id, status="completed", usage=usage)
    yield await stream_run_status(run.id, "completed", usage=usage)
