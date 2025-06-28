"""Tool output processing for runs."""

import json
import logging
import uuid
from typing import Any, Callable, Dict, List

from rose_server.assistants.store import get_assistant
from rose_server.database import current_timestamp
from rose_server.entities.messages import Message
from rose_server.entities.run_steps import RunStep
from rose_server.events import ResponseCompleted, ResponseStarted, TokenGenerated
from rose_server.events.generators import RunsGenerator
from rose_server.llms import model_cache
from rose_server.llms.registry import ModelRegistry
from rose_server.messages.store import create_message
from rose_server.runs.steps.store import create_run_step, list_run_steps, update_run_step
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.runs import RunResponse, RunStepResponse
from rose_server.tools.handlers.code_interpreter import intercept_code_interpreter_tool_call

logger = logging.getLogger(__name__)


async def process_tool_outputs(
    run: RunResponse,
    tool_outputs: List[Dict[str, Any]],
    update_run: Callable[..., Any],
    registry: ModelRegistry,
) -> Dict[str, Any]:
    """Process tool outputs and generate continuation response."""
    await update_run(run.id, status="in_progress")
    steps = await list_run_steps(run.id)
    tool_step = next(
        (s for s in steps if s.type == "tool_calls" and s.status == "in_progress"),
        None,
    )

    # Check if we need to intercept code_interpreter calls
    intercepted_outputs = []
    if tool_step and tool_step.step_details.get("tool_calls"):
        for tool_call in tool_step.step_details["tool_calls"]:
            # Parse the tool call
            parsed_call = {
                "tool": tool_call.get("function", {}).get("name"),
                "arguments": json.loads(tool_call.get("function", {}).get("arguments", "{}")),
            }

            # Try to intercept code_interpreter
            result = await intercept_code_interpreter_tool_call(parsed_call, run.assistant_id)
            if result:
                tool_name, output = result
                intercepted_outputs.append({"tool_call_id": tool_call["id"], "output": output})

        # If we intercepted any calls, use those outputs instead
        if intercepted_outputs:
            tool_outputs = intercepted_outputs

    if tool_step:
        await update_run_step(
            tool_step.id,
            status="completed",
            step_details={
                "tool_calls": tool_step.step_details.get("tool_calls", []),
                "tool_outputs": tool_outputs,
            },
        )
    continuation_step_entity = RunStep(
        id=f"step_{uuid.uuid4().hex}",
        created_at=current_timestamp(),
        run_id=run.id,
        assistant_id=run.assistant_id,
        thread_id=run.thread_id,
        type="message_creation",
        step_details={"message_creation": {"message_id": None}},
        status="in_progress",
    )
    await create_run_step(continuation_step_entity)
    continuation_step = RunStepResponse(**continuation_step_entity.model_dump())
    await get_assistant(run.assistant_id)
    tool_results = []
    for output in tool_outputs:
        tool_results.append(f"Tool {output.get('tool_call_id', 'unknown')} result: {output.get('output', '')}")
    tool_results_text = "\n".join(tool_results)
    continuation_prompt = f"Based on the tool results:\n{tool_results_text}\n\nPlease provide a response to the user."
    response_text = "I've processed the tool results."
    usage = {}
    try:
        config = await registry.get_model_config(run.model)
        if config:
            llm = await model_cache.get_model(run.model, config)
            generator = RunsGenerator(llm)
            messages = [ChatMessage(role="user", content=continuation_prompt)]
            response_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            async for event in generator.generate_events(
                messages, temperature=run.temperature, top_p=run.top_p, enable_tools=False
            ):
                if isinstance(event, ResponseStarted):
                    prompt_tokens = getattr(event, "prompt_tokens", 0)
                elif isinstance(event, TokenGenerated):
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
    except Exception as e:
        logger.error(f"Error generating continuation response: {str(e)}")
    message = Message(
        id=f"msg_{uuid.uuid4().hex}",
        created_at=current_timestamp(),
        thread_id=run.thread_id,
        role="assistant",
        content=[{"type": "text", "text": {"value": response_text, "annotations": []}}],
        assistant_id=run.assistant_id,
        run_id=run.id,
        meta={},
    )
    message = await create_message(message)
    await update_run_step(
        continuation_step.id,
        status="completed",
        step_details={"message_creation": {"message_id": message.id}},
        usage=usage,
    )
    await update_run(run.id, status="completed", usage=usage)
    return {
        "continuation_step": continuation_step,
        "message": message,
        "usage": usage,
    }
