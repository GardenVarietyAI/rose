"""Tool output processing for runs."""

import logging
from typing import Any, Dict, List

from rose_server.assistants.store import get_assistant_store
from rose_server.events import ResponseCompleted, ResponseStarted, TokenGenerated
from rose_server.events.generators import RunsGenerator
from rose_server.llms.huggingface_llm import HuggingFaceLLM
from rose_server.runs.store import RunsStore
from rose_server.schemas.assistants import Run
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.runs import RunStepType
from rose_server.services import get_model_registry
from rose_server.threads.store import ThreadStore

logger = logging.getLogger(__name__)


async def process_tool_outputs(run: Run, tool_outputs: List[Dict[str, Any]], runs_store: RunsStore) -> Dict[str, Any]:
    """Process tool outputs and generate continuation response."""
    await runs_store.update_run_status(run.id, "in_progress")
    steps = await runs_store.list_run_steps(run.id)
    tool_step = next(
        (s for s in steps if s.type == RunStepType.TOOL_CALLS and s.status == "in_progress"),
        None,
    )
    if tool_step:
        await runs_store.update_run_step(
            tool_step.id,
            status="completed",
            step_details={
                "tool_calls": tool_step.step_details.get("tool_calls", []),
                "tool_outputs": tool_outputs,
            },
        )
    continuation_step = await runs_store.create_run_step(
        run_id=run.id,
        assistant_id=run.assistant_id,
        thread_id=run.thread_id,
        step_type=RunStepType.MESSAGE_CREATION,
        step_details={"message_creation": {"message_id": None}},
    )
    thread_store = ThreadStore()
    assistant_store = get_assistant_store()
    await assistant_store.get_assistant(run.assistant_id)
    tool_results = []
    for output in tool_outputs:
        tool_results.append(f"Tool {output.get('tool_call_id', 'unknown')} result: {output.get('output', '')}")
    tool_results_text = "\n".join(tool_results)
    continuation_prompt = f"Based on the tool results:\n{tool_results_text}\n\nPlease provide a response to the user."
    response_text = "I've processed the tool results."
    usage = {}
    try:
        registry = get_model_registry()
        config = registry.get_model_config(run.model)
        if config:
            llm = HuggingFaceLLM(config)
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
    message = await thread_store.create_message(
        thread_id=run.thread_id,
        role="assistant",
        content=[{"type": "text", "text": response_text}],
        metadata={"run_id": run.id, "assistant_id": run.assistant_id},
    )
    await runs_store.update_run_step(
        continuation_step.id,
        status="completed",
        step_details={"message_creation": {"message_id": message.id}},
        usage=usage,
    )
    await runs_store.update_run_status(run.id, "completed", usage=usage)
    return {
        "continuation_step": continuation_step,
        "message": message,
        "usage": usage,
    }
