"""Built-in tool execution for runs."""

import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from rose_server.entities.run_steps import RunStep
from rose_server.runs.steps.store import create_run_step, update_run_step
from rose_server.schemas.runs import RunStepResponse
from rose_server.tools.handlers.file_search import intercept_file_search_tool_call
from rose_server.tools.toolbox import BUILTIN_TOOLS
from rose_server.vector_stores.chroma import Chroma

logger = logging.getLogger(__name__)


async def execute_builtin_tool(
    *,
    tool_call: Dict[str, Any],
    run_id: str,
    assistant_id: str,
    thread_id: str,
    chroma: Chroma = None,
) -> Optional[Tuple[RunStepResponse, str]]:
    """Execute a built-in tool and return the step and output.

    Args:
        tool_call: Parsed tool call with 'tool' and 'arguments'
        run_id: Current run ID
        assistant_id: Assistant ID
        thread_id: Thread ID

    Returns:
        Tuple of (RunStepResponse, output) if built-in tool was executed,
        None if not a built-in tool
    """
    tool_name = tool_call.get("tool", "")

    # Check if it's a built-in tool
    if tool_name not in BUILTIN_TOOLS:
        return None

    tool_config = BUILTIN_TOOLS[tool_name]
    if not tool_config.get("supported", False):
        # Return a friendly error for unsupported tools
        step_entity = RunStep(
            created_at=int(time.time()),
            run_id=run_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            type="tool_calls",
            status="failed",
            step_details={"tool_calls": []},
            last_error={
                "code": "unsupported_tool",
                "message": f"The '{tool_name}' tool is not currently supported. Please use a function tool instead.",
            },
        )
        await create_run_step(step_entity)
        step = RunStepResponse(**step_entity.model_dump())
        return step, f"Error: The '{tool_name}' tool is not currently supported."

    tool_type = tool_name

    # Create tool call step
    step_entity = RunStep(
        created_at=int(time.time()),
        run_id=run_id,
        assistant_id=assistant_id,
        thread_id=thread_id,
        type="tool_calls",
        status="in_progress",
        step_details={"tool_calls": []},
    )
    logger.info(f"Creating tool call step {step_entity.id} for {tool_name}")

    # Execute the tool
    try:
        if tool_type == "file_search":
            result = await intercept_file_search_tool_call(chroma, tool_call, assistant_id)

            if result:
                _, output = result
                # Create file search step details
                tool_call_detail = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "file_search",
                    "file_search": {"query": tool_call.get("arguments", {}).get("query", ""), "results": output},
                }
            else:
                raise Exception("Search failed")

        else:
            raise Exception(f"Tool '{tool_name}' execution not implemented")

        # Update step with tool call details
        step_entity.step_details = {"tool_calls": [tool_call_detail]}
        await create_run_step(step_entity)

        # Mark as completed
        await update_run_step(step_entity.id, status="completed")

        step = RunStepResponse(**step_entity.model_dump())
        return step, output

    except Exception as e:
        logger.error(f"Error executing built-in tool {tool_name}: {str(e)}")
        # Mark step as failed
        await update_run_step(
            step_entity.id, status="failed", last_error={"code": "tool_execution_error", "message": str(e)}
        )
        step = RunStepResponse(**step_entity.model_dump())
        return step, f"Error: {str(e)}"
