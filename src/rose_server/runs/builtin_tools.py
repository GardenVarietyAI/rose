"""Built-in tool execution for runs."""

import logging
import uuid
from typing import Any, Dict, Optional, Tuple

from rose_server.database import current_timestamp
from rose_server.entities.run_steps import RunStep
from rose_server.runs.steps.store import create_run_step, update_run_step
from rose_server.schemas.runs import RunStepResponse
from rose_server.tools.handlers.code_interpreter import intercept_code_interpreter_tool_call
from rose_server.tools.handlers.retrieval import intercept_retrieval_tool_call

logger = logging.getLogger(__name__)


async def execute_builtin_tool(
    *,
    tool_call: Dict[str, Any],
    run_id: str,
    assistant_id: str,
    thread_id: str,
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
    is_builtin = False
    tool_type = None

    if tool_name == "code_interpreter":
        is_builtin = True
        tool_type = "code_interpreter"
    elif tool_name == "file_search":
        is_builtin = True
        tool_type = "file_search"

    if not is_builtin:
        return None

    # Create tool call step
    step_id = f"step_{uuid.uuid4().hex}"
    logger.info(f"Creating tool call step {step_id} for {tool_name}")

    step_entity = RunStep(
        id=step_id,
        created_at=current_timestamp(),
        run_id=run_id,
        assistant_id=assistant_id,
        thread_id=thread_id,
        type="tool_calls",
        status="in_progress",
        step_details={"tool_calls": []},
    )

    # Execute the tool
    try:
        if tool_type == "code_interpreter":
            # Execute code
            result = await intercept_code_interpreter_tool_call(tool_call, assistant_id)

            if result:
                _, output = result
                # Create proper code interpreter step details
                tool_call_detail = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "code",  # OpenAI uses "code" not "function" for code_interpreter
                    "code": {
                        "input": tool_call.get("arguments", {}).get("code", ""),
                        "outputs": [{"type": "logs", "logs": output}],
                    },
                }
            else:
                raise Exception("Code execution failed")

        else:  # file_search
            result = await intercept_retrieval_tool_call(tool_call, assistant_id)

            if result:
                _, output = result
                # Create retrieval step details
                tool_call_detail = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "file_search",
                    "file_search": {"query": tool_call.get("arguments", {}).get("query", ""), "results": output},
                }
            else:
                raise Exception("Search failed")

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
