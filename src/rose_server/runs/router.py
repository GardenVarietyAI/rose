"""API router for runs endpoints."""

import logging
from typing import Any, Dict, Union

from fastapi import APIRouter, Body, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from rose_server.assistants.store import get_assistant
from rose_server.models.deps import ModelRegistryDep
from rose_server.runs.executor import execute_assistant_run_streaming
from rose_server.runs.steps.router import router as steps_router
from rose_server.runs.store import cancel_run, create_run, get_run, list_runs, update_run
from rose_server.runs.tool_outputs import process_tool_outputs
from rose_server.schemas.runs import RunCreateRequest, RunListResponse, RunResponse
from rose_server.threads.store import get_thread

router = APIRouter(prefix="/v1/threads/{thread_id}/runs")
logger = logging.getLogger(__name__)

router.include_router(steps_router)


@router.post("", response_model=None)
async def create(thread_id: str, request: RunCreateRequest = Body(...)) -> Union[RunResponse, EventSourceResponse]:
    """Create a run in a thread."""
    try:
        if not await get_thread(thread_id):
            raise HTTPException(status_code=404, detail="Thread not found")
        assistant = await get_assistant(request.assistant_id)

        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        run = await create_run(
            thread_id=thread_id,
            assistant_id=request.assistant_id,
            model=request.model if request.model is not None else assistant.model,
            instructions=request.instructions or assistant.instructions,
            tools=[
                tool.model_dump() if hasattr(tool, "model_dump") else tool
                for tool in (request.tools or assistant.tools or [])
            ],
            meta=request.metadata or {},
            temperature=request.temperature if request.temperature is not None else assistant.temperature,
            top_p=request.top_p if request.top_p is not None else assistant.top_p,
            max_prompt_tokens=request.max_prompt_tokens,
            max_completion_tokens=request.max_completion_tokens,
            truncation_strategy=request.truncation_strategy,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls if request.parallel_tool_calls is not None else True,
            response_format=request.response_format,
        )

        if request.stream:
            return EventSourceResponse(
                execute_assistant_run_streaming(run, assistant),
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            events = []
            async for event in execute_assistant_run_streaming(run, assistant):
                events.append(event)

            final_run = await get_run(run.id)

            return RunResponse(**final_run.model_dump())
    except Exception as e:
        logger.error(f"Error creating run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating run: {str(e)}")


@router.get("", response_model=RunListResponse)
async def index(
    thread_id: str,
    limit: int = Query(default=20, description="Number of runs to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
) -> RunListResponse:
    """List runs in a thread."""
    if not await get_thread(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")

    try:
        runs = await list_runs(thread_id, limit=limit, order=order)
        run_data = [RunResponse.model_validate(run) for run in runs]
        return RunListResponse(
            data=run_data,
            first_id=run_data[0].id if run_data else None,
            last_id=run_data[-1].id if run_data else None,
            has_more=False,
        )
    except Exception as e:
        logger.error(f"Error listing runs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing runs: {str(e)}")


@router.get("/{run_id}", response_model=RunResponse)
async def get(thread_id: str, run_id: str) -> RunResponse:
    """Retrieve a run."""
    try:
        run = await get_run(run_id)
        if not run or run.thread_id != thread_id:
            raise HTTPException(status_code=404, detail="Run not found")
        return RunResponse.model_validate(run)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving run: {str(e)}")


@router.post("/{run_id}/cancel", response_model=RunResponse)
async def cancel(thread_id: str, run_id: str) -> RunResponse:
    """Cancel a run."""
    try:
        run = await cancel_run(run_id)
        if not run or run.thread_id != thread_id:
            raise HTTPException(status_code=404, detail="Run not found")
        return RunResponse.model_validate(run)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cancelling run: {str(e)}")


@router.post("/{run_id}/submit_tool_outputs", response_model=RunResponse)
async def submit_tool_outputs(
    thread_id: str, run_id: str, request: Dict[str, Any] = Body(...), registry: ModelRegistryDep = None
) -> RunResponse:
    """Submit tool outputs for a run that requires action."""
    tool_outputs = request.get("tool_outputs", [])
    if not tool_outputs:
        raise HTTPException(status_code=400, detail="tool_outputs required")

    run = await get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status != "requires_action":
        raise HTTPException(status_code=400, detail=f"Run is not waiting for tool outputs (status: {run.status})")

    try:
        await process_tool_outputs(run, tool_outputs, update_run, registry)
        return RunResponse.model_validate(run)
    except Exception as e:
        logger.error(f"Error processing tool outputs: {str(e)}")
        await update_run(run_id, status="failed", last_error={"code": "tool_output_error", "message": str(e)})
        raise HTTPException(status_code=500, detail=f"Error processing tool outputs: {str(e)}")
