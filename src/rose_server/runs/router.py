"""API router for runs endpoints."""

import logging
from typing import Any, Dict, Union

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from rose_server.assistants.store import get_assistant
from rose_server.database import current_timestamp
from rose_server.entities.runs import Run
from rose_server.llms.deps import ModelRegistryDep
from rose_server.runs.executor import execute_assistant_run_streaming
from rose_server.runs.store import cancel_run, create_run, get_run, list_runs, update_run
from rose_server.runs.tool_outputs import process_tool_outputs
from rose_server.schemas.runs import RunCreateRequest, RunResponse
from rose_server.threads.store import get_thread

from .steps.router import router as steps_router

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

# Include the steps router
router.include_router(steps_router)


@router.post("/threads/{thread_id}/runs", response_model=None)
async def create(thread_id: str, request: RunCreateRequest = Body(...)) -> Union[JSONResponse, EventSourceResponse]:
    """Create a run in a thread."""
    try:
        if not await get_thread(thread_id):
            return JSONResponse(status_code=404, content={"error": "Thread not found"})

        assistant = await get_assistant(request.assistant_id)

        if not assistant:
            return JSONResponse(status_code=404, content={"error": "Assistant not found"})

        run = Run(
            created_at=current_timestamp(),
            thread_id=thread_id,
            assistant_id=request.assistant_id,
            status="queued",
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

        run = await create_run(run)

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

            return JSONResponse(content=RunResponse(**final_run.model_dump()).model_dump())
    except Exception as e:
        logger.error(f"Error creating run: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error creating run: {str(e)}"})


@router.get("/threads/{thread_id}/runs")
async def index(
    thread_id: str,
    limit: int = Query(default=20, description="Number of runs to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
) -> JSONResponse:
    """List runs in a thread."""
    try:
        if not await get_thread(thread_id):
            return JSONResponse(status_code=404, content={"error": "Thread not found"})

        runs = await list_runs(thread_id, limit=limit, order=order)
        run_data = [RunResponse(**run.model_dump()).model_dump() for run in runs]

        return JSONResponse(
            content={
                "object": "list",
                "data": run_data,
                "first_id": run_data[0]["id"] if run_data else None,
                "last_id": run_data[-1]["id"] if run_data else None,
                "has_more": False,
            }
        )
    except Exception as e:
        logger.error(f"Error listing runs: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error listing runs: {str(e)}"})


@router.get("/threads/{thread_id}/runs/{run_id}")
async def get(thread_id: str, run_id: str) -> JSONResponse:
    """Retrieve a run."""
    try:
        run = await get_run(run_id)

        if not run or run.thread_id != thread_id:
            return JSONResponse(status_code=404, content={"error": "Run not found"})

        return JSONResponse(content=RunResponse(**run.model_dump()).model_dump())
    except Exception as e:
        logger.error(f"Error retrieving run: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving run: {str(e)}"})


@router.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel(thread_id: str, run_id: str) -> JSONResponse:
    """Cancel a run."""
    try:
        run = await cancel_run(run_id)

        if not run or run.thread_id != thread_id:
            return JSONResponse(status_code=404, content={"error": "Run not found"})

        return JSONResponse(content=RunResponse(**run.model_dump()).model_dump())
    except Exception as e:
        logger.error(f"Error cancelling run: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error cancelling run: {str(e)}"})


@router.post("/threads/{thread_id}/runs/{run_id}/submit_tool_outputs")
async def submit_tool_outputs(
    thread_id: str, run_id: str, request: Dict[str, Any] = Body(...), registry: ModelRegistryDep = None
) -> JSONResponse:
    """Submit tool outputs for a run that requires action."""
    try:
        tool_outputs = request.get("tool_outputs", [])
        if not tool_outputs:
            return JSONResponse(status_code=400, content={"error": "tool_outputs required"})

        run = await get_run(run_id)
        if not run:
            return JSONResponse(status_code=404, content={"error": "Run not found"})

        if run.status != "requires_action":
            return JSONResponse(
                status_code=400,
                content={"error": f"Run is not waiting for tool outputs (status: {run.status})"},
            )

        try:
            await process_tool_outputs(run, tool_outputs, update_run, registry)
            return JSONResponse(content=RunResponse(**run.model_dump()).model_dump())
        except Exception as e:
            logger.error(f"Error processing tool outputs: {str(e)}")
            await update_run(run_id, status="failed", last_error={"code": "tool_output_error", "message": str(e)})
            return JSONResponse(status_code=500, content={"error": f"Error processing tool outputs: {str(e)}"})

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        logger.error(f"Error submitting tool outputs: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
