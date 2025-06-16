"""API router for runs endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from rose_server.assistants.store import get_assistant_store
from rose_server.runs.executor import execute_assistant_run_streaming
from rose_server.runs.store import RunsStore
from rose_server.runs.tool_outputs import process_tool_outputs
from rose_server.schemas.assistants import CreateRunRequest
from rose_server.threads.store import ThreadStore

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: CreateRunRequest = Body(...)) -> JSONResponse:
    """Create a run in a thread."""
    try:
        thread_store = ThreadStore()
        if not await thread_store.get_thread(thread_id):
            return JSONResponse(status_code=404, content={"error": "Thread not found"})
        assistant_store = get_assistant_store()
        assistant = await assistant_store.get_assistant(request.assistant_id)
        if not assistant:
            return JSONResponse(status_code=404, content={"error": "Assistant not found"})
        runs_store = RunsStore()
        run = await runs_store.create_run(thread_id, request)
        if not run.model or run.model == "zephyr":
            run.model = assistant.model
        if not run.instructions:
            run.instructions = assistant.instructions
        if not run.tools:
            run.tools = assistant.tools
        if run.temperature is None:
            run.temperature = assistant.temperature
        if run.top_p is None:
            run.top_p = assistant.top_p
        if request.stream:
            return StreamingResponse(
                execute_assistant_run_streaming(run, thread_store, assistant),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            events = []
            async for event in execute_assistant_run_streaming(run, thread_store, assistant):
                events.append(event)
            runs_store = RunsStore()
            final_run = await runs_store.get_run(run.id)
            return JSONResponse(content=final_run.dict())
    except Exception as e:
        logger.error(f"Error creating run: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error creating run: {str(e)}"})


@router.get("/threads/{thread_id}/runs")
async def list_runs(
    thread_id: str,
    limit: int = Query(default=20, description="Number of runs to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
) -> JSONResponse:
    """List runs in a thread."""
    try:
        thread_store = ThreadStore()
        if not await thread_store.get_thread(thread_id):
            return JSONResponse(status_code=404, content={"error": "Thread not found"})
        runs_store = RunsStore()
        runs = await runs_store.list_runs(thread_id, limit=limit, order=order)
        run_data = [run.dict() for run in runs]
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
async def get_run(thread_id: str, run_id: str) -> JSONResponse:
    """Retrieve a run."""
    try:
        runs_store = RunsStore()
        run = await runs_store.get_run(run_id)
        if not run or run.thread_id != thread_id:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        return JSONResponse(content=run.dict())
    except Exception as e:
        logger.error(f"Error retrieving run: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving run: {str(e)}"})


@router.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str) -> JSONResponse:
    """Cancel a run."""
    try:
        runs_store = RunsStore()
        run = await runs_store.cancel_run(run_id)
        if not run or run.thread_id != thread_id:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        return JSONResponse(content=run.dict())
    except Exception as e:
        logger.error(f"Error cancelling run: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error cancelling run: {str(e)}"})


@router.get("/threads/{thread_id}/runs/{run_id}/steps")
async def list_run_steps(
    thread_id: str,
    run_id: str,
    limit: int = Query(default=20, description="Number of steps to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
) -> JSONResponse:
    """List steps for a run."""
    try:
        runs_store = RunsStore()
        run = await runs_store.get_run(run_id)
        if not run or run.thread_id != thread_id:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        steps = await runs_store.list_run_steps(run_id, limit=limit, order=order)
        step_data = [step.dict() for step in steps]
        return JSONResponse(
            content={
                "object": "list",
                "data": step_data,
                "first_id": step_data[0]["id"] if step_data else None,
                "last_id": step_data[-1]["id"] if step_data else None,
                "has_more": False,
            }
        )
    except Exception as e:
        logger.error(f"Error listing run steps: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error listing run steps: {str(e)}"})


@router.get("/threads/{thread_id}/runs/{run_id}/steps/{step_id}")
async def get_run_step(thread_id: str, run_id: str, step_id: str) -> JSONResponse:
    """Retrieve a specific run step."""
    try:
        runs_store = RunsStore()
        run = await runs_store.get_run(run_id)
        if not run or run.thread_id != thread_id:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        step = await runs_store.get_run_step(step_id)
        if not step or step.run_id != run_id:
            return JSONResponse(status_code=404, content={"error": "Step not found"})
        return JSONResponse(content=step.dict())
    except Exception as e:
        logger.error(f"Error retrieving run step: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving run step: {str(e)}"})


@router.post("/threads/{thread_id}/runs/{run_id}/submit_tool_outputs")
async def submit_tool_outputs(thread_id: str, run_id: str, request: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Submit tool outputs for a run that requires action."""
    try:
        tool_outputs = request.get("tool_outputs", [])
        if not tool_outputs:
            return JSONResponse(status_code=400, content={"error": "tool_outputs required"})
        runs_store = RunsStore()
        run = await runs_store.get_run(run_id)
        if not run:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        if run.status != "requires_action":
            return JSONResponse(
                status_code=400,
                content={"error": f"Run is not waiting for tool outputs (status: {run.status})"},
            )
        try:
            await process_tool_outputs(run, tool_outputs, runs_store)
            return JSONResponse(content=run.dict())
        except Exception as e:
            logger.error(f"Error processing tool outputs: {str(e)}")
            await runs_store.update_run_status(
                run_id, "failed", last_error={"code": "tool_output_error", "message": str(e)}
            )
            raise
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        logger.error(f"Error submitting tool outputs: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
