"""API router for run steps endpoints."""

import logging

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from rose_server.runs.steps.store import get_run_step, list_run_steps
from rose_server.schemas.runs import RunStep

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.get("/threads/{thread_id}/runs/{run_id}/steps")
async def index(
    thread_id: str,
    run_id: str,
    limit: int = Query(default=20, description="Number of steps to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
) -> JSONResponse:
    """List steps for a run."""
    try:
        steps = await list_run_steps(run_id, limit=limit, order=order)
        step_data = [RunStep(**step.model_dump()).model_dump() for step in steps]
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
async def get(thread_id: str, run_id: str, step_id: str) -> JSONResponse:
    """Retrieve a specific run step."""
    try:
        step = await get_run_step(run_id, step_id)
        if not step or step.run_id != run_id:
            return JSONResponse(status_code=404, content={"error": "Step not found"})
        return JSONResponse(content=RunStep(**step.model_dump()).model_dump())
    except Exception as e:
        logger.error(f"Error retrieving run step: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving run step: {str(e)}"})
