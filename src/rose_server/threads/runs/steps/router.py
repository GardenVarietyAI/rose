"""API router for run steps endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Query

from rose_server.schemas.runs import RunStepListResponse, RunStepResponse
from rose_server.threads.runs.steps.store import get_run_step, list_run_steps

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{run_id}/steps", response_model=RunStepListResponse)
async def index(
    thread_id: str,
    run_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Number of steps to retrieve"),
    order: str = Query(default="desc", pattern="^(asc|desc)$", description="Sort order (asc or desc)"),
) -> RunStepListResponse:
    """List steps for a run."""
    try:
        steps = await list_run_steps(run_id, limit=limit, order=order)
        step_data = [RunStepResponse.model_validate(step) for step in steps]
        return RunStepListResponse(
            data=step_data,
            first_id=step_data[0].id if step_data else None,
            last_id=step_data[-1].id if step_data else None,
            has_more=False,
        )
    except Exception as e:
        logger.error(f"Error listing run steps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing run steps: {str(e)}")


@router.get("/{run_id}/steps/{step_id}", response_model=RunStepResponse)
async def get(thread_id: str, run_id: str, step_id: str) -> RunStepResponse:
    """Retrieve a specific run step."""
    try:
        step = await get_run_step(run_id, step_id)
        if not step or step.run_id != run_id:
            raise HTTPException(status_code=404, detail="Step not found")
        return RunStepResponse.model_validate(step)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving run step: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving run step: {str(e)}")
