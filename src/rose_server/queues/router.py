import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas.jobs import JobResponse, JobUpdateRequest
from .store import fetch_job, get_jobs, request_cancel, update_job_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["jobs"])


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    type: Optional[str] = Query(None, description="Filter by job type"),
    limit: int = Query(10, description="Max number of jobs to return"),
) -> Dict[str, Any]:
    """List jobs from the queue."""
    jobs = await get_jobs(status=status, type=type, limit=limit)
    return {
        "object": "list",
        "data": [job.model_dump() for job in jobs],
        "has_more": len(jobs) == limit,
    }


@router.patch("/jobs/{job_id}")
async def update_job(job_id: int, request: JobUpdateRequest) -> JobResponse:
    """Update job status."""
    job = await fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Handle cancellation request through job store
    if request.status in ["cancelling", "cancelled"]:
        success = await request_cancel(job_id)
        if not success:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled in its current state")
        # Re-fetch to get updated state
        job = await fetch_job(job_id)
    else:
        # Normal status update
        job = await update_job_status(job_id, request.status, request.result)

    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        payload=job.payload,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )


@router.get("/jobs/{job_id}")
async def get_job(job_id: int) -> JobResponse:
    """Get a specific job by ID."""

    job = await fetch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        payload=job.payload,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )
