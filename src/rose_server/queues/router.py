import logging
import time
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from ..database import run_in_session
from ..entities.jobs import Job
from ..schemas.jobs import JobUpdateRequest
from ..services import get_job_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["jobs"])


class JobResponse(BaseModel):
    """Job response model."""

    id: int
    type: str
    status: str
    payload: Dict
    created_at: int
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    error: Optional[str] = None


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    type: Optional[str] = Query(None, description="Filter by job type"),
    limit: int = Query(10, description="Max number of jobs to return"),
) -> Dict:
    """List jobs from the queue."""

    async def get_jobs(session):
        query = select(Job)
        if status:
            query = query.where(Job.status == status)
        if type:
            query = query.where(Job.type == type)
        query = query.limit(limit).order_by(Job.created_at.desc())
        result = await session.execute(query)
        jobs = result.scalars().all()
        return [
            JobResponse(
                id=job.id,
                type=job.type,
                status=job.status,
                payload=job.payload,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error=job.error,
            )
            for job in jobs
        ]

    jobs = await run_in_session(get_jobs, read_only=True)
    return {
        "object": "list",
        "data": [job.model_dump() for job in jobs],
        "has_more": len(jobs) == limit,
    }


@router.patch("/jobs/{job_id}")
async def update_job(job_id: int, request: JobUpdateRequest) -> JobResponse:
    """Update job status."""

    # Handle cancellation request through job store
    if request.status in ["cancelling", "cancelled"]:
        job_store = get_job_store()
        success = await job_store.request_cancel(job_id)
        if not success:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled in its current state")
        # Get the updated job
        job = await run_in_session(lambda session: session.get(Job, job_id), read_only=True)
    else:
        # Normal status update
        async def update(session):
            job = await session.get(Job, job_id)
            if not job:
                return None

            job.status = request.status
            if request.status == "running" and not job.started_at:
                job.started_at = int(time.time())
            elif request.status in ["completed", "failed", "cancelled"] and not job.completed_at:
                job.completed_at = int(time.time())

            if request.result:
                job.result = request.result

            await session.commit()
            await session.refresh(job)
            return job

        job = await run_in_session(update)

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


@router.get("/jobs/{job_id}")
async def get_job(job_id: int) -> JobResponse:
    """Get a specific job by ID."""

    async def fetch_job(session):
        return await session.get(Job, job_id)

    job = await run_in_session(fetch_job, read_only=True)
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
