"""OpenAI-compatible fine-tuning API endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter

from rose_server.fine_tuning.events.router import router as events_router
from rose_server.fine_tuning.jobs.router import router as jobs_router
from rose_server.fine_tuning.store import get_job_status

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

# Include the jobs router
router.include_router(jobs_router)
router.include_router(events_router)


@router.get("/fine_tuning/queue/status")
async def get_queue_status() -> Dict[str, Any]:
    """Get queue status for debugging."""
    return await get_job_status()
