"""OpenAI-compatible fine-tuning API endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter

from .jobs.router import router as jobs_router
from .store import get_job_status

router = APIRouter()
logger = logging.getLogger(__name__)

# Include the jobs router
router.include_router(jobs_router)


@router.get("/v1/fine_tuning/queue/status")
async def get_queue_status() -> Dict[str, Any]:
    """Get queue status for debugging."""
    return await get_job_status()
