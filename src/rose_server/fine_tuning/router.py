"""OpenAI-compatible fine-tuning API endpoints."""

import logging

from fastapi import APIRouter

from rose_server.fine_tuning.events.router import router as events_router
from rose_server.fine_tuning.jobs.router import router as jobs_router

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

router.include_router(jobs_router)
router.include_router(events_router)
