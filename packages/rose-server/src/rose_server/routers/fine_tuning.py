import logging

from fastapi import APIRouter
from rose_server.routers.fine_tuning_events import router as events_router
from rose_server.routers.fine_tuning_jobs import router as jobs_router

router = APIRouter(prefix="/v1/fine_tuning")
logger = logging.getLogger(__name__)

router.include_router(jobs_router)
router.include_router(events_router)
