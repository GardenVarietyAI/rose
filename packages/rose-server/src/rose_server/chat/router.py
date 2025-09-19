import logging

from fastapi import APIRouter

from rose_server.chat.completions.router import router as completions_router

router = APIRouter(prefix="/v1/chat")
logger = logging.getLogger(__name__)

router.include_router(completions_router)
