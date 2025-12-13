from fastapi import APIRouter

from rose_server.routers.chat import router as chat_router
from rose_server.routers.models import router as models_router

router = APIRouter()
router.include_router(chat_router)
router.include_router(models_router)
