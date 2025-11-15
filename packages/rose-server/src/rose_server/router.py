import logging

from fastapi import APIRouter

from rose_server.routers.embeddings import router as embeddings_router
from rose_server.routers.files import router as files_router
from rose_server.routers.models import router as llms_router
from rose_server.routers.reranker import router as reranker_router
from rose_server.routers.responses import router as responses_router

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(embeddings_router)
router.include_router(llms_router)
router.include_router(responses_router)
router.include_router(files_router)
router.include_router(reranker_router)
