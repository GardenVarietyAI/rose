import logging

from fastapi import APIRouter

from rose_server.assistants.router import router as assistants_router
from rose_server.chat.router import router as chat_router
from rose_server.embeddings.router import router as embeddings_router
from rose_server.files.router import router as files_router
from rose_server.fine_tuning.router import router as fine_tuning_router
from rose_server.models.router import router as llms_router
from rose_server.queues.router import router as jobs_router
from rose_server.responses.router import router as responses_router
from rose_server.runs.router import router as runs_router
from rose_server.threads.router import router as threads_router
from rose_server.vector_stores.router import router as vector_stores_router
from rose_server.webhooks.router import router as webhooks_router

router = APIRouter()
logger = logging.getLogger(__name__)
router.include_router(vector_stores_router)
router.include_router(assistants_router)
router.include_router(threads_router)
router.include_router(runs_router)
router.include_router(embeddings_router)
router.include_router(llms_router)
router.include_router(responses_router)
router.include_router(fine_tuning_router)
router.include_router(files_router)
router.include_router(chat_router)
router.include_router(jobs_router)
router.include_router(webhooks_router)
