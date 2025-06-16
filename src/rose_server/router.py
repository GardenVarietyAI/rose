import logging

from fastapi import APIRouter

from rose_server.assistants.router import router as assistants_router
from rose_server.chat_completions.router import router as chat_completions_router
from rose_server.completions.router import router as completions_router
from rose_server.embeddings.router import router as embeddings_router
from rose_server.evals.router import router as evals_router
from rose_server.files.router import router as files_router
from rose_server.fine_tuning.router import router as fine_tuning_router
from rose_server.llms.router import router as llms_router
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
router.include_router(chat_completions_router)
router.include_router(completions_router)
router.include_router(evals_router)
router.include_router(jobs_router)
router.include_router(webhooks_router)
@router.post("/v1/traces")

async def create_trace():
    """Dummy trace endpoint to prevent SDK errors."""
    return {"id": "trace_dummy", "object": "trace"}
@router.get("/v1/traces/{trace_id}")

async def get_trace(trace_id: str):
    """Dummy get trace endpoint."""
    return {"id": trace_id, "object": "trace"}
@router.post("/v1/projects/{project_id}/traces")

async def create_project_trace(project_id: str):
    """Dummy project trace endpoint."""
    return {"id": "trace_dummy", "object": "trace"}