"""FastAPI application factory and configuration."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from rose_core.config.service import ServiceConfig, get_full_config
from rose_server.database import create_all_tables
from rose_server.embeddings.manager import EmbeddingManager
from rose_server.files.store import FileStore
from rose_server.fine_tuning.store import FineTuningStore
from rose_server.language_models.registry import ModelRegistry
from rose_server.queues.store import JobStore
from rose_server.router import router
from rose_server.services import Services, get_vector_store_store
from rose_server.threads.store import ThreadStore
from rose_server.tokens import TokenizerService
from rose_server.vector import ChromaDBManager
from rose_server.vector_stores.store import VectorStoreStore

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
logging.basicConfig(
    level=getattr(logging, ServiceConfig.LOG_LEVEL),
    format=ServiceConfig.LOG_FORMAT,
)
logger = logging.getLogger("rose_server")


async def log_request_body(request: Request, call_next):
    """Log all HTTP requests with method, path, headers, and POST body."""
    user_agent = request.headers.get("user-agent", "unknown")
    logger.info(f"{request.method} {request.url.path} - User-Agent: {user_agent}")
    if request.method == "POST":
        body = await request.body()
        if body:
            try:
                body_str = body.decode("utf-8")
                try:
                    body_json = json.loads(body_str)
                    logger.info(f"POST Body: {json.dumps(body_json, indent=2)}")
                except json.JSONDecodeError:
                    logger.info(f"POST Body (raw): {body_str}")
            except Exception as e:
                logger.error(f"Error logging request: {str(e)}")
        request._body = body
        request.body = lambda: body
    return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    directories = [
        ServiceConfig.DATA_DIR,
        ServiceConfig.MODEL_OFFLOAD_DIR,
        ServiceConfig.CHROMA_PERSIST_DIR,
        ServiceConfig.FINE_TUNING_CHECKPOINT_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")
    await create_all_tables()
    logger.info("SQLite database initialized with WAL mode")
    chromadb_manager = ChromaDBManager(
        host=ServiceConfig.CHROMA_HOST,
        port=ServiceConfig.CHROMA_PORT,
        persist_dir=ServiceConfig.CHROMA_PERSIST_DIR,
    )
    Services.register("chromadb_manager", chromadb_manager)
    file_store = FileStore()
    await file_store._load_existing_files()
    Services.register("file_store", file_store)
    Services.register("fine_tuning_store", FineTuningStore())
    Services.register("vector_store_store", VectorStoreStore())
    job_store = JobStore()
    await job_store.initialize()
    Services.register("job_store", job_store)
    model_registry = ModelRegistry()
    await model_registry.initialize()
    Services.register("model_registry", model_registry)
    embedding_manager = EmbeddingManager()
    Services.register("embedding_manager", embedding_manager)
    Services.register("tokenizer_service", TokenizerService(model_registry))
    logger.info(f"Services initialized: {Services.list_services()}")
    vector_store_store = get_vector_store_store()
    vector_store_store.initialize_with_client(chromadb_manager.client)
    logger.info("Vector Store Manager initialized with shared ChromaDB client")
    thread_store = ThreadStore()
    thread_store.set_chroma_client(chromadb_manager.client)
    logger.info("Thread Store initialized with shared ChromaDB client")
    yield
    await Services.shutdown()
    logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ROSE",
        description="A service for generating responses using different LLM modes",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.middleware("http")(log_request_body)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/config")
    async def get_config() -> dict:
        """Get the current service configuration."""
        return get_full_config()

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Simple health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()
