"""FastAPI application factory and configuration."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

# Disable PostHog analytics
os.environ["POSTHOG_DISABLED"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rose_core.config.service import (
    CHROMA_HOST,
    CHROMA_PERSIST_DIR,
    CHROMA_PORT,
    DATA_DIR,
    FINE_TUNING_CHECKPOINT_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    MODEL_OFFLOAD_DIR,
)
from rose_server.database import create_all_tables
from rose_server.language_models.registry import ModelRegistry
from rose_server.router import router
from rose_server.services import Services
from rose_server.tokens import TokenizerService
from rose_server.vector import ChromaDBManager

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
)
logger = logging.getLogger("rose_server")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Manage application lifecycle."""
    directories = [
        DATA_DIR,
        MODEL_OFFLOAD_DIR,
        CHROMA_PERSIST_DIR,
        FINE_TUNING_CHECKPOINT_DIR,
    ]
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir}")

    await create_all_tables()

    logger.info("SQLite database initialized with WAL mode")

    app.state.vector = ChromaDBManager(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        persist_dir=CHROMA_PERSIST_DIR,
    )

    model_registry = ModelRegistry()
    await model_registry.initialize()
    Services.register("model_registry", model_registry)
    Services.register("tokenizer_service", TokenizerService(model_registry))
    logger.info(f"Services initialized: {Services.list_services()}")

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Simple health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()
