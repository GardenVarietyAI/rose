"""FastAPI application factory and configuration."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

# Disable PostHog analytics
os.environ["POSTHOG_DISABLED"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rose_core.config.settings import settings
from rose_server.database import create_all_tables
from rose_server.middleware.auth import AuthMiddleware
from rose_server.models.registry import ModelRegistry
from rose_server.router import router
from rose_server.vector import ChromaDBManager

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format,
)
logger = logging.getLogger("rose_server")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Manage application lifecycle."""
    directories = [
        settings.data_dir,
        settings.model_offload_dir,
        settings.chroma_persist_dir,
        settings.fine_tuning_checkpoint_dir,
    ]
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir}")

    await create_all_tables()

    logger.info("SQLite database initialized with WAL mode")

    app.state.vector = ChromaDBManager(
        host=settings.chroma_host,
        port=settings.chroma_port,
        persist_dir=settings.chroma_persist_dir,
    )

    app.state.model_registry = ModelRegistry()
    logger.info("Model registry initialized")

    yield

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

    # Add auth middleware if enabled
    if settings.auth_enabled:
        app.add_middleware(AuthMiddleware)
    else:
        logger.warning("⚠️  API authentication is DISABLED. Set ROSE_SERVER_AUTH_ENABLED=true to enable.")

    app.include_router(router)

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Simple health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()
