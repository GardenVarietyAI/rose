"""FastAPI application factory and configuration."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

# Disable PostHog analytics
os.environ["POSTHOG_DISABLED"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rose_server import __version__
from rose_server.config.settings import settings
from rose_server.database import check_database_setup, create_all_tables
from rose_server.models.registry import ModelRegistry
from rose_server.router import router

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
        settings.fine_tuning_checkpoint_dir,
    ]
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir}")

    if not await check_database_setup():
        raise RuntimeError("Database not found. Please run 'dbmate up' and try again.")

    await create_all_tables()

    app.state.model_registry = ModelRegistry()
    logger.info("Model registry initialized")

    yield

    logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ROSE",
        description="A service for generating responses using different LLM modes",
        version=__version__,
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
