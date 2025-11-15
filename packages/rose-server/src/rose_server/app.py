import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

# Disable PostHog analytics
os.environ["POSTHOG_DISABLED"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tokenizers import Tokenizer

from rose_server import __version__
from rose_server._inference import InferenceServer
from rose_server.database import check_database_setup, create_all_tables, create_session_maker, get_session
from rose_server.router import router
from rose_server.settings import Settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"


logger = logging.getLogger("rose_server")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    settings = Settings()
    app.state.settings = settings

    app.state.engine, app.state.db_session_maker = create_session_maker(settings.data_dir)
    logger.info(f"Database session maker initialized with data_dir: {settings.data_dir}")

    app.state.get_db_session = lambda read_only=False: get_session(app.state.db_session_maker, read_only)

    directories = [
        settings.data_dir,
        settings.model_offload_dir,
    ]
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir}")

    if not await check_database_setup(app.state.engine):
        logger.info("Creating database...")
        db_path = Path(settings.data_dir) / "rose.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch(exist_ok=True)

    await create_all_tables(app.state.engine)

    app.state.inference_server = InferenceServer("auto")
    logger.info("Inference server initialized")

    tokenizer_path = Path(settings.data_dir) / "models/Qwen--Qwen3-0.6B/tokenizer.json"
    if tokenizer_path.exists():
        app.state.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        logger.info(f"Qwen3 tokenizer loaded from {tokenizer_path}")
    else:
        app.state.tokenizer = None
        logger.warning(f"Qwen3 tokenizer not found at {tokenizer_path}")

    yield

    logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ROSE",
        description="Run your own LLM server",
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
        return {"status": "ok"}

    return app


app = create_app()
