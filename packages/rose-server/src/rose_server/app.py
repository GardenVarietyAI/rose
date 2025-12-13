import glob
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama

from rose_server.database import check_database_setup, create_all_tables, create_session_maker, get_session
from rose_server.llms import MODELS, ModelConfig
from rose_server.router import router

logger = logging.getLogger("rose_server")


def load_model(model_path: str, n_gpu_layers: int, n_ctx: int, embedding: bool = False) -> Llama:
    matches = glob.glob(model_path)
    if not matches:
        raise FileNotFoundError(f"Model not found at {model_path}")

    resolved_path = matches[0]
    logger.info(f"Loading model from {resolved_path}")
    return Llama(
        model_path=resolved_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        embedding=embedding,
        verbose=False,
    )


def load_chat_model(config: ModelConfig) -> Llama:
    return load_model(config["path"], config["n_gpu_layers"], config["n_ctx"])


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    app.state.engine, app.state.db_session_maker = create_session_maker()
    logger.info("Database session maker initialized")

    app.state.get_db_session = lambda read_only=False: get_session(app.state.db_session_maker, read_only)

    if not await check_database_setup(app.state.engine):
        logger.info("Creating database...")

    await create_all_tables(app.state.engine)

    if "chat" not in MODELS:
        raise RuntimeError("Chat model configuration missing from MODELS")

    app.state.chat_model = load_chat_model(MODELS["chat"])

    yield

    logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ROSE",
        description="Run your own LLM server",
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
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
