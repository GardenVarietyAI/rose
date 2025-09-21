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
from rose_server._inference import EmbeddingModel, InferenceServer, RerankerModel
from rose_server.database import check_database_setup, create_all_tables
from rose_server.router import router
from rose_server.settings import Settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"


logger = logging.getLogger("rose_server")


def get_tokenizer(models_dir: str, embedding_model_name: str) -> Tokenizer:
    tokenizer_path = Path(models_dir) / embedding_model_name / "tokenizer.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    return Tokenizer.from_file(str(tokenizer_path))


def get_embedding_model(
    models_dir: str,
    embedding_model_name: str,
    embedding_model_quantization: str,
    embedding_device: str,
    embedding_dimensions: int,
) -> EmbeddingModel:
    model_path = (Path(models_dir) / embedding_model_name).resolve()

    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files found in {model_path}")

    # Find the specified quantization level
    gguf_file = next((f for f in gguf_files if embedding_model_quantization in f.name), None)

    if not gguf_file:
        available_quants = [f.name for f in gguf_files]
        raise FileNotFoundError(
            f"Quantization {embedding_model_quantization} not found in {model_path}. " f"Available: {available_quants}"
        )

    tokenizer_file = model_path / "tokenizer.json"

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

    model = EmbeddingModel(
        str(gguf_file.resolve()),
        str(tokenizer_file.resolve()),
        embedding_device,
        embedding_dimensions,
    )
    logger.info(
        f"Loaded embeddings: {gguf_file.name} "
        f"on device: {embedding_device}, "
        f"output_dims: {embedding_dimensions}"
    )
    return model


def get_reranker_model(models_dir: str) -> RerankerModel:
    model_path = Path(models_dir) / "QuantFactory--Qwen3-Reranker-0.6B-GGUF"

    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files found in {model_path}")

    gguf_file = next((f for f in gguf_files if "Q8_0" in f.name), gguf_files[0])
    tokenizer_file = model_path / "tokenizer.json"

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

    model = RerankerModel(str(gguf_file), str(tokenizer_file), "auto")
    logger.info(f"Loaded reranker: {gguf_file.name}")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    # Initialize settings
    settings = Settings()
    app.state.settings = settings

    directories = [
        settings.data_dir,
        settings.model_offload_dir,
        settings.fine_tuning_checkpoint_dir,
    ]
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir}")

    if not await check_database_setup():
        logger.info("Creating database...")
        db_path = Path(settings.data_dir) / "rose.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch(exist_ok=True)

    await create_all_tables()

    app.state.inference_server = InferenceServer("auto")
    logger.info("Inference server initialized")

    tokenizer_path = Path(settings.data_dir) / "models/Qwen--Qwen3-0.6B/tokenizer.json"
    if tokenizer_path.exists():
        app.state.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        logger.info(f"Qwen3 tokenizer loaded from {tokenizer_path}")
    else:
        app.state.tokenizer = None
        logger.warning(f"Qwen3 tokenizer not found at {tokenizer_path}")

    try:
        app.state.embedding_model = get_embedding_model(
            models_dir=settings.models_dir,
            embedding_model_name=settings.embedding_model_name,
            embedding_model_quantization=settings.embedding_model_quantization,
            embedding_device=settings.embedding_device,
            embedding_dimensions=settings.embedding_dimensions,
        )
        app.state.embedding_tokenizer = get_tokenizer(
            models_dir=settings.models_dir,
            embedding_model_name=settings.embedding_model_name,
        )
        logger.info("Embeddings and tokenizer loaded")
    except Exception as e:
        app.state.embedding_model = None
        app.state.embedding_tokenizer = None
        logger.warning(f"Failed to load embedding model: {e}")

    try:
        app.state.reranker_model = get_reranker_model(models_dir=settings.models_dir)
        logger.info("Reranker loaded")
    except Exception as e:
        app.state.reranker_model = None
        logger.warning(f"Failed to load reranker: {e}")

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
