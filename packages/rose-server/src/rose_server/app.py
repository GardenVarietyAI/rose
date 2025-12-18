import glob
import logging
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import nltk
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
from symspellpy import SymSpell

from rose_server.database import check_database_setup, create_session_maker, get_session
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
    templates_dir = Path("./packages/rose-server/src/rose_server/templates")
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    app.state.engine, app.state.db_session_maker = create_session_maker()
    logger.info("Database session maker initialized")

    app.state.get_db_session = lambda read_only=False: get_session(app.state.db_session_maker, read_only)

    if not await check_database_setup(app.state.engine):
        logger.error("Database not initialized. Run: yoyo apply --database 'sqlite:///rose_20251211.db' db/migrations")
        raise RuntimeError("Database not initialized")

    try:
        app.state.chat_model = load_chat_model(MODELS["chat"])
    except Exception as e:
        app.state.chat_model = None
        logger.warning(f"Failed to load chat model: {e}")

    try:
        nltk.download("stopwords", quiet=True)
        logger.info("NLTK stopwords loaded")
    except Exception as e:
        logger.warning(f"Failed to download NLTK stopwords: {e}")

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = data_dir / "frequency_dictionary_en_82_765.txt"

    if not dictionary_path.exists():
        logger.info(f"Downloading spell check dictionary to {dictionary_path}")
        dictionary_url = (
            "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"
        )
        try:
            urllib.request.urlretrieve(dictionary_url, dictionary_path)
            logger.info("Dictionary downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download dictionary: {e}")

    if dictionary_path.exists():
        sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
        logger.info("Spell check dictionary loaded")
        app.state.spell_checker = sym_spell
    else:
        logger.warning("Spell check dictionary not available")
        app.state.spell_checker = None

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

    @app.get("/opensearch.xml")
    async def opensearch_descriptor(request: Request) -> Any:
        base_url = str(request.base_url).rstrip("/")
        return request.app.state.templates.TemplateResponse(
            "opensearch.xml",
            {"request": request, "base_url": base_url},
            media_type="application/opensearchdescription+xml",
        ).body.decode()

    return app


app = create_app()
