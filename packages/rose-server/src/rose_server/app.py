import logging
import os
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

import httpx
import nltk
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from symspellpy import SymSpell
from yoyo import get_backend, read_migrations

from rose_server.database import create_session_maker, get_session
from rose_server.router import router
from rose_server.views.pages.opensearch import render_opensearch_xml

logger = logging.getLogger("rose_server")

DB_NAME = "rose_20251218.db"
DB_MIGRATIONS = "db/migrations"


LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1"))
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", os.getenv("OPENAI_API_KEY", ""))
NLTK_DATA = os.getenv("NLTK_DATA", "./vendor/nltk_data")
SPELLCHECK_DICTIONARY = "frequency_dictionary_en_82_765.txt"
STATIC_PATH = files("rose_server").joinpath("static")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    try:
        app.state.engine, app.state.db_session_maker = create_session_maker(DB_NAME)
        app.state.get_db_session = lambda read_only=False: get_session(app.state.db_session_maker, read_only)
        backend = get_backend(f"sqlite:///{DB_NAME}")
        migrations = read_migrations(DB_MIGRATIONS)
        with backend.lock():
            backend.apply_migrations(backend.to_apply(migrations))
    except Exception as e:
        logger.warning(f"Failed to create database session: {e}")
        raise

    headers = {}
    if LLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {LLAMA_API_KEY}"

    app.state.llama_client = httpx.AsyncClient(base_url=LLAMA_BASE_URL, headers=headers, timeout=60.0)
    logger.info(f"LLAMA_BASE_URL: {LLAMA_BASE_URL}")

    if NLTK_DATA not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA)

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary(str(files("symspellpy").joinpath(SPELLCHECK_DICTIONARY)), term_index=0, count_index=1)
    app.state.spell_checker = sym_spell

    yield

    await app.state.llama_client.aclose()
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

    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")
    app.include_router(router)

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/opensearch.xml")
    async def opensearch_descriptor(request: Request) -> Any:
        base_url = str(request.base_url).rstrip("/")
        return Response(
            content=render_opensearch_xml(base_url=base_url),
            media_type="application/opensearchdescription+xml",
        )

    return app


app = create_app()
