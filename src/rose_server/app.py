import logging
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from symspellpy import SymSpell

from alembic import command
from alembic.config import Config
from rose_server.database import create_session_maker, get_session
from rose_server.router import router
from rose_server.settings import Settings
from rose_server.views.pages.opensearch import render_opensearch_xml

logger = logging.getLogger("rose_server")

DB_NAME = "rose_20251223.db"

SPELLCHECK_PATH = files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
STATIC_PATH = files("rose_server").joinpath("static")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    try:
        app.state.engine, app.state.db_session_maker = create_session_maker(DB_NAME)
        app.state.get_db_session = lambda read_only=False: get_session(app.state.db_session_maker, read_only)

        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
    except Exception as e:
        logger.warning(f"Failed to create database session: {e}")
        raise

    settings = Settings()
    app.state.settings = settings

    headers: dict[str, str] = {}
    if settings.llama_api_key:
        headers["Authorization"] = f"Bearer {settings.llama_api_key}"

    app.state.llama_client = httpx.AsyncClient(base_url=settings.llama_base_url, headers=headers, timeout=300.0)
    logger.info(f"LLAMA_BASE_URL: {settings.llama_base_url}")

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary(str(SPELLCHECK_PATH), term_index=0, count_index=1)
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
