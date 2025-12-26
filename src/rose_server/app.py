import logging
import traceback
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
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

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException) -> JSONResponse:
        logger.error(f"HTTP {e.status_code}: {e.detail} - {request.method} {request.url}")
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
        )

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, e: ValidationError) -> JSONResponse:
        logger.error(f"Validation error on {request.method} {request.url}: {e}")
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation error", "errors": e.errors()},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, e: ValueError) -> JSONResponse:
        logger.error(f"ValueError on {request.method} {request.url}: {e}")
        return JSONResponse(
            status_code=400,
            content={"detail": str(e)},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, e: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception on {request.method} {request.url}:")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(e)},
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
