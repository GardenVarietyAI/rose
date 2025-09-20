"""FastAPI application factory and configuration."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from starlette.templating import Jinja2Templates

from rose_web import __version__
from rose_web.router import router
from rose_web.settings import settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["ANONYMIZED_TELEMETRY"] = "false"


logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rose_web")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    app.state.openai = AsyncOpenAI(base_url=settings.openai_api_url, api_key=settings.opneai_api_key)
    app.state.templates = templates

    yield

    logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ROSE Web",
        description="ROSE Web Application",
        version=__version__,
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    app.include_router(router)

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
