import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from rose_web import __version__
from rose_web.chatkit_server import RoseChatKitServer
from rose_web.database import create_all_tables, create_session_maker
from rose_web.router import router
from rose_web.settings import get_settings
from rose_web.sqlite_store import SQLiteStore

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["AGENTS_TRACING_ENABLED"] = "false"


logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rose_web")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    app.state.templates = templates

    settings = get_settings()
    data_dir = Path(settings.data_dir)
    db_path = data_dir / "chatkit.db"
    data_dir.mkdir(parents=True, exist_ok=True)

    engine, session_maker = create_session_maker(db_path)
    await create_all_tables(engine)

    data_store = SQLiteStore(session_maker)
    app.state.chatkit_server = RoseChatKitServer(data_store)

    yield

    await engine.dispose()
    logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ROSE Web",
        description="ROSE Web Application",
        version=__version__,
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "frontend" / "dist"), name="static")
    app.include_router(router)

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
