import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, TypeVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from rose_server.connect import _VecConnection
from rose_server.entities.files import UploadedFile
from rose_server.entities.fine_tuning import FineTuningEvent, FineTuningJob
from rose_server.entities.messages import Message
from rose_server.entities.models import LanguageModel
from rose_server.entities.vector_stores import Document, VectorStore

logger = logging.getLogger(__name__)


def get_db_path(data_dir: str) -> Path:
    """Get database path for given data directory."""
    return Path(data_dir) / "rose.db"


def create_session_maker(data_dir: str) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Create the async engine and a session maker for it."""
    db_path = get_db_path(data_dir)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=3600,
        connect_args={"check_same_thread": False, "timeout": 20, "factory": _VecConnection},
    )
    return engine, async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


T = TypeVar("T")


@asynccontextmanager
async def get_session(
    session_maker: async_sessionmaker[AsyncSession], read_only: bool = False
) -> AsyncGenerator[AsyncSession, None]:
    """Get async database session context manager."""
    async with session_maker() as session:
        try:
            yield session
            if not read_only:
                await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_all_tables(engine: AsyncEngine, embedding_dimensions: int) -> None:
    """Create all database tables using the provided engine."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

        await conn.execute(
            text(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec0 USING vec0(
                document_id TEXT PRIMARY KEY,
                embedding float[{embedding_dimensions}]
            )
        """)
        )
        logger.info("vec0 table created successfully")


def current_timestamp() -> int:
    """Get current Unix timestamp."""
    return int(time.time())


async def check_database_setup(engine: AsyncEngine) -> bool:
    """Check if database connection works using the provided engine."""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False


__all__ = [
    "create_session_maker",
    "get_session",
    "create_all_tables",
    "current_timestamp",
    "check_database_setup",
    "get_db_path",
    "UploadedFile",
    "FineTuningJob",
    "FineTuningEvent",
    "LanguageModel",
    "Message",
    "VectorStore",
    "Document",
]
