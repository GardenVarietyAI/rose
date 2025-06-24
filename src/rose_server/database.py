"""Database setup and model re-exports for backward compatibility.
Database setup stays here. SQLModel classes have been moved to entity files.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from rose_core.config.service import DATA_DIR

from .entities.assistants import Assistant
from .entities.files import UploadedFile
from .entities.fine_tuning import FineTuningEvent, FineTuningJob
from .entities.jobs import Job
from .entities.language_models import LanguageModel
from .entities.messages import Message
from .entities.threads import MessageMetadata, Thread

DB_PATH = Path(DATA_DIR) / "rose_server.db"
engine = create_async_engine(
    f"sqlite+aiosqlite:///{DB_PATH}",
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    connect_args={
        "check_same_thread": False,
        "timeout": 20,
    },
)
T = TypeVar("T")
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_session(read_only: bool = False) -> AsyncGenerator[AsyncSession, None]:
    """Get async database session context manager."""
    async with async_session_factory() as session:
        try:
            yield session
            if not read_only:
                await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def run_in_session(operation: Callable[[AsyncSession], Any], read_only: bool = False) -> Any:
    """
    Execute a database operation with proper session management.

    Args:
        operation: Async function that takes a session and performs database operations
        read_only: Whether this is a read-only operation (no commit needed)
    Returns:
        The result of the operation function
    """
    async with get_session(read_only=read_only) as session:
        return await operation(session)


async def create_all_tables() -> None:
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


def current_timestamp() -> int:
    """Get current Unix timestamp."""
    return int(time.time())


__all__ = [
    "engine",
    "get_session",
    "run_in_session",
    "create_all_tables",
    "current_timestamp",
    "UploadedFile",
    "FineTuningJob",
    "FineTuningEvent",
    "Job",
    "LanguageModel",
    "Assistant",
    "Thread",
    "Message",
    "MessageMetadata",
]
