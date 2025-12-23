import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from rose_server.models import Message, SearchEvent

logger = logging.getLogger(__name__)


def create_session_maker(db_name: str) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_name}",
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

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_conn: Any, _connection_record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine, async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def get_session(
    session_maker: async_sessionmaker[AsyncSession],
    read_only: bool = False,
) -> AsyncGenerator[AsyncSession, None]:
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


__all__ = [
    "create_session_maker",
    "get_session",
    "Message",
    "SearchEvent",
]
