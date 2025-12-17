import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from rose_server.models import Message, SearchEvent

logger = logging.getLogger(__name__)


def create_session_maker(
    db_url: str | None = None,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    if db_url is None:
        db_url = "sqlite+aiosqlite:///rose_20251211.db"

    engine = create_async_engine(
        db_url,
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


async def create_all_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

        await conn.execute(
            text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                role UNINDEXED,
                model UNINDEXED,
                thread_id UNINDEXED,
                created_at UNINDEXED,
                content='messages',
                content_rowid='id',
                tokenize = 'porter unicode61 remove_diacritics 2'
            )
        """)
        )

        await conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content, role, model, thread_id, created_at)
                VALUES (new.id, new.content, new.role, new.model, new.thread_id, new.created_at);
            END
        """)
        )

        await conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, role, model, thread_id, created_at)
                VALUES('delete', old.id, old.content, old.role, old.model, old.thread_id, old.created_at);
            END
        """)
        )

        await conn.execute(
            text("""
            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, role, model, thread_id, created_at)
                VALUES('delete', old.id, old.content, old.role, old.model, old.thread_id, old.created_at);
                INSERT INTO messages_fts(rowid, content, role, model, thread_id, created_at)
                VALUES (new.id, new.content, new.role, new.model, new.thread_id, new.created_at);
            END
        """)
        )


async def check_database_setup(engine: AsyncEngine) -> bool:
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
    "check_database_setup",
    "Message",
    "SearchEvent",
]
