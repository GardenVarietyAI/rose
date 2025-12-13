"""aiosqlite connection with sqlite-vec preloaded."""

import functools
import sqlite3
from typing import Any

import aiosqlite
import llama_cpp
import sqlite_vec


class _VecConnection(sqlite3.Connection):
    """sqlite3.Connection subclass that preloads sqlite-vec on creation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        try:
            self.enable_load_extension(True)
        except AttributeError as e:
            raise RuntimeError("This build of SQLite does not support loadable extensions. ") from e
        try:
            sqlite_vec.load(self)
        except Exception as e:
            raise RuntimeError(f"Failed to load sqlite-vec. Error: {e}") from e


async def connect(path: str, *, pragmas: bool = True, **kwargs: Any) -> aiosqlite.Connection:
    """Open an aiosqlite connection with sqlite-vec"""

    factory = functools.partial(_VecConnection)

    # aiosqlite will pass this factory through to sqlite3.connect()
    db = await aiosqlite.connect(path, factory=factory, **kwargs)

    # Version check
    row = await (await db.execute("SELECT vec_version()")).fetchone()
    if not row or not row[0]:
        raise RuntimeError("sqlite-vec loaded, but vec_version() returned no result")

    # Optional pragma tuning
    if pragmas:
        await db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA busy_timeout=5000;
            PRAGMA foreign_keys=ON;
            PRAGMA temp_store=MEMORY;
            PRAGMA page_size=8192;
            PRAGMA mmap_size=268435456;
        """)
        await db.commit()

    return db


async def create_all_tables(db: aiosqlite.Connection, embedding_dim: int) -> None:
    """Initialize vector embedding tables."""
    await db.execute(f"""
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message_id TEXT PRIMARY KEY,
            embedding FLOAT[{embedding_dim}]
        )
    """)
    await db.commit()


async def get_missing_embeddings(db: aiosqlite.Connection, message_ids: list[str]) -> list[str]:
    """Check which message IDs don't have embeddings yet."""
    if not message_ids:
        return []

    placeholders = ",".join("?" * len(message_ids))
    query = f"SELECT message_id FROM message_embeddings WHERE message_id IN ({placeholders})"

    cursor = await db.execute(query, message_ids)
    existing = {row[0] for row in await cursor.fetchall()}

    return [msg_id for msg_id in message_ids if msg_id not in existing]


async def store_embedding(db: aiosqlite.Connection, message_id: str, embedding: list[float]) -> None:
    """Store a message embedding in the vector database."""
    await db.execute(
        "INSERT OR REPLACE INTO message_embeddings (message_id, embedding) VALUES (?, ?)",
        (message_id, embedding),
    )
    await db.commit()


def generate_embedding(embed_model: llama_cpp.Llama, text: str) -> list[float]:
    """Generate embedding for text using the embedding model."""
    result = embed_model.create_embedding(text)
    return result["data"][0]["embedding"]  # type: ignore[index,return-value]
