"""aiosqlite connection with sqlite-vec preloaded."""

import functools
from typing import Any

import aiosqlite
import sqlite_vec

try:
    import sqlite3

    _ = sqlite3.connect(":memory:").enable_load_extension  # probe
except AttributeError:
    import pysqlite3 as sqlite3  # type: ignore


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
    """
    Open an aiosqlite connection with sqlite-vec preloaded.
    Usage: db = await connect('rose.db')
    """
    factory = functools.partial(_VecConnection)
    # aiosqlite will pass this factory through to sqlite3.connect()
    db = await aiosqlite.connect(path, factory=factory, **kwargs)

    # Fail-fast version check
    row = await (await db.execute("SELECT vec_version()")).fetchone()
    if not row or not row[0]:
        raise RuntimeError("sqlite-vec loaded, but vec_version() returned no result")

    # Optional pragma tuning
    if pragmas:
        await db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA foreign_keys=ON;
        """)
        await db.commit()

    return db
