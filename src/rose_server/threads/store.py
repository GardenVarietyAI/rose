"""SQLModel-based storage for threads and messages with ChromaDB integration."""

import logging
import uuid
from typing import Dict, List, Optional

from sqlalchemy import delete, func
from sqlmodel import select

from ..database import (
    Message as MessageDB,
    Thread as ThreadDB,
    current_timestamp,
    get_session,
)
from ..schemas.threads import Thread

logger = logging.getLogger(__name__)


def _to_openai_thread(db_thread: ThreadDB) -> Thread:
    """Convert database thread to OpenAI-compatible Thread model."""
    metadata = db_thread.meta if hasattr(db_thread, "meta") and db_thread.meta else {}
    return Thread(
        id=db_thread.id,
        object="thread",
        created_at=db_thread.created_at,
        metadata=metadata,
        tool_resources=db_thread.tool_resources
        if hasattr(db_thread, "tool_resources") and db_thread.tool_resources
        else None,
    )


async def create_thread(metadata: Dict = None) -> Thread:
    """Create a new thread."""
    thread_id = f"thread_{uuid.uuid4().hex}"
    async with get_session() as session:
        db_thread = ThreadDB(
            id=thread_id,
            created_at=current_timestamp(),
            meta=metadata or {},
            tool_resources={},
        )
        session.add(db_thread)
        await session.commit()
        await session.refresh(db_thread)
        return _to_openai_thread(db_thread)


async def get_thread(thread_id: str) -> Optional[Thread]:
    """Get a thread by ID."""

    async with get_session(read_only=True) as session:
        db_thread = await session.get(ThreadDB, thread_id)
        if db_thread:
            return _to_openai_thread(db_thread)
        return None


async def update_thread(thread_id: str, metadata: Dict) -> Optional[Thread]:
    """Update thread metadata."""
    async with get_session() as session:
        db_thread = await session.get(ThreadDB, thread_id)
        if not db_thread:
            return None
        if metadata:
            db_thread.meta = metadata
        session.add(db_thread)
        await session.commit()
        await session.refresh(db_thread)
        return _to_openai_thread(db_thread)


async def delete_thread(thread_id: str) -> bool:
    """Delete a thread and all its messages."""
    async with get_session() as session:
        db_thread = await session.get(ThreadDB, thread_id)
        if not db_thread:
            return False
        message_count_result = await session.execute(
            select(func.count(MessageDB.id)).where(MessageDB.thread_id == thread_id)
        )
        message_count = message_count_result.scalar()
        await session.execute(delete(MessageDB).where(MessageDB.thread_id == thread_id))
        await session.delete(db_thread)
        await session.commit()
        logger.info(f"Deleted thread: {thread_id} and {message_count} messages")
        return True


async def list_threads(limit: int = 20, order: str = "desc") -> List[Thread]:
    """List all threads."""
    async with get_session(read_only=True) as session:
        statement = select(ThreadDB)
        if order == "desc":
            statement = statement.order_by(ThreadDB.created_at.desc())
        else:
            statement = statement.order_by(ThreadDB.created_at.asc())
        statement = statement.limit(limit)
        db_threads = (await session.execute(statement)).scalars().all()
        return [_to_openai_thread(t) for t in db_threads]
