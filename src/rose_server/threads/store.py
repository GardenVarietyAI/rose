"""SQLModel-based storage for threads and messages with ChromaDB integration."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func
from sqlmodel import select

from rose_server.database import Message, Thread, get_session

logger = logging.getLogger(__name__)


async def create_thread(thread: Thread) -> Thread:
    """Create a new thread."""
    async with get_session() as session:
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        return thread


async def get_thread(thread_id: str) -> Optional[Thread]:
    """Get a thread by ID."""
    async with get_session(read_only=True) as session:
        thread = await session.get(Thread, thread_id)
        return thread


async def update_thread(thread_id: str, metadata: Dict[str, Any]) -> Optional[Thread]:
    """Update thread metadata."""
    async with get_session() as session:
        thread = await session.get(Thread, thread_id)

        if not thread:
            return None

        if metadata is not None:
            thread.meta = metadata

        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        return thread


async def delete_thread(thread_id: str) -> bool:
    """Delete a thread and all its messages."""
    async with get_session() as session:
        thread = await session.get(Thread, thread_id)

        if not thread:
            return False

        message_count_result = await session.execute(
            select(func.count(Message.id)).where(Message.thread_id == thread_id)
        )

        message_count = message_count_result.scalar()
        await session.execute(delete(Message).where(Message.thread_id == thread_id))
        await session.delete(thread)
        await session.commit()

        logger.info(f"Deleted thread: {thread_id} and {message_count} messages")
        return True


async def list_threads(limit: int = 20, order: str = "desc") -> List[Thread]:
    """List all threads."""
    async with get_session(read_only=True) as session:
        statement = select(Thread)

        if order == "desc":
            statement = statement.order_by(Thread.created_at.desc())
        else:
            statement = statement.order_by(Thread.created_at.asc())

        statement = statement.limit(limit)
        threads = (await session.execute(statement)).scalars().all()

        return list(threads)
