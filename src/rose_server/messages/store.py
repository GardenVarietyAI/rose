import logging
from typing import Dict, List, Optional

from sqlmodel import select

from ..database import (
    Message,
    get_session,
)

logger = logging.getLogger(__name__)


async def create_message(message: Message) -> Message:
    """Create a new message in a thread."""
    async with get_session() as session:
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message


async def get_messages(thread_id: str, limit: int = 20, order: str = "desc") -> List[Message]:
    """Get messages for a thread."""
    async with get_session(read_only=True) as session:
        statement = select(Message).where(Message.thread_id == thread_id)
        if order == "desc":
            statement = statement.order_by(Message.created_at.desc())
        else:
            statement = statement.order_by(Message.created_at.asc())
        statement = statement.limit(limit)
        messages = (await session.execute(statement)).scalars().all()
        return list(messages)


async def get_message(thread_id: str, message_id: str) -> Optional[Message]:
    """Get a specific message."""
    async with get_session(read_only=True) as session:
        message = await session.get(Message, message_id)
        if message and message.thread_id == thread_id:
            return message
        return None


async def update_message(thread_id: str, message_id: str, metadata: Dict) -> Optional[Message]:
    """Update message metadata."""
    async with get_session() as session:
        message = await session.get(Message, message_id)
        if not message or message.thread_id != thread_id:
            return None
        if metadata:
            message.meta = metadata
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message
