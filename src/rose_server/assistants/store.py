"""SQLModel-based storage for assistants with clean OpenAI compatibility."""

import logging
from typing import Any, List, Optional

from sqlmodel import select

from rose_server.database import get_session
from rose_server.entities.assistants import Assistant

logger = logging.getLogger(__name__)


async def create_assistant(assistant: Assistant) -> Assistant:
    """Create a new assistant."""
    async with get_session() as session:
        session.add(assistant)
        await session.commit()
        await session.refresh(assistant)
        return assistant


async def get_assistant(assistant_id: str) -> Optional[Assistant]:
    """Get an assistant by ID."""
    async with get_session(read_only=True) as session:
        return await session.get(Assistant, assistant_id)


async def list_assistants(limit: int = 20, order: str = "desc") -> List[Assistant]:
    """List assistants."""
    async with get_session(read_only=True) as session:
        statement = select(Assistant)
        if order == "desc":
            statement = statement.order_by(Assistant.created_at.desc())
        else:
            statement = statement.order_by(Assistant.created_at.asc())
        statement = statement.limit(limit)
        result = await session.execute(statement)
        return list(result.scalars().all())


async def update_assistant(assistant_id: str, updates: dict[str, Any]) -> Optional[Assistant]:
    """Update an assistant."""
    async with get_session() as session:
        db_assistant = await session.get(Assistant, assistant_id)
        if not db_assistant:
            return None

        # Update fields
        for field, value in updates.items():
            setattr(db_assistant, field, value)

        session.add(db_assistant)
        await session.commit()
        await session.refresh(db_assistant)
        return db_assistant


async def delete_assistant(assistant_id: str) -> bool:
    """Delete an assistant."""
    async with get_session() as session:
        db_assistant = await session.get(Assistant, assistant_id)
        if db_assistant:
            await session.delete(db_assistant)
            await session.commit()
            logger.info(f"Deleted assistant: {assistant_id}")
            return True
        return False
