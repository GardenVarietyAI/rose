"""SQLModel-based storage for assistants with clean OpenAI compatibility."""

import logging
import uuid
from typing import List, Optional

from sqlmodel import select

from rose_server.database import current_timestamp, get_session
from rose_server.entities.assistants import Assistant
from rose_server.schemas.assistants import AssistantCreateRequest, AssistantUpdateRequest

logger = logging.getLogger(__name__)


async def create_assistant(request: AssistantCreateRequest) -> Assistant:
    """Create a new assistant."""
    assistant_id = f"asst_{uuid.uuid4().hex}"
    async with get_session() as session:
        # Convert tools to dicts for JSON storage
        tools = [tool.model_dump() for tool in request.tools] if request.tools else []

        db_assistant = Assistant(
            id=assistant_id,
            created_at=current_timestamp(),
            name=request.name,
            description=request.description,
            model=request.model,
            instructions=request.instructions,
            tools=tools,
            tool_resources=request.tool_resources or {},
            meta=request.metadata or {},
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 1.0,
            response_format=request.response_format,
        )
        session.add(db_assistant)
        await session.commit()
        await session.refresh(db_assistant)

        return db_assistant


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


async def update_assistant(assistant_id: str, request: AssistantUpdateRequest) -> Optional[Assistant]:
    """Update an assistant."""
    async with get_session() as session:
        db_assistant = await session.get(Assistant, assistant_id)
        if not db_assistant:
            return None

        # Update only provided fields
        updates = request.model_dump(exclude_unset=True)
        for field, value in updates.items():
            if field == "metadata":
                # Map metadata to meta for database
                db_assistant.meta = value
            elif field == "tools":
                # Convert tool objects to dicts for JSON storage
                db_assistant.tools = [tool.model_dump() for tool in value]
            else:
                # Direct assignment for other fields
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
