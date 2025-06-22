import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from openai.types.beta.threads import Text, TextContentBlock
from sqlmodel import select

from ..database import (
    Message as MessageDB,
    Thread as ThreadDB,
    current_timestamp,
    get_session,
)
from ..schemas.threads import ThreadMessage

logger = logging.getLogger(__name__)


def _to_openai_message(db_message: MessageDB) -> ThreadMessage:
    """Convert database message to OpenAI-compatible ThreadMessage model."""
    content = db_message.content
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = [TextContentBlock(type="text", text=Text(value=content, annotations=[]))]
    metadata = db_message.meta if hasattr(db_message, "meta") and db_message.meta else {}
    return ThreadMessage(
        id=db_message.id,
        object="thread.message",
        created_at=db_message.created_at,
        thread_id=db_message.thread_id,
        role=db_message.role,
        content=content,
        metadata=metadata,
        assistant_id=db_message.assistant_id,
        run_id=db_message.run_id,
        status="completed",
        completed_at=db_message.created_at,
        incomplete_at=None,
        incomplete_details=None,
        attachments=[],
    )


def _format_message_content(content: List[Dict]) -> List[TextContentBlock]:
    """Format message content into TextContentBlock format."""
    formatted_content = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            formatted_content.append(
                TextContentBlock(type="text", text=Text(value=item.get("text", ""), annotations=[]))
            )
        else:
            formatted_content.append(item)
    return formatted_content


def _serialize_content(formatted_content: List) -> str:
    """Serialize content to JSON string."""
    return json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in formatted_content])


def _extract_metadata_fields(metadata: Optional[Dict]) -> Dict[str, Any]:
    """Extract specific fields from metadata with type conversion."""
    if not metadata:
        return {
            "model_used": None,
            "token_count": None,
            "response_time_ms": None,
            "finish_reason": None,
        }

    fields = {}
    fields["model_used"] = metadata.get("model_used")
    fields["finish_reason"] = metadata.get("finish_reason")

    # Parse numeric fields
    for field in ["token_count", "response_time_ms"]:
        if field in metadata:
            try:
                fields[field] = int(metadata[field])
            except (ValueError, TypeError):
                fields[field] = None
        else:
            fields[field] = None

    return fields


async def create_message(
    thread_id: str,
    role: str,
    content: List[Dict],
    metadata: Dict = None,
    assistant_id: str = None,
    run_id: str = None,
) -> Optional[ThreadMessage]:
    """Create a new message in a thread with optional vector embedding."""
    async with get_session() as session:
        # Verify thread exists
        db_thread = await session.get(ThreadDB, thread_id)
        if not db_thread:
            return None

        # Generate IDs and timestamps
        message_id = f"msg_{uuid.uuid4().hex}"
        created_at = current_timestamp()

        # Format and serialize content
        formatted_content = _format_message_content(content)
        content_json = _serialize_content(formatted_content)

        # Extract metadata fields
        metadata_fields = _extract_metadata_fields(metadata)

        # Create database message
        db_message = MessageDB(
            id=message_id,
            thread_id=thread_id,
            role=role,
            content=content_json,
            created_at=created_at,
            assistant_id=assistant_id,
            run_id=run_id,
            file_ids=[],
            model_used=metadata_fields["model_used"],
            token_count=metadata_fields["token_count"],
            response_time_ms=metadata_fields["response_time_ms"],
            finish_reason=metadata_fields["finish_reason"],
        )

        # Save to database
        session.add(db_message)
        session.add(db_thread)
        await session.commit()
        await session.refresh(db_message)

        # Create response message
        return ThreadMessage(
            id=message_id,
            created_at=created_at,
            thread_id=thread_id,
            role=role,
            content=formatted_content,
            metadata=metadata or {},
            assistant_id=assistant_id,
            run_id=run_id,
            object="thread.message",
            status="completed",
            completed_at=created_at,
            incomplete_at=None,
            incomplete_details=None,
            attachments=[],
        )


async def get_messages(thread_id: str, limit: int = 20, order: str = "desc") -> List[ThreadMessage]:
    """Get messages for a thread."""
    async with get_session(read_only=True) as session:
        statement = select(MessageDB).where(MessageDB.thread_id == thread_id)
        if order == "desc":
            statement = statement.order_by(MessageDB.created_at.desc())
        else:
            statement = statement.order_by(MessageDB.created_at.asc())
        statement = statement.limit(limit)
        db_messages = (await session.execute(statement)).scalars().all()
        return [_to_openai_message(m) for m in db_messages]


async def get_message(thread_id: str, message_id: str) -> Optional[ThreadMessage]:
    """Get a specific message."""
    async with get_session(read_only=True) as session:
        db_message = await session.get(MessageDB, message_id)
        if db_message and db_message.thread_id == thread_id:
            return _to_openai_message(db_message)
        return None


async def update_message(thread_id: str, message_id: str, metadata: Dict) -> Optional[ThreadMessage]:
    """Update message metadata."""
    async with get_session() as session:
        db_message = await session.get(MessageDB, message_id)
        if not db_message or db_message.thread_id != thread_id:
            return None
        if metadata:
            db_message.meta = metadata
        session.add(db_message)
        await session.commit()
        await session.refresh(db_message)
        return _to_openai_message(db_message)
