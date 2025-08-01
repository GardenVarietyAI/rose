import time
import uuid
from typing import List, Optional

from sqlmodel import select

from rose_server.database import get_session
from rose_server.entities.messages import Message
from rose_server.schemas.chat import ChatMessage


async def get_response(response_id: str) -> Optional[Message]:
    async with get_session(read_only=True) as session:
        return await session.get(Message, response_id)


async def get_chain_ids() -> List[str]:
    """Load all messages in a conversation chain."""
    async with get_session(read_only=True) as session:
        query = (
            select(Message.response_chain_id)
            .where(Message.response_chain_id.is_not(None))
            .distinct()
            .order_by(Message.created_at)
        )
        result = await session.execute(query)
        chain_ids: List[str] = result.scalars().all()
        return chain_ids


async def get_conversation_messages(response_id: str) -> List[Message]:
    """Load all messages in a conversation chain."""
    async with get_session(read_only=True) as session:
        # Get the response message
        response_msg = await session.get(Message, response_id)
        if not response_msg:
            return []

        # Load all messages in the chain
        query = (
            select(Message)
            .where(Message.response_chain_id == response_msg.response_chain_id)
            .order_by(Message.created_at)
        )

        result = await session.execute(query)
        messages: List[Message] = result.scalars().all()
        return messages


async def store_response_messages(
    messages: list[ChatMessage],
    reply_text: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    created_at: int,
    chain_id: Optional[str] = None,
) -> str:
    end_time = time.time()
    # Generate new chain_id if not provided
    if not chain_id:
        chain_id = f"chain_{uuid.uuid4().hex[:16]}"

    async with get_session() as session:
        for msg in messages:
            if msg.role == "user":
                user_message = Message(
                    thread_id=None,
                    role="user",
                    content=[{"type": "text", "text": msg.content}],
                    created_at=created_at,
                    response_chain_id=chain_id,
                    meta={"model": model},
                )
                session.add(user_message)

        assistant_message = Message(
            thread_id=None,
            role="assistant",
            content=[{"type": "text", "text": reply_text}],
            created_at=created_at,
            response_chain_id=chain_id,
            meta={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "response_time_ms": int((end_time - created_at) * 1000),
            },
        )
        session.add(assistant_message)
        await session.commit()
        id: str = assistant_message.id
        return id
