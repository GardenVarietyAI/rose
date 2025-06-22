import time
import uuid
from typing import Optional

from rose_server.database import get_session
from rose_server.entities.threads import Message
from rose_server.schemas.chat import ChatMessage


async def get_response(response_id: str) -> Optional[Message]:
    async with get_session(read_only=True) as session:
        return await session.get(Message, response_id)


async def store_response_messages(
    response_id: str,
    messages: list[ChatMessage],
    reply_text: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    created_at: int,
) -> None:
    end_time = time.time()

    async with get_session() as session:
        for msg in messages:
            if msg.role == "user":
                user_message = Message(
                    id=f"msg_{uuid.uuid4().hex[:8]}",
                    thread_id=None,
                    role="user",
                    content=[{"type": "text", "text": msg.content}],
                    created_at=created_at,
                    meta={"model": model},
                )
                session.add(user_message)

        assistant_message = Message(
            id=response_id,
            thread_id=None,
            role="assistant",
            content=[{"type": "text", "text": reply_text}],
            created_at=created_at,
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
