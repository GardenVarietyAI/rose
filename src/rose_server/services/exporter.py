import json
import random
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from rose_server.models.messages import Message
from rose_server.schemas.exporter import ChatMessage, Conversation


def _first_by_thread(messages: list[Message]) -> dict[str, Message]:
    selected: dict[str, Message] = {}
    for msg in messages:
        if msg.thread_id is None or not msg.thread_id.strip():
            continue
        if msg.content is None or not str(msg.content).strip():
            continue
        if msg.thread_id in selected:
            continue
        selected[msg.thread_id] = msg
    return selected


async def query_thread_ids(
    session: AsyncSession,
    lens_id: str | None,
    accepted_only: bool,
) -> list[str]:
    query = select(Message.thread_id).where(
        col(Message.role) == "assistant",
        col(Message.deleted_at).is_(None),
    ).distinct()

    if lens_id:
        query = query.where(col(Message.lens_id) == lens_id)

    if accepted_only:
        query = query.where(col(Message.accepted_at).is_not(None))

    result = await session.execute(query)
    return list(result.scalars().all())


async def query_user_messages(
    session: AsyncSession,
    thread_ids: list[str],
) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(
            col(Message.thread_id).in_(thread_ids),
            col(Message.role) == "user",
            col(Message.deleted_at).is_(None),
        )
        .order_by(col(Message.thread_id), col(Message.created_at).desc(), col(Message.id).desc())
    )
    return list(result.scalars().all())


async def query_assistant_messages(
    session: AsyncSession,
    thread_ids: list[str],
    lens_id: str | None,
    accepted_only: bool,
) -> list[Message]:
    query = select(Message).where(
        col(Message.thread_id).in_(thread_ids),
        col(Message.role) == "assistant",
        col(Message.deleted_at).is_(None),
    )

    if lens_id:
        query = query.where(col(Message.lens_id) == lens_id)

    if accepted_only:
        query = query.where(col(Message.accepted_at).is_not(None))

    query = query.order_by(
        col(Message.thread_id),
        col(Message.accepted_at).desc().nulls_last(),
        col(Message.created_at).desc(),
        col(Message.id).desc(),
    )

    result = await session.execute(query)
    return list(result.scalars().all())


async def build_conversations(
    user_messages: list[Message],
    assistant_messages: list[Message],
    lens_id: str | None,
    session: AsyncSession,
) -> list[Conversation]:
    lens_content = None
    if lens_id:
        result = await session.execute(
            select(Message).where(
                col(Message.uuid) == lens_id,
                col(Message.role) == "system",
                col(Message.deleted_at).is_(None),
            )
        )
        lens = result.scalar_one_or_none()
        if lens and lens.content:
            lens_content = lens.content

    user_by_thread = _first_by_thread(user_messages)
    assistant_by_thread = _first_by_thread(assistant_messages)

    conversations = []
    for thread_id, user_msg in user_by_thread.items():
        if thread_id not in assistant_by_thread:
            continue

        assistant_msg = assistant_by_thread[thread_id]
        if not user_msg.content or not assistant_msg.content:
            continue

        messages = []
        if lens_content:
            messages.append(ChatMessage(role="system", content=lens_content))

        messages.append(ChatMessage(role="user", content=user_msg.content))
        messages.append(ChatMessage(role="assistant", content=assistant_msg.content))

        conversations.append(Conversation(messages=messages))

    return conversations


def split_dataset(
    conversations: list[Conversation],
    ratio: float,
) -> tuple[list[Conversation], list[Conversation]]:
    shuffled = conversations.copy()
    random.Random(42).shuffle(shuffled)
    split_point = int(len(shuffled) * ratio)
    return shuffled[:split_point], shuffled[split_point:]


def write_jsonl(conversations: list[Conversation], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        for conv in conversations:
            f.write(f"{json.dumps(conv.model_dump())}\n")
