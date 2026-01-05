from sqlalchemy import Subquery, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from rose_server.models.messages import Message

FACTSHEET_OBJECT = "factsheet"


def _latest_revision_subquery() -> Subquery:
    return (
        select(
            col(Message.root_message_id).label("root_id"),
            func.max(col(Message.id)).label("max_id"),
        )
        .where(
            col(Message.object) == FACTSHEET_OBJECT,
            col(Message.deleted_at).is_(None),
        )
        .group_by(col(Message.root_message_id))
        .subquery()
    )


async def resolve_factsheet_uuid_to_root(session: AsyncSession, factsheet_uuid: str) -> str | None:
    result = await session.execute(
        select(Message)
        .where(
            col(Message.uuid) == factsheet_uuid,
            col(Message.object) == FACTSHEET_OBJECT,
        )
        .limit(1)
    )
    message = result.scalar_one_or_none()
    if message is None:
        return None
    if message.meta and message.meta.get("root_message_id"):
        return str(message.meta["root_message_id"])
    return message.uuid


async def get_latest_factsheet_revision(session: AsyncSession, root_message_id: str) -> Message | None:
    result = await session.execute(
        select(Message)
        .where(
            col(Message.object) == FACTSHEET_OBJECT,
            col(Message.root_message_id) == root_message_id,
            col(Message.deleted_at).is_(None),
        )
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _is_factsheet_hashtag_unique(
    session: AsyncSession,
    *,
    hashtag: str,
    exclude_root_id: str | None = None,
) -> bool:
    subquery = _latest_revision_subquery()
    result = await session.execute(
        select(Message)
        .join(
            subquery,
            (col(Message.root_message_id) == subquery.c.root_id) & (col(Message.id) == subquery.c.max_id),
        )
        .where(
            col(Message.object) == FACTSHEET_OBJECT,
            col(Message.tag) == hashtag,
            col(Message.deleted_at).is_(None),
        )
    )
    messages = list(result.scalars().all())
    for msg in messages:
        root_id = msg.meta.get("root_message_id") if msg.meta else msg.uuid
        if exclude_root_id and root_id == exclude_root_id:
            continue
        return False
    return True


async def validate_hashtag_unique(
    session: AsyncSession,
    *,
    hashtag: str,
    exclude_root_id: str | None = None,
) -> bool:
    return await _is_factsheet_hashtag_unique(session, hashtag=hashtag, exclude_root_id=exclude_root_id)


async def list_factsheets_messages(session: AsyncSession) -> list[Message]:
    subquery = _latest_revision_subquery()
    result = await session.execute(
        select(Message)
        .join(
            subquery,
            (col(Message.root_message_id) == subquery.c.root_id) & (col(Message.id) == subquery.c.max_id),
        )
        .where(
            col(Message.object) == FACTSHEET_OBJECT,
            col(Message.deleted_at).is_(None),
        )
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
    )
    return list(result.scalars().all())


async def get_factsheet_message(session: AsyncSession, factsheet_id: str) -> Message | None:
    root_id = await resolve_factsheet_uuid_to_root(session, factsheet_id)
    if root_id is None:
        return None
    return await get_latest_factsheet_revision(session, root_id)
