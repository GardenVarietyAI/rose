from pydantic import ValidationError
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from rose_server.models.message_types import LensMessage
from rose_server.models.messages import Message

LENS_OBJECT = "lens"


async def _get_latest_revision_ids(session: AsyncSession) -> list[int]:
    result = await session.execute(
        select(func.max(col(Message.id)))
        .where(
            col(Message.object) == LENS_OBJECT,
            col(Message.deleted_at).is_(None),
        )
        .group_by(col(Message.root_message_id))
    )
    return [row[0] for row in result.all()]


async def resolve_lens_uuid_to_root(session: AsyncSession, lens_uuid: str) -> str | None:
    result = await session.execute(
        select(Message)
        .where(
            col(Message.uuid) == lens_uuid,
            col(Message.object) == LENS_OBJECT,
        )
        .limit(1)
    )
    message = result.scalar_one_or_none()
    if message is None:
        return None
    if message.meta and message.meta.get("root_message_id"):
        return str(message.meta["root_message_id"])
    return message.uuid


async def get_latest_lens_revision(session: AsyncSession, root_message_id: str) -> Message | None:
    result = await session.execute(
        select(Message)
        .where(
            col(Message.object) == LENS_OBJECT,
            col(Message.root_message_id) == root_message_id,
            col(Message.deleted_at).is_(None),
        )
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def validate_at_name_unique(session: AsyncSession, at_name: str, exclude_root_id: str | None = None) -> bool:
    latest_ids = await _get_latest_revision_ids(session)
    if not latest_ids:
        return True

    result = await session.execute(
        select(Message)
        .where(
            col(Message.id).in_(latest_ids),
            col(Message.at_name) == at_name,
        )
    )
    messages = list(result.scalars().all())
    for msg in messages:
        root_id = msg.meta.get("root_message_id") if msg.meta else msg.uuid
        if exclude_root_id and root_id == exclude_root_id:
            continue
        return False

    return True


async def list_lenses_messages(session: AsyncSession) -> list[Message]:
    latest_ids = await _get_latest_revision_ids(session)
    if not latest_ids:
        return []

    result = await session.execute(
        select(Message)
        .where(col(Message.id).in_(latest_ids))
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
    )
    return list(result.scalars().all())


async def get_lens_message(session: AsyncSession, lens_id: str) -> Message | None:
    root_id = await resolve_lens_uuid_to_root(session, lens_id)
    if root_id is None:
        return None
    return await get_latest_lens_revision(session, root_id)


async def list_lens_picker_options(session: AsyncSession) -> list[tuple[str, str, str]]:
    lenses = await list_lenses_messages(session)
    options: list[tuple[str, str, str]] = []
    for lens_message in lenses:
        try:
            lens = LensMessage(message=lens_message)
        except ValidationError as e:
            raise ValueError("Invalid lens message") from e
        options.append((lens.lens_id, lens.at_name, lens.label))
    return options
