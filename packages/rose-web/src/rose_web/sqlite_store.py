from typing import Any
from uuid import uuid4

from chatkit.store import Store
from chatkit.types import Attachment, FileAttachment, ImageAttachment, Page, ThreadItem, ThreadMetadata
from pydantic import AnyUrl, TypeAdapter
from sqlalchemy import asc, delete, desc
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import col, select

from rose_web.entities import AttachmentEntity, Thread, ThreadItemEntity

thread_item_adapter: TypeAdapter[ThreadItem] = TypeAdapter(ThreadItem)
attachment_adapter: TypeAdapter[Attachment] = TypeAdapter(Attachment)


class SQLiteStore(Store[Any]):
    def __init__(self, session_maker: async_sessionmaker[AsyncSession]):
        self.session_maker = session_maker

    def generate_thread_id(self, context: Any) -> str:
        return str(uuid4())

    def generate_item_id(self, item_type: str, thread_id: ThreadMetadata, context: Any) -> str:
        return str(uuid4())

    def generate_attachment_id(self, mime_type: str, context: Any) -> str:
        return str(uuid4())

    async def create_attachment(self, input: Any, context: Any) -> Attachment:
        if not input.get("filename"):
            raise ValueError("Attachment filename is required")
        if not input.get("mime_type"):
            raise ValueError("Attachment mime_type is required")
        if not input.get("data"):
            raise ValueError("Attachment data is required")

        attachment_id = self.generate_attachment_id(input["mime_type"], context)
        mime_type = input["mime_type"]
        filename = input["filename"]

        attachment: Attachment
        if mime_type.startswith("image/"):
            data_url = f"data:{mime_type};base64,{input['data']}"
            attachment = ImageAttachment(
                id=attachment_id, name=filename, mime_type=mime_type, type="image", preview_url=AnyUrl(data_url)
            )
        else:
            attachment = FileAttachment(id=attachment_id, name=filename, mime_type=mime_type, type="file")

        attachment_with_data = {
            "id": attachment_id,
            "type": attachment.type,
            "mime_type": mime_type,
            "filename": filename,
            "data": input["data"],
        }
        if hasattr(attachment, "preview_url"):
            attachment_with_data["preview_url"] = str(attachment.preview_url)

        async with self.session_maker() as session:
            entity = AttachmentEntity(id=attachment_id, data=str(attachment_with_data))
            session.add(entity)
            await session.commit()

        return attachment

    async def load_thread(self, thread_id: str, context: Any) -> ThreadMetadata:
        async with self.session_maker() as session:
            result = await session.execute(select(Thread).where(col(Thread.id) == thread_id))
            thread = result.scalar_one_or_none()

            if not thread:
                raise KeyError(f"Thread {thread_id} not found")

            return ThreadMetadata.model_validate_json(thread.data)

    async def save_thread(self, thread: ThreadMetadata, context: Any) -> None:
        async with self.session_maker() as session:
            result = await session.execute(select(Thread).where(col(Thread.id) == thread.id))
            existing = result.scalar_one_or_none()

            if existing:
                existing.data = thread.model_dump_json()
            else:
                entity = Thread(id=thread.id, data=thread.model_dump_json())
                session.add(entity)

            await session.commit()

    async def delete_thread(self, thread_id: str, context: Any) -> None:
        async with self.session_maker() as session:
            await session.execute(delete(ThreadItemEntity).where(col(ThreadItemEntity.thread_id) == thread_id))
            await session.execute(delete(Thread).where(col(Thread.id) == thread_id))
            await session.commit()

    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: Any,
    ) -> Page[ThreadMetadata]:
        async with self.session_maker() as session:
            query = select(Thread)

            if after:
                if order == "asc":
                    query = query.where(col(Thread.id) > after)
                else:
                    query = query.where(col(Thread.id) < after)

            query = query.order_by(asc(Thread.id) if order == "asc" else desc(Thread.id)).limit(limit + 1)

            result = await session.execute(query)
            rows = result.scalars().all()

            has_more = len(rows) > limit
            threads = [ThreadMetadata.model_validate_json(row.data) for row in rows[:limit]]

            return Page(data=threads, has_more=has_more, after=rows[limit - 1].id if has_more and threads else None)

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: Any,
    ) -> Page[ThreadItem]:
        async with self.session_maker() as session:
            query = select(ThreadItemEntity).where(col(ThreadItemEntity.thread_id) == thread_id)

            if after:
                if order == "asc":
                    query = query.where(col(ThreadItemEntity.id) > after)
                else:
                    query = query.where(col(ThreadItemEntity.id) < after)

            query = query.order_by(asc(ThreadItemEntity.id) if order == "asc" else desc(ThreadItemEntity.id)).limit(
                limit + 1
            )

            result = await session.execute(query)
            rows = result.scalars().all()

            has_more = len(rows) > limit
            items = [thread_item_adapter.validate_json(row.data) for row in rows[:limit]]

            return Page(data=items, has_more=has_more, after=rows[limit - 1].id if has_more and items else None)

    async def add_thread_item(self, thread_id: str, item: ThreadItem, context: Any) -> None:
        await self.save_item(thread_id, item, context)

    async def save_item(self, thread_id: str, item: ThreadItem, context: Any) -> None:
        async with self.session_maker() as session:
            result = await session.execute(
                select(ThreadItemEntity).where(
                    col(ThreadItemEntity.id) == item.id, col(ThreadItemEntity.thread_id) == thread_id
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.data = item.model_dump_json()
            else:
                entity = ThreadItemEntity(id=item.id, thread_id=thread_id, data=item.model_dump_json())
                session.add(entity)

            await session.commit()

    async def load_item(self, thread_id: str, item_id: str, context: Any) -> ThreadItem:
        async with self.session_maker() as session:
            result = await session.execute(
                select(ThreadItemEntity).where(
                    col(ThreadItemEntity.id) == item_id, col(ThreadItemEntity.thread_id) == thread_id
                )
            )
            item = result.scalar_one_or_none()

            if not item:
                raise KeyError(f"Item {item_id} not found in thread {thread_id}")

            return thread_item_adapter.validate_json(item.data)

    async def delete_thread_item(self, thread_id: str, item_id: str, context: Any) -> None:
        async with self.session_maker() as session:
            await session.execute(
                delete(ThreadItemEntity).where(
                    col(ThreadItemEntity.id) == item_id, col(ThreadItemEntity.thread_id) == thread_id
                )
            )
            await session.commit()

    async def save_attachment(self, attachment: Attachment, context: Any) -> None:
        async with self.session_maker() as session:
            result = await session.execute(select(AttachmentEntity).where(col(AttachmentEntity.id) == attachment.id))
            existing = result.scalar_one_or_none()

            if existing:
                existing.data = attachment.model_dump_json()
            else:
                entity = AttachmentEntity(id=attachment.id, data=attachment.model_dump_json())
                session.add(entity)

            await session.commit()

    async def load_attachment(self, attachment_id: str, context: Any) -> Attachment:
        async with self.session_maker() as session:
            result = await session.execute(select(AttachmentEntity).where(col(AttachmentEntity.id) == attachment_id))
            attachment = result.scalar_one_or_none()

            if not attachment:
                raise KeyError(f"Attachment {attachment_id} not found")

            return attachment_adapter.validate_json(attachment.data)

    async def delete_attachment(self, attachment_id: str, context: Any) -> None:
        async with self.session_maker() as session:
            await session.execute(delete(AttachmentEntity).where(col(AttachmentEntity.id) == attachment_id))
            await session.commit()
