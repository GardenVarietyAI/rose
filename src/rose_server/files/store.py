import asyncio
import logging
import time
import uuid
from typing import BinaryIO, List, Optional

from openai.types import FileDeleted, FileObject, FilePurpose
from sqlalchemy import select

from rose_server.database import get_session
from rose_server.entities.files import UploadedFile
from rose_server.fs import (
    delete_file as delete_file_from_disk,
    read_file,
    save_file,
)

logger = logging.getLogger(__name__)


async def create_file(file: BinaryIO, purpose: FilePurpose, filename: Optional[str] = None) -> FileObject:
    file_id = f"file-{uuid.uuid4().hex[:6]}"
    content = await asyncio.to_thread(file.read)
    file_size = len(content)
    if not filename:
        filename = f"{file_id}.txt"

    # Save file to disk
    await save_file(file_id, content)

    # Save to database
    uploaded_file = UploadedFile(
        id=file_id,
        object="file",
        bytes=file_size,
        created_at=int(time.time()),
        filename=filename,
        purpose=purpose,
        status="processed",
        storage_path=file_id,
    )

    async with get_session() as session:
        session.add(uploaded_file)
        await session.commit()

        file_obj = FileObject(
            id=file_id,
            object="file",
            bytes=file_size,
            created_at=uploaded_file.created_at,
            filename=filename,
            purpose=purpose,  # type: ignore
            status="processed",
        )

        logger.info(f"Created file {file_id} with filename {filename}")
        return file_obj


async def get_file(file_id: str) -> Optional[FileObject]:
    async with get_session(read_only=True) as session:
        result = await session.execute(select(UploadedFile).where(UploadedFile.id == file_id))
        uploaded_file = result.scalar_one_or_none()

        if not uploaded_file:
            return None

        return FileObject(
            id=uploaded_file.id,
            object=uploaded_file.object,
            bytes=uploaded_file.bytes,
            created_at=uploaded_file.created_at,
            filename=uploaded_file.filename,
            purpose=uploaded_file.purpose,  # type: ignore
            status=uploaded_file.status or "processed",
        )


async def get_file_content(file_id: str) -> Optional[bytes]:
    # First check if file exists in database
    file_obj = await get_file(file_id)
    if not file_obj:
        return None

    return await read_file(file_id)


async def list_files(
    purpose: Optional[str] = None,
    limit: int = 20,
    after: Optional[str] = None,
) -> List[FileObject]:
    """List files with optional filtering."""
    async with get_session(read_only=True) as session:
        query = select(UploadedFile)

        if purpose:
            query = query.where(UploadedFile.purpose == purpose)

        query = query.order_by(UploadedFile.created_at.desc())

        if after:
            # Get the created_at time of the 'after' file
            after_result = await session.execute(select(UploadedFile.created_at).where(UploadedFile.id == after))
            after_created_at = after_result.scalar_one_or_none()
            if after_created_at:
                query = query.where(UploadedFile.created_at < after_created_at)

        query = query.limit(limit)
        result = await session.execute(query)
        files = result.scalars().all()

        return [
            FileObject(
                id=f.id,
                object=f.object,
                bytes=f.bytes,
                created_at=f.created_at,
                filename=f.filename,
                purpose=f.purpose,  # type: ignore
                status=f.status or "processed",
            )
            for f in files
        ]


async def delete_file(file_id: str) -> Optional[FileDeleted]:
    async with get_session() as session:
        result = await session.execute(select(UploadedFile).where(UploadedFile.id == file_id))
        uploaded_file = result.scalar_one_or_none()

        if not uploaded_file:
            return None

        # Delete from filesystem
        await delete_file_from_disk(file_id)

        # Delete from database
        await session.delete(uploaded_file)
        await session.commit()

        logger.info(f"Deleted file {file_id}")
        return FileDeleted(id=file_id, object="file", deleted=True)
