import asyncio
import logging
import time
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
    content = await asyncio.to_thread(file.read)
    file_size = len(content)

    # Save to database first to get the auto-generated ID
    uploaded_file = UploadedFile(
        object="file",
        bytes=file_size,
        created_at=int(time.time()),
        filename=filename or "file.txt",
        purpose=purpose,
        status="processed",
        storage_path="",  # Will be updated after we have the ID
    )

    async with get_session() as session:
        session.add(uploaded_file)
        await session.commit()
        await session.refresh(uploaded_file)

        # Now we have the ID, update filename if not provided
        if not filename:
            uploaded_file.filename = f"{uploaded_file.id}.txt"

        # Save file to disk using the generated ID
        await save_file(uploaded_file.id, content)

        # Update storage path
        uploaded_file.storage_path = uploaded_file.id
        await session.commit()

        file_obj = FileObject(
            id=uploaded_file.id,
            object="file",
            bytes=file_size,
            created_at=uploaded_file.created_at,
            filename=uploaded_file.filename,
            purpose=purpose,  # type: ignore
            status="processed",
        )

        logger.info(f"Created file {uploaded_file.id} with filename {uploaded_file.filename}")
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
