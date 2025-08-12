import logging
import time
from typing import List, Optional

from openai.types import FilePurpose
from sqlalchemy import delete, select

from rose_server.database import get_session
from rose_server.entities.files import UploadedFile

logger = logging.getLogger(__name__)


async def create_file(file_size: int, purpose: FilePurpose, filename: str, content: bytes) -> UploadedFile:
    """Create file with content stored as BLOB in database."""
    uploaded_file = UploadedFile(
        object="file",
        bytes=file_size,
        created_at=int(time.time()),
        filename=filename,
        purpose=purpose,
        status="processed",
        storage_path="BLOB",
        content=content,
    )

    async with get_session() as session:
        session.add(uploaded_file)
        await session.commit()
        await session.refresh(uploaded_file)
        logger.info(f"Created file {uploaded_file.id} with BLOB content, filename {uploaded_file.filename}")
        return uploaded_file


async def get_file_content(file_id: str) -> Optional[bytes]:
    """Get file content from BLOB storage."""
    async with get_session(read_only=True) as session:
        result = await session.execute(select(UploadedFile.content).where(UploadedFile.id == file_id))
        return result.scalar_one_or_none()


async def get_file(file_id: str) -> Optional[UploadedFile]:
    async with get_session(read_only=True) as session:
        result = await session.execute(select(UploadedFile).where(UploadedFile.id == file_id))
        return result.scalar_one_or_none()


async def list_files(
    purpose: Optional[str] = None,
    limit: int = 20,
    after: Optional[str] = None,
) -> List[UploadedFile]:
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
        files: List[UploadedFile] = result.scalars().all()
        return files


async def delete_file(file_id: str) -> bool:
    async with get_session() as session:
        delete_stmt = delete(UploadedFile).where(UploadedFile.id == file_id)
        result = await session.execute(delete_stmt)

        # Check if any rows were actually deleted
        if result.rowcount == 0:
            return False

        await session.commit()
        logger.info(f"Deleted file {file_id}")
        return True
