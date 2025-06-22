import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import BinaryIO, List, Optional

import aiofiles
import aiofiles.os
from openai.types import FileDeleted, FileObject, FilePurpose

logger = logging.getLogger(__name__)

BASE_PATH = Path("data/uploads")


def _generate_file_id() -> str:
    return f"file-{uuid.uuid4().hex[:6]}"


async def create_file(file: BinaryIO, purpose: FilePurpose, filename: Optional[str] = None) -> FileObject:
    file_id = _generate_file_id()
    content = await asyncio.to_thread(file.read)
    file_size = len(content)
    if not filename:
        filename = f"{file_id}.txt"

    # Create base uploads directory
    await aiofiles.os.makedirs(BASE_PATH, exist_ok=True)

    # Save file directly with file_id as filename
    file_path = BASE_PATH / file_id
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    file_obj = FileObject(
        id=file_id,
        object="file",
        bytes=file_size,
        created_at=int(time.time()),
        filename=filename,
        purpose=purpose,
        status="processed",
    )

    logger.info(f"Created file {file_id} with filename {filename}")
    return file_obj


async def get_file(file_id: str) -> Optional[FileObject]:
    # TODO: Implement database lookup
    return None


async def get_file_content(file_id: str) -> Optional[bytes]:
    file_path = BASE_PATH / file_id
    if await aiofiles.os.path.exists(file_path):
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()
    return None


async def list_files(
    purpose: Optional[str] = None,
    limit: int = 20,
    after: Optional[str] = None,
) -> List[FileObject]:
    """List files with optional filtering."""
    # TODO: Implement database lookup
    return []


async def delete_file(file_id: str) -> Optional[FileDeleted]:
    file_path = BASE_PATH / file_id
    if await aiofiles.os.path.exists(file_path):
        await aiofiles.os.remove(file_path)
        logger.info(f"Deleted file {file_id}")
        return FileDeleted(id=file_id, object="file", deleted=True)
    return None
