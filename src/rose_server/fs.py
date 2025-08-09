"""File system operations for file storage."""

import logging
from pathlib import Path
from typing import Optional

import aiofiles
import aiofiles.os

from rose_server.config.settings import settings

logger = logging.getLogger(__name__)

UPLOADS_PATH = Path(settings.data_dir) / "uploads"
RESULTS_PATH = Path(settings.data_dir) / "results"


async def save_file(file_id: str, content: bytes) -> None:
    await aiofiles.os.makedirs(UPLOADS_PATH, exist_ok=True)
    file_path = UPLOADS_PATH / file_id
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)


async def save_results_file(file_id: str, content: bytes) -> None:
    await aiofiles.os.makedirs(UPLOADS_PATH, exist_ok=True)
    file_path = RESULTS_PATH / file_id
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)


async def read_file(file_id: str) -> Optional[bytes]:
    file_path = UPLOADS_PATH / file_id
    file_path_exists = await aiofiles.os.path.exists(file_path)
    if file_path_exists:
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()
    return None


async def delete_file(file_id: str) -> bool:
    file_path = UPLOADS_PATH / file_id
    if await aiofiles.os.path.exists(file_path):
        await aiofiles.os.remove(file_path)
        return True
    return False
