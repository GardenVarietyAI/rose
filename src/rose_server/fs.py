"""File system operations for file storage."""

import logging
from pathlib import Path
from typing import Optional

import aiofiles
import aiofiles.os

from rose_core.config.settings import settings

logger = logging.getLogger(__name__)

BASE_PATH = Path(settings.data_dir) / "uploads"


def get_file_path(file_id: str) -> Path:
    return BASE_PATH / file_id


async def ensure_upload_dir() -> None:
    await aiofiles.os.makedirs(BASE_PATH, exist_ok=True)


async def check_file_path(file_path: Path) -> bool:
    return await aiofiles.os.path.exists(file_path)


async def save_file(file_id: str, content: bytes) -> None:
    await ensure_upload_dir()
    file_path = get_file_path(file_id)
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)


async def read_file(file_id: str) -> Optional[bytes]:
    file_path = get_file_path(file_id)
    if await check_file_path(file_path):
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()
    return None


async def delete_file(file_id: str) -> bool:
    file_path = get_file_path(file_id)
    if await aiofiles.os.path.exists(file_path):
        await aiofiles.os.remove(file_path)
        return True
    return False
