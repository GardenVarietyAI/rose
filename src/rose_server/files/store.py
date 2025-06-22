import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import BinaryIO, List, Optional

import aiofiles
import aiofiles.os
from openai.types import FileDeleted, FileObject, FilePurpose
from pydantic import ValidationError

from rose_server.schemas.files import TrainingData

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

    # Create purpose directory
    purpose_dir = BASE_PATH / purpose
    await aiofiles.os.makedirs(purpose_dir, exist_ok=True)

    # Save file as purpose/filename
    file_path = purpose_dir / filename
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

    # Store id->path mapping
    mapping_file = BASE_PATH / "mappings.json"
    mappings = {}
    if await aiofiles.os.path.exists(mapping_file):
        async with aiofiles.open(mapping_file, "r") as f:
            mappings = json.loads(await f.read())

    mappings[file_id] = f"{purpose}/{filename}"
    async with aiofiles.open(mapping_file, "w") as f:
        await f.write(json.dumps(mappings, indent=2))

    logger.info(f"Created file {file_id} at {purpose}/{filename}")
    return file_obj


async def get_file(file_id: str) -> Optional[FileObject]:
    mapping_file = BASE_PATH / "mappings.json"
    if not await aiofiles.os.path.exists(mapping_file):
        return None

    async with aiofiles.open(mapping_file, "r") as f:
        mappings = json.loads(await f.read())

    if file_id not in mappings:
        return None

    path_parts = mappings[file_id].split("/", 1)
    purpose = path_parts[0]
    filename = path_parts[1] if len(path_parts) > 1 else "unknown"
    file_path = BASE_PATH / mappings[file_id]

    if not await aiofiles.os.path.exists(file_path):
        return None

    stat = await aiofiles.os.stat(file_path)
    return FileObject(
        id=file_id,
        object="file",
        bytes=stat.st_size,
        created_at=int(stat.st_ctime),
        filename=filename,
        purpose=purpose,
        status="processed",
    )


async def get_file_content(file_id: str) -> Optional[bytes]:
    mapping_file = BASE_PATH / "mappings.json"
    if not await aiofiles.os.path.exists(mapping_file):
        return None

    async with aiofiles.open(mapping_file, "r") as f:
        mappings = json.loads(await f.read())

    if file_id not in mappings:
        return None

    file_path = BASE_PATH / mappings[file_id]
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
    mapping_file = BASE_PATH / "mappings.json"
    if not await aiofiles.os.path.exists(mapping_file):
        return []

    async with aiofiles.open(mapping_file, "r") as f:
        mappings = json.loads(await f.read())

    files = []
    for file_id, path in mappings.items():
        file_obj = await get_file(file_id)
        if file_obj and (not purpose or file_obj.purpose == purpose):
            files.append(file_obj)

    files.sort(key=lambda f: f.created_at, reverse=True)

    if after:
        try:
            after_idx = next(i for i, f in enumerate(files) if f.id == after)
            files = files[after_idx + 1 :]
        except StopIteration:
            files = []

    return files[:limit]


async def delete_file(file_id: str) -> Optional[FileDeleted]:
    mapping_file = BASE_PATH / "mappings.json"
    if not await aiofiles.os.path.exists(mapping_file):
        return None

    async with aiofiles.open(mapping_file, "r") as f:
        mappings = json.loads(await f.read())

    if file_id not in mappings:
        return None

    file_path = BASE_PATH / mappings[file_id]
    if await aiofiles.os.path.exists(file_path):
        await aiofiles.os.remove(file_path)

    del mappings[file_id]
    async with aiofiles.open(mapping_file, "w") as f:
        await f.write(json.dumps(mappings, indent=2))

    logger.info(f"Deleted file {file_id}")
    return FileDeleted(id=file_id, object="file", deleted=True)


# Status tracking removed - not needed with simplified approach


async def validate_jsonl(file_id: str) -> tuple[bool, Optional[str]]:
    """Validate that a file contains valid JSONL for fine-tuning."""
    content = await get_file_content(file_id)
    if not content:
        return False, "File not found"
    try:
        lines = content.decode("utf-8").strip().split("\n")
        valid_lines = 0
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                TrainingData(**data)
                valid_lines += 1
            except json.JSONDecodeError as e:
                return False, f"Line {i + 1}: Invalid JSON - {str(e)}"
            except ValidationError as e:
                return False, f"Line {i + 1}: {str(e)}"
        if valid_lines == 0:
            return False, "File contains no valid training examples"
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"
