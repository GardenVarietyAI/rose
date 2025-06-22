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

    # Create base uploads directory
    await aiofiles.os.makedirs(BASE_PATH, exist_ok=True)

    # Save file directly with file_id as filename
    file_path = BASE_PATH / file_id
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Save metadata
    metadata = {
        "id": file_id,
        "filename": filename,
        "purpose": purpose,
        "created_at": int(time.time()),
        "bytes": file_size,
    }
    meta_path = BASE_PATH / f"{file_id}.json"
    async with aiofiles.open(meta_path, "w") as f:
        await f.write(json.dumps(metadata))

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
    mapping_file = BASE_PATH / "mappings.json"
    if not await aiofiles.os.path.exists(mapping_file):
        return None

    async with aiofiles.open(mapping_file, "r") as f:
        mappings = json.loads(await f.read())

    if file_id not in mappings:
        return None

    file_path = BASE_PATH / mappings[file_id]

    # Extract purpose from file metadata or default to "fine-tune"
    # Since we're simplifying, we'll need to determine purpose another way
    purpose = "fine-tune"  # Default purpose
    filename = mappings[file_id]

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
        # If filtering by purpose, check the path prefix
        if purpose and not path.startswith(f"{purpose}/"):
            continue

        file_obj = await get_file(file_id)
        if file_obj:
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
