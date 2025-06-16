"""Unified file storage for all OpenAI file endpoints."""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional

import aiofiles
import aiofiles.os
from openai.types import FileDeleted, FileObject, FilePurpose
from pydantic import ValidationError

from rose_server.schemas.files import TrainingData

logger = logging.getLogger(__name__)


class FileStore:
    """Manages file storage for all purposes (fine-tuning, vector stores, assistants, etc.)."""

    def __init__(self, base_path: str = "data/uploads"):
        self.base_path = Path(base_path)
        self._files: Dict[str, FileObject] = {}

    async def _load_existing_files(self):
        """Load metadata for existing files from disk."""
        await aiofiles.os.makedirs(self.base_path, exist_ok=True)
        metadata_path = self.base_path / "metadata.json"
        if await aiofiles.os.path.exists(metadata_path):
            try:
                async with aiofiles.open(metadata_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content)
                    for file_id, file_data in data.items():
                        self._files[file_id] = FileObject(**file_data)
                logger.info(f"Loaded {len(self._files)} existing files")
            except Exception as e:
                logger.error(f"Failed to load file metadata: {e}")

    async def _save_metadata(self):
        """Save file metadata to disk."""
        metadata_path = self.base_path / "metadata.json"
        try:
            data = {file_id: file_obj.model_dump() for file_id, file_obj in self._files.items()}
            content = json.dumps(data, indent=2)
            async with aiofiles.open(metadata_path, "w") as f:
                await f.write(content)
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")

    def _generate_file_id(self, purpose: str) -> str:
        """Generate a file ID with purpose prefix."""
        prefix_map = {
            "fine-tune": "fine",
            "fine-tune-results": "fine",
            "assistants": "asst",
            "assistants_output": "asst",
            "batch": "batch",
            "batch_output": "batch",
            "vision": "vis",
        }
        prefix = prefix_map.get(purpose, "file")
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    async def create_file(self, file: BinaryIO, purpose: FilePurpose, filename: Optional[str] = None) -> FileObject:
        """Create a new file."""
        file_id = self._generate_file_id(purpose)
        content = await asyncio.to_thread(file.read)
        file_size = len(content)
        if not filename:
            filename = f"{file_id}.txt"
        file_path = self.base_path / file_id
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        file_obj = FileObject(
            id=file_id,
            object="file",
            bytes=file_size,
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
            status="uploaded",
            status_details=None,
            expires_at=None,
        )
        self._files[file_id] = file_obj
        await self._save_metadata()
        logger.info(f"Created file {file_id} with purpose {purpose}")
        return file_obj

    def get_file(self, file_id: str) -> Optional[FileObject]:
        """Get file metadata."""
        return self._files.get(file_id)

    async def get_file_content(self, file_id: str) -> Optional[bytes]:
        """Get file content."""
        file_path = self.base_path / file_id
        if file_path.exists():
            async with aiofiles.open(file_path, "rb") as f:
                return await f.read()
        return None

    def list_files(
        self,
        purpose: Optional[str] = None,
        limit: int = 20,
        after: Optional[str] = None,
    ) -> List[FileObject]:
        """List files with optional filtering."""
        files = list(self._files.values())
        if purpose:
            files = [f for f in files if f.purpose == purpose]
        files.sort(key=lambda f: f.created_at, reverse=True)
        if after:
            try:
                after_idx = next(i for i, f in enumerate(files) if f.id == after)
                files = files[after_idx + 1 :]
            except StopIteration:
                files = []
        return files[:limit]

    async def delete_file(self, file_id: str) -> Optional[FileDeleted]:
        """Delete a file."""
        if file_id not in self._files:
            return None
        file_path = self.base_path / file_id
        if file_path.exists():
            await aiofiles.os.remove(file_path)
        del self._files[file_id]
        await self._save_metadata()
        logger.info(f"Deleted file {file_id}")
        return FileDeleted(id=file_id, object="file", deleted=True)

    async def update_file_status(self, file_id: str, status: str, status_details: Optional[str] = None):
        """Update the status of a file."""
        if file_id not in self._files:
            return False
        old_file = self._files[file_id]
        updated_file = FileObject(
            id=old_file.id,
            object=old_file.object,
            bytes=old_file.bytes,
            created_at=old_file.created_at,
            filename=old_file.filename,
            purpose=old_file.purpose,
            status=status,
            status_details=status_details,
            expires_at=old_file.expires_at,
        )
        self._files[file_id] = updated_file
        await self._save_metadata()
        logger.info(f"Updated file {file_id} status to {status}")
        return True

    async def validate_jsonl(self, file_id: str) -> tuple[bool, Optional[str]]:
        """Validate that a file contains valid JSONL for fine-tuning."""
        content = await self.get_file_content(file_id)
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
