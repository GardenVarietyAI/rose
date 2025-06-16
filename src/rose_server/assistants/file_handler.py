"""Handle file attachments for assistants."""
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import aiofiles.os

from rose_server import vector
from rose_server.config import EmbeddingConfig
from rose_server.embeddings.embedding import generate_embeddings

logger = logging.getLogger(__name__)

class AssistantFileHandler:
    """Handle file operations for assistant document retrieval."""

    def __init__(self, base_path: str = "./data/assistant_files", max_file_size_mb: int = 10):
        """Initialize file handler with storage path.

        Args:
            base_path: Base directory for storing assistant files
            max_file_size_mb: Maximum file size in MB (default 10MB)
        """
        self.base_path = Path(base_path)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    async def attach_file_to_assistant(
        self, assistant_id: str, file_content: str, filename: str, file_id: Optional[str] = None
    ) -> str:
        """
        Attach a file to an assistant by chunking and embedding it.
        Args:
            assistant_id: The assistant to attach the file to
            file_content: The text content of the file
            filename: Original filename for metadata
            file_id: Optional file ID (will generate if not provided)
        Returns:
            The file_id of the attached file
        """
        content_size = len(file_content.encode("utf-8"))
        if content_size > self.max_file_size_bytes:
            raise ValueError(f"File too large: {content_size} bytes exceeds limit of {self.max_file_size_bytes} bytes")
        if not file_id:
            file_hash = hashlib.sha256(file_content.encode()).hexdigest()[:8]
            file_id = f"file_{file_hash}"
        file_path = self.base_path / assistant_id / file_id
        await aiofiles.os.makedirs(file_path.parent, exist_ok=True)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(file_content)
        chunks = self._chunk_text(file_content, chunk_size=500, overlap=50)
        collection_name = f"assistant_{assistant_id}_docs"
        vectors = []
        documents = []
        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_chunk_{i}"
            try:
                embedding_response = generate_embeddings(
                    model=EmbeddingConfig.DEFAULT_EMBEDDING_MODEL,
                    input=[chunk],
                )
                if embedding_response.get("data"):
                    embedding = embedding_response["data"][0]["embedding"]
                    vectors.append(embedding)
                    documents.append(chunk)
                    ids.append(chunk_id)
                    metadatas.append(
                        {
                            "file_id": file_id,
                            "filename": filename,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        }
                    )
                else:
                    logger.warning(f"No embedding generated for chunk {i} of file {file_id}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {i} of file {file_id}: {e}")
        if vectors:
            try:
                vector.add_vectors(
                    collection_name=collection_name, vectors=vectors, documents=documents, ids=ids, metadatas=metadatas
                )
                logger.info(f"Attached file {file_id} to assistant {assistant_id} with {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to add vectors to collection {collection_name}: {e}")
                raise
        else:
            logger.warning(f"No vectors generated for file {file_id} - file attachment may be incomplete")
        return file_id

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Chunk text into overlapping segments.

        Args:
            text: The text to chunk
            chunk_size: Target size for each chunk in characters
            overlap: Number of characters to overlap between chunks
        Returns:
            List of text chunks
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if current_chunk and len(current_chunk) + len(para) > chunk_size:
                chunks.append(current_chunk.strip())
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        if current_chunk:
            chunks.append(current_chunk.strip())
        if not chunks or any(len(c) > chunk_size * 2 for c in chunks):
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i : i + chunk_size]
                if chunk:
                    chunks.append(chunk)
        return chunks

    async def list_assistant_files(self, assistant_id: str) -> List[Dict[str, str]]:
        """List all files attached to an assistant."""
        assistant_path = self.base_path / assistant_id
        try:
            if not await aiofiles.os.path.exists(str(assistant_path)):
                return []
            files = []
            entries = await aiofiles.os.listdir(str(assistant_path))
            for file_id in entries:
                file_path = assistant_path / file_id
                if await aiofiles.os.path.isfile(str(file_path)):
                    files.append({"file_id": file_id, "path": str(file_path)})
            return files
        except OSError as e:
            logger.warning(f"Error listing files for assistant {assistant_id}: {e}")
            return []