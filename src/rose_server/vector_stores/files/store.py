"""Vector store file operations."""

import asyncio
import logging
import time

import numpy as np
from chonkie import TokenChunker
from sqlalchemy import text
from sqlmodel import select

from rose_server.config.settings import settings
from rose_server.database import get_session
from rose_server.embeddings.embedding import embedding_model, get_tokenizer
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import Document, VectorStore, VectorStoreFile

logger = logging.getLogger(__name__)


class VectorStoreNotFoundError(ValueError):
    """Vector store does not exist."""

    pass


class FileNotFoundError(ValueError):
    """File does not exist."""

    pass


class EmptyFileError(ValueError):
    """File has no content."""

    pass


class ChunkingError(ValueError):
    """Failed to generate chunks from file."""

    pass


async def _get_existing_file(session, vector_store_id: str, file_id: str):
    """Check if file is already in vector store."""
    existing = await session.execute(
        select(VectorStoreFile).where(
            VectorStoreFile.vector_store_id == vector_store_id, VectorStoreFile.file_id == file_id
        )
    )
    existing_file = existing.fetchone()
    return existing_file[0] if existing_file else None


async def _get_uploaded_file(session, file_id: str):
    """Get uploaded file and validate it exists."""
    file_result = await session.execute(select(UploadedFile).where(UploadedFile.id == file_id))
    file_row = file_result.fetchone()
    if not file_row:
        raise FileNotFoundError(f"File {file_id} not found")

    uploaded_file = file_row[0]
    if not uploaded_file.content:
        raise EmptyFileError(f"File {file_id} has no content")
    return uploaded_file


def _decode_file_content(uploaded_file, file_id: str):
    """Decode file content with error handling."""
    try:
        content = uploaded_file.content.decode("utf-8")
        decode_errors = False
    except UnicodeDecodeError:
        content = uploaded_file.content.decode("utf-8", errors="replace")
        decode_errors = True
        logger.warning(
            f"File {file_id} ({uploaded_file.filename}) contains invalid UTF-8 bytes. "
            f"Decoded with replacement characters."
        )
    return content, decode_errors


async def _process_file_chunks(content: str, file_id: str):
    """Chunk content and generate embeddings."""
    tokenizer = get_tokenizer(settings.default_embedding_model)
    chunker = TokenChunker(
        chunk_size=settings.default_chunk_size, chunk_overlap=settings.default_chunk_overlap, tokenizer=tokenizer
    )
    chunks = chunker.chunk(content)

    if not chunks:
        raise ChunkingError(f"No chunks generated from file {file_id}")

    model = embedding_model()
    chunk_texts = [chunk.text for chunk in chunks]  # type: ignore
    embeddings = await asyncio.to_thread(lambda: list(model.embed(chunk_texts)))

    expected_dim = settings.default_embedding_dimensions
    if embeddings:
        got_dim = len(embeddings[0])
        if got_dim != expected_dim:
            raise ValueError(f"Embedding dimension mismatch: got {got_dim}, expected {expected_dim}")

    return chunks, embeddings


async def _store_chunk_documents(session, vector_store_id: str, uploaded_file, chunks, embeddings, decode_errors: bool):
    """Store document chunks with embeddings."""
    created_at = int(time.time())

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_meta = {
            "file_id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "total_chunks": len(chunks),
            "start_index": chunk.start_index,  # type: ignore
            "end_index": chunk.end_index,  # type: ignore
        }
        if decode_errors:
            chunk_meta["decode_errors"] = True

        document = Document(
            vector_store_id=vector_store_id,
            chunk_index=idx,
            content=chunk.text,  # type: ignore
            meta=chunk_meta,
            created_at=created_at,
        )
        session.add(document)
        await session.flush()

        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        await session.execute(
            text("INSERT OR REPLACE INTO vec0 (document_id, embedding) VALUES (:doc_id, :embedding)"),
            {"doc_id": document.id, "embedding": embedding_blob},
        )

    return created_at


async def add_file_to_vector_store(vector_store_id: str, file_id: str) -> VectorStoreFile:
    """Add a file to a vector store by chunking and embedding it."""
    async with get_session() as session:
        # Validate vector store exists
        vector_store = await session.get(VectorStore, vector_store_id)
        if not vector_store:
            raise VectorStoreNotFoundError(f"Vector store {vector_store_id} not found")

        # Try to create record, handling race conditions with database constraint
        vector_store_file = VectorStoreFile(
            vector_store_id=vector_store_id, file_id=file_id, status="in_progress", created_at=int(time.time())
        )

        try:
            session.add(vector_store_file)
            await session.flush()
        except Exception:
            # Race condition: file already exists, get existing record
            await session.rollback()
            existing = await _get_existing_file(session, vector_store_id, file_id)
            if existing:
                return existing
            raise

        try:
            uploaded_file = await _get_uploaded_file(session, file_id)
            content, decode_errors = _decode_file_content(uploaded_file, file_id)
            chunks, embeddings = await _process_file_chunks(content, file_id)
            created_at = await _store_chunk_documents(
                session, vector_store_id, uploaded_file, chunks, embeddings, decode_errors
            )

            vector_store.last_used_at = created_at

            vector_store_file.status = "completed"
            await session.commit()
            return vector_store_file

        except Exception:
            await session.rollback()
            vector_store_file.status = "failed"
            await session.commit()
            raise
