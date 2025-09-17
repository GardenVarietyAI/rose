"""Vector store file operations."""

import io
import logging
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from chonkie import TokenChunker
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from rose_server.database import get_session
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import Document, VectorStore, VectorStoreFile

logger = logging.getLogger(__name__)
PDF_MAGIC_BYTES = b"%PDF-"


class VectorStoreNotFoundError(ValueError):
    """Vector store does not exist."""


class FileNotFoundError(ValueError):
    """File does not exist."""


class EmptyFileError(ValueError):
    """File has no content."""


class ChunkingError(ValueError):
    """Failed to generate chunks from file."""


def decode_file_content(content: bytes, filename: str) -> Tuple[str, bool]:
    """Pure function to decode file content with PDF and text support."""
    if not content:
        raise EmptyFileError(f"File {filename} has no content")

    if content.startswith(PDF_MAGIC_BYTES):
        try:
            # Create BytesIO wrapper for pypdf (content already in memory from upload)
            reader = PdfReader(io.BytesIO(content))

            # Extract text from all pages
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            text_content = "\n\n".join(text_parts)
            if not text_content.strip():
                raise ValueError("No text content found in PDF")

            return text_content, False

        except (PdfReadError, ValueError) as e:
            raise ValueError(f"Failed to process PDF file: {str(e)}")

    # Handle text files
    try:
        text_content = content.decode("utf-8")
        decode_errors = False
    except UnicodeDecodeError:
        text_content = content.decode("utf-8", errors="replace")
        decode_errors = True

    return text_content, decode_errors


def create_chunks(text: str, tokenizer: Any, chunk_size: int, chunk_overlap: int) -> List[Any]:
    chunker = TokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
    )
    return chunker.chunk(text)


async def store_file_chunks_with_embeddings(
    vector_store_id: str,
    file_id: str,
    chunks: List[Any],
    embeddings: List[np.ndarray],
    decode_errors: bool,
) -> VectorStoreFile:
    """Store file chunks with pre-computed embeddings."""
    async with get_session() as session:
        vector_store = await session.get(VectorStore, vector_store_id)
        if not vector_store:
            raise VectorStoreNotFoundError(f"Vector store {vector_store_id} not found")

        uploaded_file = await session.get(UploadedFile, file_id)
        if not uploaded_file:
            raise FileNotFoundError(f"Uploaded file {file_id} not found")

        # Upsert vector store file record
        await session.execute(
            insert(VectorStoreFile)
            .values(vector_store_id=vector_store_id, file_id=file_id)
            .on_conflict_do_nothing(index_elements=[VectorStoreFile.vector_store_id, VectorStoreFile.file_id])
        )

        # Get the vector store file record
        vector_store_file = await session.scalar(
            select(VectorStoreFile).where(
                VectorStoreFile.vector_store_id == vector_store_id,
                VectorStoreFile.file_id == file_id,
            )
        )

        if vector_store_file.status != "in_progress":
            return vector_store_file  # don't double-ingest

        try:
            # Store chunks and embeddings
            created_at = await _store_chunk_documents(
                session,
                vector_store_id,
                uploaded_file,
                chunks,
                embeddings,
                decode_errors,
            )

            # Mark as completed
            vector_store_file.status = "completed"
            vector_store_file.last_error = None
            await session.commit()

            # Update last_used_at
            await session.execute(
                update(VectorStore).where(VectorStore.id == vector_store_id).values(last_used_at=created_at)
            )
            await session.commit()

            return vector_store_file

        except Exception as e:
            # Mark as failed
            logger.error(f"Failed to add file {file_id} to vector store {vector_store_id}: {str(e)}")
            vector_store_file.status = "failed"
            vector_store_file.last_error = str(e)
            await session.commit()
            raise


async def get_uploaded_file(file_id: str) -> UploadedFile:
    async with get_session(read_only=True) as session:
        uploaded_file = await session.get(UploadedFile, file_id)
        if not uploaded_file:
            raise FileNotFoundError(f"Uploaded file {file_id} not found")
        return uploaded_file


async def add_file_to_vector_store(
    vector_store_id: str, file_id: str, embeddings: List[np.ndarray], chunks: List[Any], decode_errors: bool
) -> VectorStoreFile:
    """Store a file with pre-computed embeddings and chunks."""
    return await store_file_chunks_with_embeddings(
        vector_store_id=vector_store_id,
        file_id=file_id,
        chunks=chunks,
        embeddings=embeddings,
        decode_errors=decode_errors,
    )


async def _store_chunk_documents(
    session: AsyncSession,
    vector_store_id: str,
    uploaded_file: UploadedFile,
    chunks: Sequence[Any],
    embeddings: List[Any],
    decode_errors: bool,
) -> int:
    """Store document chunks with embeddings."""
    # Create documents for all chunks
    created_at = int(time.time())
    documents = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        meta = {
            "file_id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "total_chunks": len(chunks),
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "decode_errors": decode_errors,
        }

        doc = Document(
            id=f"{uploaded_file.id}#{i}",
            vector_store_id=vector_store_id,
            chunk_index=i,
            content=chunk.text,
            meta=meta,
            created_at=created_at,
        )
        documents.append(doc)

    # Delete old docs for this file
    await session.execute(delete(Document).where(Document.id.like(f"{uploaded_file.id}#%")))

    # Insert new documents
    for doc in documents:
        session.add(doc)

    # Insert embeddings in batch
    embedding_data = []
    for doc, embedding in zip(documents, embeddings):
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        embedding_data.append({"doc_id": doc.id, "embedding": embedding_blob})

    await session.execute(
        text("INSERT OR REPLACE INTO vec0 (document_id, embedding) VALUES (:doc_id, :embedding)"), embedding_data
    )

    await session.commit()
    return created_at


async def get_vector_store_file(vector_store_id: str, file_id: str) -> Optional[VectorStoreFile]:
    """Get a specific file from a vector store."""
    async with get_session() as session:
        return await session.scalar(
            select(VectorStoreFile).where(
                VectorStoreFile.vector_store_id == vector_store_id,
                VectorStoreFile.file_id == file_id,
            )
        )


async def list_vector_store_files(
    vector_store_id: str, order: str = "asc", limit: int = 20, after: Optional[str] = None, before: Optional[str] = None
) -> Tuple[List[VectorStoreFile], bool]:
    """List files in a vector store."""
    async with get_session() as session:
        query = select(VectorStoreFile).where(VectorStoreFile.vector_store_id == vector_store_id)

        # Handle pagination cursors
        if after:
            query = query.where(VectorStoreFile.id > after)
        if before:
            query = query.where(VectorStoreFile.id < before)

        # Apply ordering
        if order == "desc":
            query = query.order_by(VectorStoreFile.created_at.desc())
        else:
            query = query.order_by(VectorStoreFile.created_at)

        # Get one extra to check if there are more
        query = query.limit(limit + 1)

        result = await session.execute(query)
        files = list(result.scalars().all())

        has_more = len(files) > limit
        if has_more:
            files = files[:limit]

        return files, has_more


async def remove_file_from_vector_store(vector_store_id: str, file_id: str) -> bool:
    """Remove a file from a vector store and delete associated documents & embeddings.

    Returns:
        bool: True if file was deleted, False if not found
    """
    async with get_session() as session:
        # Check if the file exists in this vector store
        vsf = await session.scalar(
            select(VectorStoreFile).where(
                VectorStoreFile.vector_store_id == vector_store_id,
                VectorStoreFile.file_id == file_id,
            )
        )

        if not vsf:
            return False

        # Delete documents and their embeddings
        doc_ids = await session.scalars(
            select(Document.id).where(Document.id.like(f"{file_id}#%"), Document.vector_store_id == vector_store_id)
        )
        doc_ids = list(doc_ids)

        if doc_ids:
            # Delete embeddings in vec0 for these docs (expanding IN)
            placeholders = ", ".join([f":doc_id_{i}" for i in range(len(doc_ids))])
            params = {f"doc_id_{i}": doc_id for i, doc_id in enumerate(doc_ids)}
            await session.execute(text(f"DELETE FROM vec0 WHERE document_id IN ({placeholders})"), params)

            # Delete documents
            await session.execute(delete(Document).where(Document.id.in_(doc_ids)))

        # Delete the VectorStoreFile record
        await session.delete(vsf)
        await session.commit()

        return True
