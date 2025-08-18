"""Vector store file operations."""

import asyncio
import logging
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from chonkie import TokenChunker
from sqlalchemy import bindparam, delete, select, text, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from rose_server.config.settings import settings
from rose_server.database import get_session
from rose_server.embeddings.embedding import embedding_model, get_tokenizer
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import Document, VectorStore, VectorStoreFile

logger = logging.getLogger(__name__)


class VectorStoreNotFoundError(ValueError):
    """Vector store does not exist."""


class FileNotFoundError(ValueError):
    """File does not exist."""


class EmptyFileError(ValueError):
    """File has no content."""


class ChunkingError(ValueError):
    """Failed to generate chunks from file."""


def _decode_file_content(uploaded_file: UploadedFile, file_id: str) -> Tuple[str, bool]:
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


def _get_chunker() -> TokenChunker:
    return TokenChunker(
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
        tokenizer=get_tokenizer(settings.default_embedding_model),
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
    created_at = int(time.time())
    documents = []
    # Create all documents first
    for idx, chunk in enumerate(chunks):
        chunk_meta = {
            "file_id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "total_chunks": len(chunks),
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
        }
        if decode_errors:
            chunk_meta["decode_errors"] = True

        document = Document(vector_store_id=vector_store_id, chunk_index=idx, content=chunk.text, meta=chunk_meta)
        session.add(document)
        documents.append(document)

    # Batch flush all documents
    await session.flush()

    # Insert embeddings in batch
    embedding_data = []
    for doc, embedding in zip(documents, embeddings):
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        embedding_data.append({"doc_id": doc.id, "embedding": embedding_blob})

    await session.execute(
        text("INSERT OR REPLACE INTO vec0 (document_id, embedding) VALUES (:doc_id, :embedding)"), embedding_data
    )

    return created_at


async def add_file_to_vector_store(vector_store_id: str, file_id: str) -> VectorStoreFile:
    async with get_session() as session:
        vector_store = await session.get(VectorStore, vector_store_id)
        if not vector_store:
            raise VectorStoreNotFoundError(f"Vector store {vector_store_id} not found")

        uploaded_file = await session.get(UploadedFile, file_id)
        if not uploaded_file:
            raise FileNotFoundError(f"Uploaded file {file_id} not found")

        # Upsert
        await session.execute(
            insert(VectorStoreFile)
            .values(vector_store_id=vector_store_id, file_id=file_id, status="in_progress")
            .on_conflict_do_nothing(index_elements=[VectorStoreFile.vector_store_id, VectorStoreFile.file_id])
        )

        # Get the newly inserted or already existent row
        vector_store_file = await session.scalar(
            select(VectorStoreFile).where(
                VectorStoreFile.vector_store_id == vector_store_id,
                VectorStoreFile.file_id == file_id,
            )
        )

        if vector_store_file.status != "in_progress":
            return vector_store_file  # don’t double-ingest

        try:
            content, decode_errors = _decode_file_content(uploaded_file, file_id)
            chunker = _get_chunker()
            chunks = chunker.chunk(content)
            if not chunks:
                raise ChunkingError(f"No chunks generated from file {file_id}")

            texts = [c.text for c in chunks]

            model = embedding_model()
            embeddings = await asyncio.to_thread(lambda: list(model.embed(texts)))
            if embeddings:
                got, exp = len(embeddings[0]), settings.default_embedding_dimensions
                if got != exp:
                    raise ValueError(f"Embedding dimension mismatch: got {got}, expected {exp}")

            created_at = await _store_chunk_documents(
                session,
                vector_store_id,
                uploaded_file,
                chunks,
                embeddings,
                decode_errors,
            )

            vector_store.last_used_at = created_at
            vector_store_file.status = "completed"
            await session.execute(
                update(VectorStoreFile)
                .where(
                    VectorStoreFile.vector_store_id == vector_store_id,
                    VectorStoreFile.file_id == file_id,
                )
                .values(status="completed")
            )
            await session.commit()

            return vector_store_file

        except Exception:
            await session.rollback()
            # mark failed in a fresh tx (so the status actually persists)
            async with get_session() as s:
                await s.execute(
                    update(VectorStoreFile)
                    .where(
                        VectorStoreFile.vector_store_id == vector_store_id,
                        VectorStoreFile.file_id == file_id,
                    )
                    .values(status="failed")
                )
                await s.commit()
            raise


async def get_vector_store_file(vector_store_file_id: str) -> Optional[VectorStoreFile]:
    """Get vector store file by ID."""
    async with get_session(read_only=True) as session:
        return await session.get(VectorStoreFile, vector_store_file_id)


async def list_vector_store_files(
    vector_store_id: str,
    limit: int = 20,
    order: str = "desc",
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> List[VectorStoreFile]:
    """List files in a vector store with pagination."""
    async with get_session() as session:
        query = select(VectorStoreFile).where(VectorStoreFile.vector_store_id == vector_store_id)

        if after:
            query = query.where(VectorStoreFile.id > after)
        if before:
            query = query.where(VectorStoreFile.id < before)

        if order == "asc":
            query = query.order_by(VectorStoreFile.created_at.asc())
        else:
            query = query.order_by(VectorStoreFile.created_at.desc())

        query = query.limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())


async def delete_file_from_vector_store(vector_store_id: str, file_id: str) -> int:
    """Remove a file from a vector store and delete associated documents & embeddings.
    Returns the number of documents deleted.
    """
    async with get_session() as session:
        # Find the mapping row using both keys (safer than .get with a single id)
        vector_store_file = await session.scalar(
            select(VectorStoreFile).where(
                VectorStoreFile.vector_store_id == vector_store_id,
                VectorStoreFile.file_id == file_id,
            )
        )
        if not vector_store_file:
            raise FileNotFoundError(f"File {file_id} not found in vector store {vector_store_id}")

        # Collect document ids for this file in this vector store
        rows = (
            await session.execute(select(Document.id, Document.meta).where(Document.vector_store_id == vector_store_id))
        ).all()
        doc_ids = [doc_id for (doc_id, meta) in rows if meta and meta.get("file_id") == file_id]

        if doc_ids:
            # Delete embeddings in vec0 for these docs (expanding IN)
            stmt_vec_del = text("DELETE FROM vec0 WHERE document_id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            )
            await session.execute(stmt_vec_del, {"ids": doc_ids})

            # Bulk delete documents
            await session.execute(delete(Document).where(Document.id.in_(doc_ids)))

        # Remove the vector store ↔ file link
        await session.delete(vector_store_file)
        await session.commit()
        return len(doc_ids)
