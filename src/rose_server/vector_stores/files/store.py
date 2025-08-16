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


async def add_file_to_vector_store(vector_store_id: str, file_id: str) -> VectorStoreFile:
    """Add a file to a vector store by chunking and embedding it."""
    async with get_session() as session:
        # Check if file is already in vector store
        existing = await session.execute(
            select(VectorStoreFile).where(
                VectorStoreFile.vector_store_id == vector_store_id, VectorStoreFile.file_id == file_id
            )
        )
        existing_file = existing.fetchone()
        if existing_file:
            return existing_file[0]

        # Create status tracking record
        vector_store_file = VectorStoreFile(
            vector_store_id=vector_store_id, file_id=file_id, status="in_progress", created_at=int(time.time())
        )
        session.add(vector_store_file)
        await session.flush()

        try:
            # Get the file content
            file_result = await session.execute(select(UploadedFile).where(UploadedFile.id == file_id))
            file_row = file_result.fetchone()
            if not file_row:
                raise ValueError(f"File {file_id} not found")

            uploaded_file = file_row[0]
            if not uploaded_file.content:
                raise ValueError(f"File {file_id} has no content")

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

            # Chunk the content using Chonkie with our cached tokenizer
            tokenizer = get_tokenizer(settings.default_embedding_model)
            chunker = TokenChunker(
                chunk_size=settings.default_chunk_size,
                chunk_overlap=settings.default_chunk_overlap,
                tokenizer=tokenizer,
            )
            chunks = chunker.chunk(content)

            # Generate embeddings for all chunks
            model = embedding_model()
            chunk_texts = [chunk.text for chunk in chunks]  # type: ignore
            embeddings = await asyncio.to_thread(lambda: list(model.embed(chunk_texts)))

            # Dimension validation
            expected_dim = settings.default_embedding_dimensions
            if embeddings:
                got_dim = len(embeddings[0])
                if got_dim != expected_dim:
                    raise ValueError(f"Embedding dimension mismatch: got {got_dim}, expected {expected_dim}")

            if not chunks:
                raise ValueError(f"No chunks generated from file {file_id}")

            # Process each chunk
            documents = []
            created_at = int(time.time())
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create document entry for each chunk
                chunk_meta = {
                    "file_id": file_id,
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
                await session.flush()  # Get the document.id

                # Store embedding in vec0 virtual table
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await session.execute(
                    text("INSERT OR REPLACE INTO vec0 (document_id, embedding) VALUES (:doc_id, :embedding)"),
                    {"doc_id": document.id, "embedding": embedding_blob},
                )

                documents.append(document)

            # Update vector store last_used_at on ingest
            vector_store = await session.get(VectorStore, vector_store_id)
            if vector_store:
                vector_store.last_used_at = created_at

            # Mark processing as completed
            vector_store_file.status = "completed"
            await session.commit()
            return vector_store_file

        except Exception:
            await session.rollback()
            vector_store_file.status = "failed"
            await session.commit()
            raise
