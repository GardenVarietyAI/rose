"""Vector store CRUD operations."""

import asyncio
import json
import time
from typing import List, Optional

import numpy as np
from chonkie import TokenChunker
from sqlalchemy import text
from sqlmodel import select

from rose_server.config.settings import settings
from rose_server.database import get_session
from rose_server.embeddings.embedding import embedding_model
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import Document, DocumentSearchResult, VectorStore


async def create_vector_store(name: str) -> VectorStore:
    """Create a new vector store."""
    vector_store = VectorStore(
        object="vector_store",
        name=name,
        dimensions=settings.default_embedding_dimensions,
        created_at=int(time.time()),
        last_used_at=None,
        meta={},
    )

    async with get_session() as session:
        session.add(vector_store)
        await session.commit()
        return vector_store


async def get_vector_store(vector_store_id: str) -> Optional[VectorStore]:
    """Get vector store by ID."""
    async with get_session(read_only=True) as session:
        return await session.get(VectorStore, vector_store_id)


async def list_vector_stores() -> List[VectorStore]:
    """List all vector stores."""
    async with get_session(read_only=True) as session:
        result = await session.execute(select(VectorStore))
        return [row[0] for row in result.fetchall()]


async def add_file_to_vector_store(vector_store_id: str, file_id: str) -> Document:
    """Add a file to a vector store by chunking and embedding it."""
    async with get_session() as session:
        # Get the file content
        file_result = await session.execute(select(UploadedFile).where(UploadedFile.id == file_id))
        file_row = file_result.fetchone()
        if not file_row:
            raise ValueError(f"File {file_id} not found")

        uploaded_file = file_row[0]
        if not uploaded_file.content:
            raise ValueError(f"File {file_id} has no content")

        content = uploaded_file.content.decode("utf-8")

        # Chunk the content using Chonkie
        chunker = TokenChunker(
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
            tokenizer="tiktoken"
        )
        chunks = chunker.chunk(content)

        # Generate embeddings for all chunks
        model = embedding_model()
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await asyncio.to_thread(model.embed, chunk_texts)

        # Dimension validation
        expected_dim = settings.default_embedding_dimensions
        if embeddings:
            got_dim = len(embeddings[0])
            if got_dim != expected_dim:
                raise ValueError(f"Embedding dimension mismatch: got {got_dim}, expected {expected_dim}")

        created_at = int(time.time())
        first_document: Optional[Document] = None

        # Process each chunk
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create document entry for each chunk
            document = Document(
                vector_store_id=vector_store_id,
                chunk_index=idx,
                content=chunk.text,
                meta={"file_id": file_id, "filename": uploaded_file.filename, "total_chunks": len(chunks)},
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

            if first_document is None:
                first_document = document

        # Update vector store last_used_at on ingest
        vector_store = await session.get(VectorStore, vector_store_id)
        if vector_store:
            vector_store.last_used_at = created_at

        await session.commit()
        # Return the first chunk's document for backward compatibility
        return first_document or Document(
            vector_store_id=vector_store_id,
            chunk_index=0,
            content="",
            meta={"file_id": file_id, "filename": uploaded_file.filename},
            created_at=created_at,
        )


async def search_vector_store(
    vector_store_id: str, query: str, max_results: int = 10, update_last_used: bool = True
) -> List[DocumentSearchResult]:
    """Search documents in a vector store using vector similarity."""
    async with get_session(read_only=not update_last_used) as session:
        # Generate query embedding
        model = embedding_model()
        query_embedding = await asyncio.to_thread(lambda: list(model.embed([query]))[0])
        query_blob = np.array(query_embedding, dtype=np.float32).tobytes()

        # Vector similarity search using cosine distance
        result = await session.execute(
            text("""
                SELECT d.id, d.vector_store_id, d.chunk_index, d.content, d.meta, d.created_at,
                       vec_distance_cosine(v.embedding, :query_vector) as distance
                FROM documents d
                JOIN vec0 v ON d.id = v.document_id
                WHERE d.vector_store_id = :vector_store_id
                ORDER BY distance ASC, d.created_at DESC, d.id
                LIMIT :max_results
            """),
            {"query_vector": query_blob, "vector_store_id": vector_store_id, "max_results": max_results},
        )

        # Convert results to DocumentSearchResult objects with scores
        results = []
        for row in result.fetchall():
            # Parse meta JSON string back to dict
            raw_meta = row[4]
            if isinstance(raw_meta, (dict, list)):
                meta = raw_meta
            else:
                meta = json.loads(raw_meta) if raw_meta else {}
            doc = Document(
                id=row[0], vector_store_id=row[1], chunk_index=row[2], content=row[3], meta=meta, created_at=row[5]
            )
            distance = row[6]
            results.append(DocumentSearchResult(document=doc, score=distance))

        # Update last_used_at timestamp if requested
        if update_last_used and results:
            vector_store = await session.get(VectorStore, vector_store_id)
            if vector_store:
                vector_store.last_used_at = int(time.time())
                session.add(vector_store)

        return results
