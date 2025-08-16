"""Vector store CRUD operations."""

import asyncio
import json
import logging
import time
from typing import List, Optional, Union

import numpy as np
from chonkie import TokenChunker
from sqlalchemy import text
from sqlmodel import select

from rose_server.config.settings import settings
from rose_server.database import get_session
from rose_server.embeddings.embedding import embedding_model, get_tokenizer
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import Document, DocumentSearchResult, VectorStore

logger = logging.getLogger(__name__)


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
            chunk_size=settings.default_chunk_size, chunk_overlap=settings.default_chunk_overlap, tokenizer=tokenizer
        )
        chunks = chunker.chunk(content)

        # Generate embeddings for all chunks
        model = embedding_model()
        chunk_texts = []
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'text'):
                chunk_texts.append(chunk.text)
            else:
                raise AttributeError(f"Chunk at index {i} does not have a 'text' attribute: {repr(chunk)}")
        embeddings = await asyncio.to_thread(lambda: list(model.embed(chunk_texts)))

        # Dimension validation
        expected_dim = settings.default_embedding_dimensions
        if embeddings:
            got_dim = len(embeddings[0])
            if got_dim != expected_dim:
                raise ValueError(f"Embedding dimension mismatch: got {got_dim}, expected {expected_dim}")

        created_at = int(time.time())
        
        if not chunks:
            raise ValueError(f"No chunks generated from file {file_id}")

        # Process each chunk
        documents = []
        try:
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create document entry for each chunk
                # Validate chunk structure
                start_index = getattr(chunk, 'start_index', None)
                end_index = getattr(chunk, 'end_index', None)
                if start_index is None or end_index is None:
                    raise AttributeError(f"Chunk object missing 'start_index' or 'end_index' attribute: {chunk}")
                
                chunk_meta = {
                    "file_id": file_id, 
                    "filename": uploaded_file.filename, 
                    "total_chunks": len(chunks),
                    "start_index": start_index,
                    "end_index": end_index,
                }
                if decode_errors:
                    chunk_meta["decode_errors"] = True
                    
                document = Document(
                    vector_store_id=vector_store_id,
                    chunk_index=idx,
                    content=chunk.text,
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

            await session.commit()
            return documents[0]
        except Exception:
            await session.rollback()
            raise


async def search_vector_store(
    vector_store_id: str, query: Union[str, List[float]], max_results: int = 10, update_last_used: bool = True
) -> List[DocumentSearchResult]:
    """Search documents in a vector store using vector similarity."""
    max_results = max(1, min(100, max_results))
    
    async with get_session(read_only=not update_last_used) as session:
        # Handle both text and vector queries
        if isinstance(query, str):
            # Generate query embedding
            model = embedding_model()
            query_embedding = await asyncio.to_thread(lambda: list(model.embed([query]))[0])
        else:
            # Direct vector input - validate dimensions
            expected_dim = settings.default_embedding_dimensions
            got_dim = len(query)
            if got_dim != expected_dim:
                raise ValueError(f"Query vector dimension mismatch: got {got_dim}, expected {expected_dim}")
            query_embedding = query

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
            similarity = 1.0 - distance
            results.append(DocumentSearchResult(document=doc, score=similarity))

        # Update last_used_at timestamp if requested (even if 0 hits)
        if update_last_used:
            vector_store = await session.get(VectorStore, vector_store_id)
            if vector_store:
                vector_store.last_used_at = int(time.time())
                session.add(vector_store)

        return results
