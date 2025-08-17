"""Vector store CRUD operations."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sqlalchemy import text
from sqlmodel import select

from rose_server.config.settings import settings
from rose_server.database import get_session
from rose_server.embeddings.embedding import embedding_model
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
    )

    async with get_session() as session:
        session.add(vector_store)
        await session.commit()
        return vector_store


async def get_vector_store(vector_store_id: str) -> Optional[VectorStore]:
    """Get vector store by ID."""
    async with get_session(read_only=True) as session:
        return await session.get(VectorStore, vector_store_id)


async def update_vector_store(
    vector_store_id: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
) -> Optional[VectorStore]:
    """Update thread metadata."""
    async with get_session() as session:
        vector_store = await session.get(VectorStore, vector_store_id)

        if not vector_store:
            return None

        if name is not None:
            vector_store.name = name

        if metadata is not None:
            base = (vector_store.meta or {}).copy()
            base.update(metadata)
            vector_store.meta = base  # reassign so SQLAlchemy tracks the change

        await session.flush()
        await session.refresh(vector_store)
        return vector_store


async def list_vector_stores() -> List[VectorStore]:
    """List all vector stores."""
    async with get_session(read_only=True) as session:
        result = await session.execute(select(VectorStore))
        return [row[0] for row in result.fetchall()]


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
