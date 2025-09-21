import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import text
from sqlmodel import delete, func, select

from rose_server.database import get_session
from rose_server.entities.vector_stores import Document, DocumentSearchResult, VectorStore, VectorStoreFile

logger = logging.getLogger(__name__)


async def create_vector_store(name: str, dimensions: int) -> VectorStore:
    """Create a new vector store."""
    vector_store = VectorStore(
        object="vector_store",
        name=name,
        dimensions=dimensions,
        last_used_at=None,
    )

    async with get_session() as session:
        session.add(vector_store)
        await session.commit()
        return vector_store


async def get_vector_store(vector_store_id: str) -> Optional[VectorStore]:
    async with get_session(read_only=True) as session:
        return await session.get(VectorStore, vector_store_id)


async def update_vector_store(
    vector_store_id: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
) -> Optional[VectorStore]:
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
    async with get_session(read_only=True) as session:
        result = await session.execute(select(VectorStore))
        return [row[0] for row in result.fetchall()]


async def search_vector_store(
    vector_store_id: str, query_embedding: List[float], max_results: int = 10, update_last_used: bool = True
) -> List[DocumentSearchResult]:
    async with get_session(read_only=not update_last_used) as session:
        # Get the vector store to check dimensions
        vector_store = await session.get(VectorStore, vector_store_id)
        if not vector_store:
            raise ValueError(f"Vector store {vector_store_id} not found")

        got_dim = len(query_embedding)
        if got_dim != vector_store.dimensions:
            raise ValueError(
                f"Query vector dimension mismatch: got {got_dim}, expected {vector_store.dimensions}",
            )

        query_blob = np.array(query_embedding, dtype=np.float32).tobytes()
        max_results = max(1, min(100, max_results))

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
                id=row[0],
                vector_store_id=row[1],
                chunk_index=row[2],
                content=row[3],
                meta=meta,
                created_at=row[5],
            )

            distance = row[6]
            similarity = 1.0 - distance
            results.append(DocumentSearchResult(document=doc, score=similarity))

        return results


async def update_last_used_timestamp(vector_store_id: str) -> None:
    """Update the last_used_at timestamp for a vector store."""
    async with get_session() as session:
        vector_store = await session.get(VectorStore, vector_store_id)
        if vector_store:
            vector_store.last_used_at = int(time.time())
            session.add(vector_store)
            await session.commit()


async def delete_vector_store(vector_store_id: str) -> bool:
    async with get_session() as session:
        vector_store = await session.get(VectorStore, vector_store_id)

        if not vector_store:
            return False

        file_count_result = await session.execute(
            select(func.count(VectorStoreFile.id)).where(VectorStoreFile.vector_store_id == vector_store_id)
        )

        file_count = file_count_result.scalar()
        await session.execute(delete(VectorStoreFile).where(VectorStoreFile.vector_store_id == vector_store_id))
        await session.delete(vector_store)
        await session.commit()

        logger.info(f"Deleted vector store: {vector_store_id} and {file_count} files")
        return True
