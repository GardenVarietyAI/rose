"""Vector store CRUD operations."""

import time
import uuid
from typing import Optional

from rose_server.database import get_session
from rose_server.entities.vector_stores import VectorStore


async def create_vector_store(name: str) -> VectorStore:
    """Create a new vector store."""
    vector_store = VectorStore(
        id=f"vs_{uuid.uuid4().hex[:24]}",
        object="vector_store",
        name=name,
        dimensions=384,  # Default for bge-small-en-v1.5
        created_at=int(time.time()),
        last_used_at=None,
        meta={}
    )
    
    async with get_session() as session:
        session.add(vector_store)
        await session.commit()
        return vector_store


async def get_vector_store(vector_store_id: str) -> Optional[VectorStore]:
    """Get vector store by ID."""
    async with get_session(read_only=True) as session:
        return await session.get(VectorStore, vector_store_id)