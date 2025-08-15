"""Vector store CRUD operations."""

import time
from typing import List, Optional

from sqlmodel import select

from rose_server.database import get_session
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import Document, VectorStore


async def create_vector_store(name: str) -> VectorStore:
    """Create a new vector store."""
    vector_store = VectorStore(
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
        
        # Create a document entry for the file
        document = Document(
            vector_store_id=vector_store_id,
            chunk_index=0,
            content=uploaded_file.content.decode('utf-8'),
            meta={"file_id": file_id, "filename": uploaded_file.filename},
            created_at=int(time.time()),
        )
        session.add(document)
        await session.commit()
        return document


async def search_vector_store(vector_store_id: str, query: str, max_results: int = 10) -> List[Document]:
    """Search documents in a vector store using text matching."""
    async with get_session(read_only=True) as session:
        # Simple text search for now - will add vector search later
        result = await session.execute(
            select(Document)
            .where(Document.vector_store_id == vector_store_id)
            .where(Document.content.contains(query))
            .limit(max_results)
        )
        return [row[0] for row in result.fetchall()]