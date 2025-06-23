"""Dependencies for vector stores."""

from typing import Annotated

from fastapi import Depends, Request

from rose_server.vector import ChromaDBManager


def get_vector_manager(request: Request) -> ChromaDBManager:
    """Get the vector manager from app state."""
    return request.app.state.vector


VectorManager = Annotated[ChromaDBManager, Depends(get_vector_manager)]
