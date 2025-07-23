"""Dependencies for vector stores."""

from typing import Annotated

from fastapi import Depends, Request

from rose_server.vector_stores.chroma import Chroma


def get_vector_manager(request: Request) -> Chroma:
    """Get the vector manager from app state."""
    return request.app.state.chroma


VectorManager = Annotated[Chroma, Depends(get_vector_manager)]
