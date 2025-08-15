"""Dependencies for vector stores."""

import logging
from typing import Annotated, Any

from fastapi import Depends

logger = logging.getLogger(__name__)


class StubVectorManager:
    """Stub vector manager that logs warnings when ChromaDB functionality is used."""

    def __init__(self):
        logger.warning("VectorManager initialized without ChromaDB support")

    def list_collections(self):
        logger.warning("list_collections called but ChromaDB is not available")
        return []

    def get_collection_info(self, name: str):
        logger.warning(f"get_collection_info({name}) called but ChromaDB is not available")
        return {"metadata": {}}

    def get_or_create_collection(self, name: str, **kwargs):
        logger.warning(f"get_or_create_collection({name}) called but ChromaDB is not available")
        return self

    def delete_collection(self, name: str):
        logger.warning(f"delete_collection({name}) called but ChromaDB is not available")
        return True

    @property
    def client(self):
        logger.warning("ChromaDB client accessed but not available")
        return self

    def get_collection(self, name: str):
        logger.warning(f"get_collection({name}) called but ChromaDB is not available")
        return self

    def delete(self, **kwargs):
        logger.warning("delete called but ChromaDB is not available")
        return None

    def query(self, **kwargs):
        logger.warning("query called but ChromaDB is not available")
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

    def modify(self, **kwargs):
        logger.warning("modify called but ChromaDB is not available")
        return None


def get_vector_manager() -> StubVectorManager:
    """Get a stub vector manager that logs warnings."""
    return StubVectorManager()


VectorManager = Annotated[Any, Depends(get_vector_manager)]
