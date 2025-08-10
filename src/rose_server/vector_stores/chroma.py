import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class Chroma:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, persist_dir: Optional[str] = None):
        """Initialize ChromaDB manager.

        Args:
            host: ChromaDB host (defaults to localhost)
            port: ChromaDB port (defaults to 8003)
            persist_dir: Local persistence directory (defaults to ./data/chroma)
        """
        self.host = host or os.getenv("CHROMA_HOST", "localhost")
        self.port = port or int(os.getenv("CHROMA_PORT", "8003"))
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self._client: Optional[chromadb.Client] = None
        self._default_embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self._init_client()

    def _init_client(self) -> None:
        try:
            self._client = chromadb.HttpClient(host=self.host, port=self.port)
            logger.info(f"Connected to ChromaDB at {self.host}:{self.port}")
            self._client.heartbeat()
        except Exception as e:
            logger.warning(f"Failed to connect to ChromaDB server at {self.host}:{self.port}: {e}")
            logger.info(f"Falling back to persistent client at {self.persist_dir}")
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)

    @property
    def client(self) -> chromadb.Client:
        if self._client is None:
            self._init_client()
        return self._client

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None, embedding_function: Optional[Any] = None
    ) -> Collection:
        """Get or create a ChromaDB collection.
        Args:
            name: Collection name
            metadata: Optional collection metadata
            embedding_function: Optional embedding function
        Returns:
            ChromaDB collection instance
        """
        try:
            collection = self.client.get_collection(name=name)
            logger.debug(f"Retrieved existing collection: {name}")
        except Exception:
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {},
                embedding_function=embedding_function or self._default_embedding_function,
            )
            logger.info(f"Created new collection: {name}")
        return collection

    def delete_collection(self, name: str) -> bool:
        """Delete a ChromaDB collection.

        Args:
            name: Collection name to delete
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            self.client.delete_collection(name=name)
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection {name}: {e}")
            return False

    def list_collections(self) -> List[str]:
        """List all ChromaDB collections.

        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Collection name to check
        Returns:
            True if collection exists
        """
        return name in self.list_collections()

    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to a collection.
        Args:
            collection_name: Name of the collection
            vectors: List of embedding vectors
            documents: List of document texts
            ids: List of unique IDs for the vectors
            metadatas: Optional list of metadata dicts
        """
        collection = self.get_or_create_collection(collection_name)
        collection.add(embeddings=vectors, documents=documents, ids=ids, metadatas=metadatas)
        logger.debug(f"Added {len(vectors)} vectors to collection {collection_name}")

    def query_vectors(
        self,
        collection_name: str,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query vectors from a collection.
        Args:
            collection_name: Name of the collection
            query_embeddings: Query embedding vectors
            query_texts: Query texts (will be embedded automatically)
            n_results: Number of results to return
            where: Metadata filter conditions
            include: Fields to include in results
        Returns:
            Query results dict
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            results = collection.query(
                query_embeddings=query_embeddings,
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                include=include or ["documents", "metadatas", "distances"],
            )
            logger.debug(f"Queried collection {collection_name}, got {len(results.get('ids', []))} results")
            return results
        except Exception as e:
            logger.error(f"Failed to query collection {collection_name}: {e}")
            raise

    def delete_vectors(self, collection_name: str, ids: List[str]) -> None:
        """Delete vectors from a collection.

        Args:
            collection_name: Name of the collection
            ids: IDs of vectors to delete
        """
        collection = self.get_or_create_collection(collection_name)
        collection.delete(ids=ids)
        logger.debug(f"Deleted {len(ids)} vectors from collection {collection_name}")

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection.

        Args:
            collection_name: Name of the collection
        Returns:
            Collection information dict
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            return {"name": collection.name, "count": collection.count(), "metadata": collection.metadata}
        except Exception as e:
            logger.error(f"Failed to get info for collection {collection_name}: {e}")
            raise
