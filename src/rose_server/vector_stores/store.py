import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from rose_server.embeddings import generate_embeddings
from rose_server.model_registry import get_embedding_models
from rose_server.services import get_chromadb_manager
from rose_server.vector import ChromaDBManager
from rose_server.vector_stores.models import (
    Vector,
    VectorBatch,
    VectorSearch,
    VectorSearchResult,
    VectorStoreCreate,
    VectorStoreList,
    VectorStoreMetadata,
    VectorStoreUpdate,
)

logger = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
_META_EXCLUDE = {"display_name", "dimensions", "created_at"}


class VectorStoreNotFoundError(RuntimeError):
    pass


class VectorStoreStore:
    """OpenAI-compatible vector-store operations backed by ChromaDB."""

    def __init__(self) -> None:
        self._client = None

    def _manager(self) -> ChromaDBManager:
        if self._client:
            mgr = ChromaDBManager()
            mgr._client = self._client
            return mgr
        return get_chromadb_manager()

    def _collection(self, cid: str):
        mgr = self._manager()
        if cid not in mgr.list_collections():
            raise VectorStoreNotFoundError(cid)
        return mgr.get_collection(cid)

    def initialize(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self._client = ChromaDBManager(host=host, port=port).client
        logger.info("Vector-store backend initialised (%s:%s)", host, port)

    def initialize_with_client(self, client) -> None:
        self._client = client
        logger.info("Vector-store backend initialised with external client")

    async def list_vector_stores(self) -> VectorStoreList:
        mgr = self._manager()
        stores = []
        for name in mgr.list_collections():
            try:
                meta = mgr.get_collection_info(name).get("metadata", {})
                stores.append(
                    VectorStoreMetadata(
                        id=name,
                        name=meta.get("display_name", name),
                        dimensions=meta.get("dimensions", 0),
                        metadata=meta,
                        created_at=int(meta.get("created_at", time.time())),
                    )
                )
            except Exception as exc:
                logger.warning("Skip collection %s: %s", name, exc)
        return VectorStoreList(data=stores)

    async def create_vector_store(self, p: VectorStoreCreate) -> VectorStoreMetadata:
        vid = f"vs_{uuid.uuid4().hex}"
        meta_in = p.metadata or {}
        model = meta_in.get("model", DEFAULT_EMBEDDING_MODEL)
        try:
            sample_dim = len(generate_embeddings("x", model_name=model)["data"][0]["embedding"])
        except Exception as exc:
            registry = get_embedding_models()
            sample_dim = registry.get(model, registry[DEFAULT_EMBEDDING_MODEL])["dimensions"]
            logger.warning("Fallback to registry dimension for %s (reason: %s)", model, exc)
        meta = {
            **meta_in,
            "display_name": p.name,
            "dimensions": sample_dim,
            "created_at": int(time.time()),
        }
        self._manager().get_or_create_collection(vid, metadata=meta)
        logger.info("Created vector store %s (%s)", p.name, vid)
        public_meta = {k: v for k, v in meta.items() if k not in _META_EXCLUDE}
        return VectorStoreMetadata(
            id=vid, name=p.name, dimensions=sample_dim, metadata=public_meta, created_at=meta["created_at"]
        )

    async def get_vector_store(self, vid: str) -> VectorStoreMetadata:
        col = self._collection(vid)
        meta = col.metadata or {}
        return VectorStoreMetadata(
            id=vid,
            name=meta.get("display_name", vid),
            dimensions=meta.get("dimensions", 0),
            metadata={k: v for k, v in meta.items() if k not in _META_EXCLUDE},
            created_at=int(meta.get("created_at", time.time())),
        )

    async def update_vector_store(self, vid: str, p: VectorStoreUpdate) -> VectorStoreMetadata:
        col = self._collection(vid)
        meta = dict(col.metadata or {})
        if p.name:
            meta["display_name"] = p.name
        if p.metadata:
            meta.update(p.metadata)
        col.modify(metadata=meta)
        logger.info("Updated vector store %s", vid)
        return VectorStoreMetadata(
            id=vid,
            name=meta.get("display_name", vid),
            dimensions=meta.get("dimensions", 0),
            metadata={k: v for k, v in meta.items() if k not in _META_EXCLUDE},
            created_at=int(meta.get("created_at", time.time())),
        )

    async def delete_vector_store(self, vid: str) -> Dict[str, Any]:
        mgr = self._manager()
        if vid not in mgr.list_collections():
            raise VectorStoreNotFoundError(vid)
        mgr.delete_collection(vid)
        logger.info("Deleted vector store %s", vid)
        return {"id": vid, "object": "vector_store.deleted", "deleted": True}

    async def add_vectors(self, vid: str, batch: VectorBatch) -> Dict[str, Any]:
        col = self._collection(vid)
        dim = col.metadata.get("dimensions", 0)
        if dim:
            for v in batch.vectors:
                if len(v) != dim:
                    raise ValueError(f"Vector dim mismatch: expected {dim}, got {len(v)}")
        metas = batch.metadata or [{}] * len(batch.vectors)
        col.add(embeddings=batch.vectors, ids=batch.ids, metadatas=metas)
        logger.info("Added %d vectors to %s", len(batch.vectors), vid)
        return {"object": "list", "data": [{"id": i, "object": "vector"} for i in batch.ids]}

    async def delete_vectors(self, vid: str, ids: List[str]) -> Dict[str, Any]:
        self._collection(vid).delete(ids=ids)
        logger.info("Deleted %d vectors from %s", len(ids), vid)
        return {"object": "list", "data": [{"id": i, "object": "vector.deleted", "deleted": True} for i in ids]}

    async def _query_vector(self, q: Union[str, List[float]], model: str) -> Tuple[List[float], Dict]:
        if isinstance(q, list):
            return q, {"prompt_tokens": 0, "total_tokens": 0}
        res = generate_embeddings(q, model_name=model)
        return res["data"][0]["embedding"], res["usage"]

    async def search_vectors(self, vid: str, s: VectorSearch) -> VectorSearchResult:
        col = self._collection(vid)
        model = getattr(s, "embedding_model", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL
        q_vec, usage = await self._query_vector(s.query, model)
        res = col.query(
            query_embeddings=[q_vec],
            n_results=s.max_num_results,
            where=s.filters,
            include=["metadatas", "embeddings", "distances"],
        )
        out = []
        for i, vec_id in enumerate(res["ids"][0]):
            meta = (res.get("metadatas") or [[]])[0][i] or {}
            vec = Vector(id=vec_id, metadata=meta)
            if s.include_values and "embeddings" in res:
                vec.values = res["embeddings"][0][i]
            out.append(vec)
        return VectorSearchResult(data=out, usage=usage)
