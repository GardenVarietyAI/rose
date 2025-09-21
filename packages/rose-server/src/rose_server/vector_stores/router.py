import logging
from typing import Any, Dict

from chonkie import TokenChunker
from fastapi import APIRouter, Body, HTTPException, Path, Request

from rose_server.config.settings import settings
from rose_server.embeddings.service import generate_embeddings, generate_query_embedding
from rose_server.schemas.vector_stores import (
    VectorSearch,
    VectorSearchChunk,
    VectorSearchResult,
    VectorSearchUsage,
    VectorStoreCreate,
    VectorStoreList,
    VectorStoreMetadata,
    VectorStoreUpdate,
)
from rose_server.vector_stores.files.router import router as files_router
from rose_server.vector_stores.files.service import decode_file_content
from rose_server.vector_stores.files.store import (
    FileNotFoundError as StoreFileNotFoundError,
    VectorStoreNotFoundError as StoreVectorStoreNotFoundError,
    get_uploaded_file,
    store_file_chunks_with_embeddings,
)
from rose_server.vector_stores.store import (
    create_vector_store,
    delete_vector_store,
    get_vector_store,
    list_vector_stores,
    search_vector_store,
    update_vector_store,
)

router = APIRouter(prefix="/v1/vector_stores")
logger = logging.getLogger(__name__)

router.include_router(files_router)

_INTERNAL_FIELDS = frozenset(["file_id", "filename", "total_chunks", "start_index", "end_index", "decode_errors"])


class VectorStoreNotFoundError(RuntimeError):
    pass


@router.get("")
async def index() -> VectorStoreList:
    try:
        stores = await list_vector_stores()
        return VectorStoreList(
            data=[
                VectorStoreMetadata(
                    id=store.id,
                    name=store.name,
                    dimensions=store.dimensions,
                    metadata=store.meta or {},
                    created_at=store.created_at,
                )
                for store in stores
            ]
        )
    except Exception as e:
        logger.error(f"Error listing vector stores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing vector stores: {str(e)}")


@router.post("")
async def create(req: Request, request: VectorStoreCreate = Body(...)) -> VectorStoreMetadata:
    try:
        vector_store = await create_vector_store(request.name, settings.embedding_dimensions)
        logger.info(f"Created vector store {request.name} ({vector_store.id})")

        # TODO: batch this operation
        if request.file_ids:
            if not req.app.state.embedding_model or not req.app.state.embedding_tokenizer:
                raise HTTPException(status_code=500, detail="Embedding model not initialized")

            for file_id in request.file_ids:
                try:
                    uploaded_file = await get_uploaded_file(file_id)
                    text_content, decode_errors = decode_file_content(uploaded_file.content, uploaded_file.filename)
                    chunker = TokenChunker(
                        chunk_size=settings.default_chunk_size,
                        chunk_overlap=settings.default_chunk_overlap,
                        tokenizer=req.app.state.embedding_tokenizer,
                    )
                    chunks = chunker.chunk(text_content)

                    if not chunks:
                        raise ValueError(f"No chunks generated from file {file_id}")

                    texts = [chunk.text for chunk in chunks]
                    embeddings, _ = await generate_embeddings(texts, req.app.state.embedding_model)

                    await store_file_chunks_with_embeddings(vector_store.id, file_id, chunks, embeddings, decode_errors)
                    logger.info(f"Added file {file_id} to vector store {request.name} ({vector_store.id})")
                except (StoreVectorStoreNotFoundError, StoreFileNotFoundError) as e:
                    raise HTTPException(status_code=404, detail=str(e))
                except Exception as e:
                    raise HTTPException(status_code=422, detail=f"Failed to process file {file_id}: {str(e)}")

        return VectorStoreMetadata(
            id=vector_store.id,
            name=vector_store.name,
            dimensions=vector_store.dimensions,
            metadata=vector_store.meta or {},
            created_at=vector_store.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")


@router.get("/{vector_store_id}")
async def get(vector_store_id: str = Path(..., description="The ID of the vector store")) -> VectorStoreMetadata:
    try:
        vector_store = await get_vector_store(vector_store_id)
        if not vector_store:
            raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

        return VectorStoreMetadata(
            id=vector_store.id,
            name=vector_store.name,
            dimensions=vector_store.dimensions,
            metadata=vector_store.meta or {},
            created_at=vector_store.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting vector store: {str(e)}")


@router.post("/{vector_store_id}")
async def update(vector_store_id: str = Path(...), request: VectorStoreUpdate = Body(...)) -> VectorStoreMetadata:
    try:
        vector_store = await update_vector_store(
            vector_store_id=vector_store_id,
            name=request.name,
            metadata=request.metadata,
        )

        if not vector_store:
            raise HTTPException(status_code=404, detail="VectorStore not found")

        logger.info(f"Updated vector store {vector_store_id}")

        return VectorStoreMetadata(
            id=vector_store_id,
            name=vector_store.name,
            dimensions=vector_store.dimensions,
            metadata=vector_store.meta or {},
            created_at=vector_store.created_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating vector store: {str(e)}")


@router.delete("/{vector_store_id}")
async def delete(
    vector_store_id: str = Path(..., description="The ID of the vector store"),  # noqa: ARG001
) -> Dict[str, Any]:
    success = await delete_vector_store(vector_store_id)
    if not success:
        raise HTTPException(status_code=404, detail="VectorStore not found")
    return {"id": vector_store_id, "object": "vector_store.deleted", "deleted": True}


@router.post("/{vector_store_id}/search")
async def search_store(
    req: Request, vector_store_id: str = Path(...), request: VectorSearch = Body(...)
) -> VectorSearchResult:
    vector_store = await get_vector_store(vector_store_id)
    if not vector_store:
        raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

    if not req.app.state.embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")

    query_embedding = await generate_query_embedding(request.query, req.app.state.embedding_model)
    documents = await search_vector_store(vector_store_id, query_embedding, request.max_num_results)

    search_chunks = []
    for doc in documents:
        meta = doc.document.meta or {}
        attributes = {k: v for k, v in meta.items() if k not in _INTERNAL_FIELDS}

        chunk = VectorSearchChunk(
            file_id=meta["file_id"],
            filename=meta["filename"],
            similarity=doc.score,
            attributes=attributes,
            content=[{"type": "text", "text": doc.document.content}],
        )
        search_chunks.append(chunk)

    return VectorSearchResult(
        search_query=request.query,
        data=search_chunks,
        first_id=None,
        last_id=None,
        has_more=False,
        next_page=None,
        usage=VectorSearchUsage(
            prompt_tokens=0,
            total_tokens=0,
        ),
    )
