import json
import logging
import time
from typing import Any, Dict, List

import numpy as np
from chonkie import TokenChunker
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Path, Request
from rose_server.entities.files import UploadedFile
from rose_server.entities.vector_stores import Document, DocumentSearchResult, VectorStore, VectorStoreFile
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
from rose_server.services.vector_store_documents import prepare_documents_and_embeddings
from rose_server.services.vector_store_files import decode_file_content
from sqlalchemy import (
    text,
    update as sql_update,
)
from sqlalchemy.dialects.sqlite import insert
from sqlmodel import (
    col,
    delete as sql_delete,
    func,
    select,
)

router = APIRouter(prefix="/v1/vector_stores")
logger = logging.getLogger(__name__)

_INTERNAL_FIELDS = frozenset(["file_id", "filename", "total_chunks", "start_index", "end_index", "decode_errors"])


@router.get("")
async def index(req: Request) -> VectorStoreList:
    try:
        async with req.app.state.get_db_session(read_only=True) as session:
            result = await session.execute(select(VectorStore))
            stores = [row[0] for row in result.fetchall()]
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
        vector_store = VectorStore(
            object="vector_store",
            name=request.name,
            dimensions=req.app.state.settings.embedding_dimensions,
            last_used_at=None,
        )

        async with req.app.state.get_db_session() as session:
            session.add(vector_store)
            await session.commit()
        logger.info(f"Created vector store {request.name} ({vector_store.id})")

        if request.file_ids:
            if not req.app.state.embedding_model or not req.app.state.embedding_tokenizer:
                raise HTTPException(status_code=500, detail="Embedding model not initialized")

            for file_id in request.file_ids:
                try:
                    async with req.app.state.get_db_session(read_only=True) as file_session:
                        uploaded_file = await file_session.get(UploadedFile, file_id)
                        if not uploaded_file:
                            raise HTTPException(status_code=404, detail=f"Uploaded file {file_id} not found")
                    text_content, decode_errors = decode_file_content(uploaded_file.content, uploaded_file.filename)
                    chunker = TokenChunker(
                        chunk_size=req.app.state.settings.default_chunk_size,
                        chunk_overlap=req.app.state.settings.default_chunk_overlap,
                        tokenizer=req.app.state.embedding_tokenizer,
                    )
                    chunks = chunker.chunk(text_content)

                    if not chunks:
                        raise ValueError(f"No chunks generated from file {file_id}")

                    texts = [chunk.text for chunk in chunks]
                    embeddings, _ = await req.app.state.embedding_model.encode_batch(texts)

                    async with req.app.state.get_db_session() as store_session:
                        # Upsert vector store file record
                        await store_session.execute(
                            insert(VectorStoreFile)
                            .values(vector_store_id=vector_store.id, file_id=file_id)
                            .on_conflict_do_nothing(
                                index_elements=[VectorStoreFile.vector_store_id, VectorStoreFile.file_id]
                            )
                        )

                        # Get the vector store file record
                        vsf = await store_session.scalar(
                            select(VectorStoreFile).where(
                                VectorStoreFile.vector_store_id == vector_store.id,
                                VectorStoreFile.file_id == file_id,
                            )
                        )

                        if vsf and vsf.status == "in_progress":
                            documents, embedding_data, created_at = prepare_documents_and_embeddings(
                                uploaded_file, vector_store.id, chunks, embeddings, decode_errors
                            )

                            await store_session.execute(
                                sql_delete(Document).where(col(Document.id).like(f"{file_id}#%"))
                            )

                            for doc in documents:
                                store_session.add(doc)

                            # Batch insert embeddings using sqlite-vec
                            await store_session.execute(
                                text(
                                    "INSERT OR REPLACE INTO vec0 (document_id, embedding) VALUES (:doc_id, :embedding)"
                                ),
                                embedding_data,
                            )

                            await store_session.commit()

                            if vsf:
                                vsf.status = "completed"
                                vsf.last_error = None
                            await store_session.commit()

                            await store_session.execute(
                                sql_update(VectorStore)
                                .where(VectorStore.id == vector_store.id)  # type: ignore[arg-type]
                                .values(last_used_at=created_at)
                            )
                            await store_session.commit()
                    logger.info(f"Added file {file_id} to vector store {request.name} ({vector_store.id})")
                except HTTPException:
                    raise
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
async def get(
    req: Request, vector_store_id: str = Path(..., description="The ID of the vector store")
) -> VectorStoreMetadata:
    try:
        async with req.app.state.get_db_session(read_only=True) as session:
            vector_store = await session.get(VectorStore, vector_store_id)
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
async def update(
    req: Request, vector_store_id: str = Path(...), request: VectorStoreUpdate = Body(...)
) -> VectorStoreMetadata:
    try:
        async with req.app.state.get_db_session() as session:
            vector_store = await session.get(VectorStore, vector_store_id)

            if not vector_store:
                raise HTTPException(status_code=404, detail="VectorStore not found")

            if request.name is not None:
                vector_store.name = request.name

            if request.metadata is not None:
                base = (vector_store.meta or {}).copy()
                base.update(request.metadata)
                vector_store.meta = base  # reassign so SQLAlchemy tracks the change

            await session.flush()
            await session.refresh(vector_store)

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
    req: Request,
    vector_store_id: str = Path(..., description="The ID of the vector store"),
) -> Dict[str, Any]:
    async with req.app.state.get_db_session() as session:
        vector_store = await session.get(VectorStore, vector_store_id)

        if not vector_store:
            raise HTTPException(status_code=404, detail="VectorStore not found")

        file_count_result = await session.execute(
            select(func.count()).where(VectorStoreFile.vector_store_id == vector_store_id)
        )

        file_count = file_count_result.scalar()
        await session.execute(sql_delete(VectorStoreFile).where(VectorStoreFile.vector_store_id == vector_store_id))  # type: ignore[arg-type]
        await session.delete(vector_store)
        await session.commit()

        logger.info(f"Deleted vector store: {vector_store_id} and {file_count} files")
    return {"id": vector_store_id, "object": "vector_store.deleted", "deleted": True}


@router.post("/{vector_store_id}/search")
async def search_store(
    req: Request,
    background_tasks: BackgroundTasks,
    vector_store_id: str = Path(...),
    request: VectorSearch = Body(...),
) -> VectorSearchResult:
    async with req.app.state.get_db_session(read_only=True) as session:
        vector_store = await session.get(VectorStore, vector_store_id)
    if not vector_store:
        raise HTTPException(status_code=404, detail=f"Vector store {vector_store_id} not found")

    if not req.app.state.embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")

    query_embedding = await req.app.state.embedding_model.encode(request.query)
    query_tokens = len(req.app.state.embedding_tokenizer.encode(request.query))

    async with req.app.state.get_db_session(read_only=True) as search_session:
        vector_store_check = await search_session.get(VectorStore, vector_store_id)
        if not vector_store_check:
            raise ValueError(f"Vector store {vector_store_id} not found")

        got_dim = len(query_embedding)
        if got_dim != vector_store_check.dimensions:
            raise ValueError(
                f"Query vector dimension mismatch: got {got_dim}, expected {vector_store_check.dimensions}",
            )

        query_blob = np.array(query_embedding, dtype=np.float32).tobytes()
        max_results = max(1, min(100, request.max_num_results))

        # Vector search using cosine distance with sqlite-vec
        result = await search_session.execute(
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

        documents: List[DocumentSearchResult] = []
        for row in result.fetchall():
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
            documents.append(DocumentSearchResult(document=doc, score=similarity))

    async def update_last_used() -> None:
        async with req.app.state.get_db_session() as session:
            vs = await session.get(VectorStore, vector_store_id)
            if vs:
                vs.last_used_at = int(time.time())
                session.add(vs)
                await session.commit()

    background_tasks.add_task(update_last_used)

    search_chunks = []
    for search_result in documents:
        meta = search_result.document.meta or {}
        attributes = {k: v for k, v in meta.items() if k not in _INTERNAL_FIELDS}

        chunk = VectorSearchChunk(
            file_id=meta["file_id"],
            filename=meta["filename"],
            score=search_result.score,
            attributes=attributes,
            content=[{"type": "text", "text": search_result.document.content}],
        )
        search_chunks.append(chunk)

    return VectorSearchResult(
        search_query=request.query,
        data=search_chunks,
        has_more=False,
        next_page=None,
        usage=VectorSearchUsage(
            prompt_tokens=query_tokens,
            total_tokens=query_tokens,
        ),
    )
