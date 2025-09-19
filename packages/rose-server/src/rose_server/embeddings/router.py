"""Router module for embeddings API endpoints."""

import asyncio

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from rose_server.embeddings.service import compute_embeddings_with_tokens
from rose_server.schemas.embeddings import EmbeddingRequest, EmbeddingResponse

router = APIRouter(prefix="/v1/embeddings")


@router.post("", response_model=EmbeddingResponse)
async def create_embeddings(req: Request, request: EmbeddingRequest) -> EmbeddingResponse:
    if not req.app.state.embedding_model or not req.app.state.embedding_tokenizer:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")

    try:
        embeddings, total_tokens = await asyncio.to_thread(
            compute_embeddings_with_tokens,
            request.input if isinstance(request.input, list) else [request.input],
            req.app.state.embedding_model,
            req.app.state.embedding_tokenizer,
        )

        return EmbeddingResponse(
            object="list",
            data=[
                {
                    "object": "embedding",
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding),
                    "index": i,
                }
                for i, embedding in enumerate(embeddings)
            ],
            model=request.model,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
