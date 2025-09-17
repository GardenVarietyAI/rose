"""Router module for embeddings API endpoints."""

import asyncio
from typing import Any, List, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from rose_server.schemas.embeddings import EmbeddingRequest, EmbeddingResponse

router = APIRouter(prefix="/v1/embeddings")


def compute_embeddings(
    texts: List[str], model: Any, tokenizer: Any, batch_size: int = 32
) -> Tuple[List[np.ndarray], int]:
    """Pure function to compute embeddings from texts."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = list(model.embed(batch))
        all_embeddings.extend(batch_embeddings)

    total_tokens = sum(len(tokenizer.encode(text).ids) for text in texts)
    return all_embeddings, total_tokens


@router.post("", response_model=EmbeddingResponse)
async def create_embeddings(req: Request, request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings.

    Args:
        request: The embeddings request containing input texts and model
    Returns:
        JSON response in OpenAI format with embeddings
    """
    if not req.app.state.embedding_model or not req.app.state.embedding_tokenizer:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")

    try:
        texts = request.input if isinstance(request.input, list) else [request.input]

        embeddings, total_tokens = await asyncio.to_thread(
            compute_embeddings, texts, req.app.state.embedding_model, req.app.state.embedding_tokenizer
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
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": {"message": str(e), "type": "invalid_request_error"}})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"error": {"message": f"An error occurred: {str(e)}", "type": "server_error"}}
        )
