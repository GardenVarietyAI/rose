"""Router module for embeddings API endpoints."""

from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from rose_server.schemas.embeddings import EmbeddingRequest, EmbeddingResponse

router = APIRouter(prefix="/v1/embeddings")


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
        texts: List[str] = request.input if isinstance(request.input, list) else [request.input]

        batch_size = 32
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = list(req.app.state.embedding_model.embed(batch))
            all_embeddings.extend(batch_embeddings)

        total_tokens = 0
        for text in texts:
            tokens = req.app.state.embedding_tokenizer.encode(text)
            total_tokens += len(tokens.ids)

        return EmbeddingResponse(
            object="list",
            data=[
                {
                    "object": "embedding",
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding),
                    "index": i,
                }
                for i, embedding in enumerate(all_embeddings)
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
