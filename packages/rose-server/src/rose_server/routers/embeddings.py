import logging
from typing import List, Union

import llama_cpp
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["embeddings"])


class CreateEmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "Qwen/Qwen3-Embedding-0.6B-GGUF"


@router.post("/embeddings")
async def create_embedding(request: Request, body: CreateEmbeddingRequest) -> llama_cpp.CreateEmbeddingResponse:
    llama: llama_cpp.Llama = request.app.state.embed_model

    try:
        response: llama_cpp.CreateEmbeddingResponse = llama.create_embedding(input=body.input)
        return response
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
