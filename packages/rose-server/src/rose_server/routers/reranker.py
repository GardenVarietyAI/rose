import logging
import time
from typing import Any, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["reranking"])


class RerankRequest(BaseModel):
    model: str = "Qwen/Qwen3-Embedding-0.6B-GGUF"
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: bool = True


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@router.post("/rerank")
async def rerank(request: Request, body: RerankRequest) -> dict[str, Any]:
    try:
        embed_model = request.app.state.embed_model

        query_response = embed_model.create_embedding(input=body.query)
        query_embedding = np.array(query_response["data"][0]["embedding"])

        doc_responses = [embed_model.create_embedding(input=doc) for doc in body.documents]
        doc_embeddings = [np.array(resp["data"][0]["embedding"]) for resp in doc_responses]

        scores_with_indices = [
            (i, cosine_similarity(query_embedding, doc_emb), doc)
            for i, (doc_emb, doc) in enumerate(zip(doc_embeddings, body.documents))
        ]

        scores_with_indices.sort(key=lambda x: x[1], reverse=True)

        if body.top_n:
            scores_with_indices = scores_with_indices[: body.top_n]

        results = [
            {
                "index": index,
                "relevance_score": score,
                "document": doc if body.return_documents else None,
            }
            for index, score, doc in scores_with_indices
        ]

        return {"id": f"rerank-{int(time.time() * 1000)}", "results": results, "meta": {}}

    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
