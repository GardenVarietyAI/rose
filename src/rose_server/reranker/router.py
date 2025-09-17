import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field

from rose_server.reranker.service import RerankerService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["reranking"])


class RerankRequest(BaseModel):
    model: str = Field(default="Qwen3-Reranker-0.6B-ONNX")
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_n: Optional[int] = Field(default=None, description="Number of results to return")
    return_documents: bool = Field(default=True, description="Include documents in response")
    max_chunks_per_doc: Optional[int] = Field(default=None, description="Max chunks per document")


class RerankResult(BaseModel):
    index: int = Field(..., description="Original document index")
    relevance_score: float = Field(..., description="Relevance score 0-1")
    document: Optional[str] = Field(default=None, description="Original document text")


class RerankResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"rerank-{int(time.time() * 1000)}")
    results: List[RerankResult] = Field(..., description="Reranked results")
    meta: Dict[str, Any] = Field(default_factory=dict)


@router.post("/rerank")
async def rerank(
    req: Request,
    request: RerankRequest = Body(...),
) -> RerankResponse:
    try:
        # Get reranker from app state (initialized at startup)
        if not hasattr(req.app.state, "reranker"):
            # Initialize on first use if needed
            req.app.state.reranker = RerankerService("data/models/Qwen3-Reranker-0.6B-ONNX")
            logger.info("Initialized reranker service")

        reranker = req.app.state.reranker

        # Score all documents
        scores = []
        for i, doc in enumerate(request.documents):
            score = reranker.score(request.query, doc)
            scores.append((i, score, doc))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_n if specified
        if request.top_n:
            scores = scores[: request.top_n]

        # Format results
        results = []
        for index, score, doc in scores:
            result = RerankResult(
                index=index,
                relevance_score=score,
                document=doc if request.return_documents else None,
            )
            results.append(result)

        return RerankResponse(
            results=results,
            meta={
                "api_version": {"version": "1.0"},
                "billed_units": {"search_units": len(request.documents)},
            },
        )

    except Exception as e:
        logger.error(f"Rerank error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rerank/models")
async def list_models() -> Dict[str, Any]:
    return {
        "models": [
            {
                "name": "Qwen3-Reranker-0.6B-ONNX",
                "endpoints": ["rerank"],
                "finetuned": False,
                "context_length": 32768,
                "default_endpoints": ["rerank"],
            }
        ]
    }
