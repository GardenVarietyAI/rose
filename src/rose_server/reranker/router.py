import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Request

from rose_server.reranker import service
from rose_server.schemas.reranker import RerankRequest, RerankResponse, RerankResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["reranking"])


@router.post("/rerank")
async def rerank(
    req: Request,
    request: RerankRequest = Body(...),
) -> RerankResponse:
    try:
        if not hasattr(req.app.state, "reranker_session") or not hasattr(req.app.state, "reranker_tokenizer"):
            raise HTTPException(status_code=500, detail="Reranker not initialized")

        session = req.app.state.reranker_session
        tokenizer = req.app.state.reranker_tokenizer

        # Score all documents
        scores = []
        for i, doc in enumerate(request.documents):
            relevance_score = service.score(request.query, doc, session, tokenizer)
            scores.append((i, relevance_score, doc))

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
            meta={},
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
