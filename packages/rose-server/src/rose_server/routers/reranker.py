import logging

from fastapi import APIRouter, Body, HTTPException, Request
from rose_server.schemas.reranker import RerankRequest, RerankResponse, RerankResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/rerank", tags=["reranking"])


@router.post("")
async def rerank(
    req: Request,
    request: RerankRequest = Body(...),
) -> RerankResponse:
    try:
        if not hasattr(req.app.state, "reranker_model") or not req.app.state.reranker_model:
            raise HTTPException(status_code=500, detail="Reranker not initialized")

        model = req.app.state.reranker_model

        scores_with_indices = await model.score_batch(
            [request.query] * len(request.documents),
            request.documents,
        )

        # Combine with indices and documents
        scores = [(i, score, doc) for i, (score, doc) in enumerate(zip(scores_with_indices, request.documents))]

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        if request.top_n:
            scores = scores[: request.top_n]

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
