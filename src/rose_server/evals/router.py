"""FastAPI router for evaluation endpoints"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas.evals import EvalCreateRequest, EvalDeleteResponse, EvalListResponse, EvalResponse
from .store import EvalStore

router = APIRouter(prefix="/v1/evals", tags=["evals"])


@router.post("", response_model=EvalResponse)
async def create_evaluation(request: EvalCreateRequest) -> EvalResponse:
    """Create a new evaluation"""
    store = EvalStore()
    eval_id = f"eval-{uuid.uuid4().hex[:16]}"

    eval = await store.create(
        id=eval_id,
        name=request.name,
        data_source_config=request.data_source_config,
        testing_criteria=request.testing_criteria,
        metadata=request.metadata,
    )

    return EvalResponse(**eval.model_dump())


@router.get("/{eval_id}", response_model=EvalResponse)
async def get_evaluation(eval_id: str) -> EvalResponse:
    """Get evaluation."""
    store = EvalStore()
    eval = await store.get(eval_id)

    if not eval:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return EvalResponse(**eval.model_dump())


@router.delete("/{eval_id}", response_model=EvalDeleteResponse)
async def delete_evaluation(eval_id: str) -> EvalDeleteResponse:
    """Delete an evaluation."""
    store = EvalStore()
    deleted = await store.delete(eval_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return EvalDeleteResponse(id=eval_id, object="eval", deleted=True)


@router.get("", response_model=EvalListResponse)
async def list_evaluations(
    limit: int = Query(20, description="Number of evaluations to return"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
) -> EvalListResponse:
    """List evaluations"""
    store = EvalStore()
    evals = await store.list(limit=limit)

    return EvalListResponse(
        object="list",
        data=[EvalResponse(**eval.model_dump()) for eval in evals],
        has_more=False,
    )
