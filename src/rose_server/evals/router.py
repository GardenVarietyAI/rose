"""FastAPI router for evaluation endpoints"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas.evals import EvalCreateRequest, EvalDeleteResponse, EvalListResponse, EvalResponse
from .store import create_eval, delete_eval, get_eval, list_evals

router = APIRouter(prefix="/v1/evals", tags=["evals"])


@router.post("", response_model=EvalResponse)
async def create_evaluation(request: EvalCreateRequest) -> EvalResponse:
    """Create a new evaluation"""
    eval_id = f"eval-{uuid.uuid4().hex[:16]}"

    eval = await create_eval(
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
    eval = await get_eval(eval_id)

    if not eval:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return EvalResponse(**eval.model_dump())


@router.delete("/{eval_id}", response_model=EvalDeleteResponse)
async def delete_evaluation(eval_id: str) -> EvalDeleteResponse:
    """Delete an evaluation."""
    deleted = await delete_eval(eval_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return EvalDeleteResponse(id=eval_id, object="eval", deleted=True)


@router.get("", response_model=EvalListResponse)
async def list_evaluations(
    limit: int = Query(20, description="Number of evaluations to return"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
) -> EvalListResponse:
    """List evaluations"""
    evals = await list_evals(limit=limit)

    return EvalListResponse(
        object="list",
        data=[EvalResponse(**eval.model_dump()) for eval in evals],
        has_more=False,
    )
