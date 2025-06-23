"""FastAPI router for evaluation endpoints."""

import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from rose_server.evals.runs.store import create_eval_run, get_eval_run, list_eval_runs
from rose_server.evals.store import get_eval
from rose_server.queues.facade import EvalJob
from rose_server.schemas.evals import (
    DataSourceConfig,
    EvalRunCreateRequest,
    EvalRunResponse,
)

router = APIRouter(prefix="/v1/evals/{eval_id}/runs", tags=["evals"])


@router.post("", response_model=EvalRunResponse)
async def create(eval_id: str, request: EvalRunCreateRequest) -> EvalRunResponse:
    """Create an evaluation run"""

    eval_def = await get_eval(eval_id)

    if not eval_def:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    run_id = f"evalrun_{uuid.uuid4().hex}"
    model = request.data_source.model

    eval_run = await create_eval_run(
        id=run_id,
        eval_id=eval_id,
        name=request.name,
        model=model,
        data_source=request.data_source.model_dump(),
    )

    metadata: Dict[str, Any] = {
        "data_source": request.data_source.model_dump(),
        "eval_def_id": eval_id,
        "run_name": request.name,
        "eval_data_source_config": eval_def.data_source_config,
        "eval_testing_criteria": eval_def.testing_criteria,
    }

    if request.data_source.max_samples is not None:
        metadata["max_samples"] = request.data_source.max_samples

    await EvalJob.dispatch(
        eval_id=run_id,
        model=model,
        eval=eval_def.name,
        metadata=metadata,
    )

    return EvalRunResponse(
        id=run_id,
        object="eval.run",
        eval_id=eval_id,
        name=request.name,
        model=model,
        status="queued",
        created_at=eval_run.created_at,
        data_source=request.data_source,
    )


@router.get("", response_model=List[EvalRunResponse])
async def runs(
    eval_id: str,
    limit: int = 20,
) -> List[EvalRunResponse]:
    """List runs for a specific evaluation."""
    eval_def = await get_eval(eval_id)

    if not eval_def:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    runs = await list_eval_runs(eval_id=eval_id, limit=limit)

    return [
        EvalRunResponse(
            id=run.id,
            object="eval.run",
            eval_id=run.eval_id,
            name=run.name,
            model=run.model,
            status=run.status,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
            data_source=DataSourceConfig(**run.data_source) if run.data_source else None,
            result_counts=run.result_counts,
            report_url=run.report_url,
            results=run.results,
            error=run.error_message,
            metadata=run.meta,
        )
        for run in runs
    ]


@router.get("/{run_id}", response_model=EvalRunResponse)
async def get(
    eval_id: str,
    run_id: str,
) -> EvalRunResponse:
    """Get a specific evaluation run."""
    run = await get_eval_run(run_id)
    if not run or run.eval_id != eval_id:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    return EvalRunResponse(
        id=run.id,
        object="eval.run",
        eval_id=run.eval_id,
        name=run.name,
        model=run.model,
        status=run.status,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        data_source=DataSourceConfig(**run.data_source) if run.data_source else None,
        result_counts=run.result_counts,
        report_url=run.report_url,
        results=run.results,
        error=run.error_message,
        metadata=run.meta,
    )
