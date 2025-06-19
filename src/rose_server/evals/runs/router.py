"""FastAPI router for evaluation endpoints."""

import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from rose_server.queues.facade import EvalJob
from rose_server.schemas.evals import (
    DataSourceConfig,
    EvalRunCreateRequest,
    EvalRunResponse,
)

from ..samples.store import EvalSampleStore
from ..store import EvalStore
from .store import EvalRunStore

router = APIRouter(prefix="/v1/evals/{eval_id}/runs", tags=["evals"])


@router.post("", response_model=EvalRunResponse)
async def create_eval_run(
    eval_id: str,
    request: EvalRunCreateRequest,
) -> EvalRunResponse:
    """Create an evaluation run"""

    eval_store = EvalStore()
    eval_def = await eval_store.get(eval_id)

    if not eval_def:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    run_id = f"evalrun_{uuid.uuid4().hex}"
    model = request.data_source.model

    run_store = EvalRunStore()
    eval_run = await run_store.create(
        id=run_id,
        eval_id=eval_id,
        name=request.name,
        model=model,
        data_source=request.data_source.dict(),
    )

    metadata: Dict[str, Any] = {
        "data_source": request.data_source.dict(),
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
async def list_eval_runs(
    eval_id: str,
    limit: int = 20,
) -> List[EvalRunResponse]:
    """List runs for a specific evaluation."""
    eval_store = EvalStore()
    eval_def = await eval_store.get(eval_id)

    if not eval_def:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    run_store = EvalRunStore()
    runs = await run_store.list(eval_id=eval_id, limit=limit)

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
async def get_eval_run(
    eval_id: str,
    run_id: str,
) -> EvalRunResponse:
    """Get a specific evaluation run."""
    run_store = EvalRunStore()
    run = await run_store.get(run_id)
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


@router.get("/{run_id}/samples")
async def list_eval_samples(
    eval_id: str,
    run_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    only_failed: bool = Query(False, description="Only return failed samples"),
) -> Dict[str, Any]:
    """Get evaluation samples for a specific run."""
    run_store = EvalRunStore()
    run = await run_store.get(run_id)
    if not run or run.eval_id != eval_id:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    sample_store = EvalSampleStore()
    samples = await sample_store.list(eval_run_id=run_id, limit=limit, offset=offset, only_failed=only_failed)
    counts = await sample_store.count(run_id)
    return {
        "object": "list",
        "data": [
            {
                "id": sample.id,
                "object": "eval.sample",
                "eval_run_id": sample.eval_run_id,
                "sample_index": sample.sample_index,
                "input": sample.input,
                "expected_output": sample.ideal,
                "actual_output": sample.completion,
                "score": sample.score,
                "passed": sample.passed,
                "response_time": sample.response_time,
                "tokens_used": sample.tokens_used,
                "metadata": sample.meta,
                "created_at": sample.created_at,
            }
            for sample in samples
        ],
        "total": counts["total"],
        "has_more": offset + limit < counts["total"],
        "counts": counts,
    }


@router.get("/{run_id}/samples/{sample_id}")
async def get_eval_sample(
    eval_id: str,
    run_id: str,
    sample_id: str,
) -> Dict[str, Any]:
    """Get a specific evaluation sample."""
    run_store = EvalRunStore()
    run = await run_store.get(run_id)
    if not run or run.eval_id != eval_id:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    sample_store = EvalSampleStore()
    sample = await sample_store.get(sample_id)
    if not sample or sample.eval_run_id != run_id:
        raise HTTPException(status_code=404, detail="Evaluation sample not found")
    return {
        "id": sample.id,
        "object": "eval.sample",
        "eval_run_id": sample.eval_run_id,
        "sample_index": sample.sample_index,
        "input": sample.input,
        "expected_output": sample.ideal,
        "actual_output": sample.completion,
        "score": sample.score,
        "passed": sample.passed,
        "response_time": sample.response_time,
        "tokens_used": sample.tokens_used,
        "metadata": sample.meta,
        "created_at": sample.created_at,
    }
