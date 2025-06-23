"""FastAPI router for evaluation endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from rose_server.evals.samples.store import count_eval_samples, get_eval_sample, list_eval_samples
from rose_server.runs.store import get_eval_run

router = APIRouter(prefix="/v1/evals/{eval_id}/runs", tags=["evals"])


@router.get("/{run_id}/samples")
async def samples(
    eval_id: str,
    run_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    only_failed: bool = Query(False, description="Only return failed samples"),
) -> Dict[str, Any]:
    """Get evaluation samples for a specific run."""

    run = await get_eval_run(run_id)
    if not run or run.eval_id != eval_id:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    samples = await list_eval_samples(eval_run_id=run_id, limit=limit, offset=offset, only_failed=only_failed)
    counts = await count_eval_samples(run_id)

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
async def get(
    eval_id: str,
    run_id: str,
    sample_id: str,
) -> Dict[str, Any]:
    """Get a specific evaluation sample."""

    run = await get_eval_run(run_id)
    if not run or run.eval_id != eval_id:
        raise HTTPException(status_code=404, detail="Evaluation run not found")

    sample = await get_eval_sample(sample_id)

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
