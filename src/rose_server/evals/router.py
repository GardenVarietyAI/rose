"""FastAPI router for evaluation endpoints."""

import json
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..queues.facade import EvalJob
from ..schemas.evals import (
    DataSourceSchema,
    EvalDeleteResponse,
    EvalListResponse,
    EvalObject,
    EvalRunCreateRequest,
    EvalRunResponse,
    StoredCompletionsDataSourceConfig,
    StringCheckGrader,
    TextSimilarityGrader,
)
from .store import EvalStore

router = APIRouter(prefix="/v1/evals", tags=["evals"])
EVAL_CRITERIA_MAP = {
    "gsm8k": lambda: [
        StringCheckGrader(
            name="exact_match",
            id=f"exact_match-{uuid.uuid4()}",
            type="string_check",
            input="{{sample.output}}",
            reference="{{item.expected}}",
            operation="eq",
        )
    ],
    "humaneval": lambda: [
        StringCheckGrader(
            name="code_exact_match",
            id=f"code_exact_match-{uuid.uuid4()}",
            type="string_check",
            input="{{sample.output}}",
            reference="{{item.expected}}",
            operation="eq",
        )
    ],
    "mmlu": lambda: [
        StringCheckGrader(
            name="multiple_choice",
            id=f"multiple_choice-{uuid.uuid4()}",
            type="string_check",
            input="{{sample.output}}",
            reference="{{item.answer}}",
            operation="eq",
        )
    ],
    "default": lambda: [
        TextSimilarityGrader(
            name="f1_score",
            id=f"f1_score-{uuid.uuid4()}",
            type="text_similarity",
            input="{{sample.output}}",
            reference="{{item.expected}}",
            metric="f1",
            pass_threshold=0.8,
        )
    ],
}


def _get_testing_criteria(eval_name: str) -> List:
    """Get default testing criteria based on eval name."""
    criteria_fn = EVAL_CRITERIA_MAP.get(eval_name.lower(), EVAL_CRITERIA_MAP["default"])
    return criteria_fn()


def _get_data_source_schema(eval_name: str) -> DataSourceSchema:
    """Get default data source schema based on eval name."""
    return DataSourceSchema(
        type="object",
        properties={
            "item": {"type": "object", "properties": {"input": {"type": "string"}, "expected": {"type": "string"}}},
            "sample": {"type": "object", "properties": {"output": {"type": "string"}}},
        },
        required=["item", "sample"],
    )


def _reconstruct_eval_object(eval) -> EvalObject:
    """Reconstruct EvalObject from database entity."""
    data_source_config = StoredCompletionsDataSourceConfig(**eval.data_source_config)
    testing_criteria = []
    for criterion in eval.testing_criteria:
        if criterion.get("type") == "string_check":
            testing_criteria.append(StringCheckGrader(**criterion))
        elif criterion.get("type") == "text_similarity":
            testing_criteria.append(TextSimilarityGrader(**criterion))
    return EvalObject(
        id=eval.id,
        object="eval",
        name=eval.name,
        data_source_config=data_source_config,
        testing_criteria=testing_criteria,
        created_at=eval.created_at,
        metadata=eval.meta or {},
    )


@router.post("", response_model=EvalObject)
async def create_evaluation(
    data_source_config: Dict[str, Any],
    testing_criteria: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> EvalObject:
    """Create a new evaluation definition - OpenAI SDK compatible."""
    store = EvalStore()
    eval_id = f"eval_{uuid.uuid4().hex}"
    if not name and metadata and "name" in metadata:
        name = metadata["name"]
    elif not name:
        name = f"eval_{eval_id[:8]}"
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Incoming data_source_config: {data_source_config}")
    logger.info(f"Metadata: {data_source_config.get('metadata')}")
    processed_data_source = StoredCompletionsDataSourceConfig(
        type=data_source_config.get("type", "stored_completions"),
        metadata=data_source_config.get("metadata") or {},
        data_schema=_get_data_source_schema(name),
    )
    processed_criteria = []
    for criterion in testing_criteria:
        if criterion.get("type") == "string_check":
            processed = {
                "type": "string_check",
                "name": criterion.get("name", "exact_match"),
                "id": criterion.get("id", f"{criterion.get('name', 'exact_match')}-{uuid.uuid4()}"),
                "input": criterion.get("input", "{{sample.output}}"),
                "reference": criterion.get("reference", "{{item.expected}}"),
                "operation": criterion.get("operation", "eq"),
            }
            processed_criteria.append(processed)
        elif criterion.get("type") == "text_similarity":
            processed = {
                "type": "text_similarity",
                "name": criterion.get("name", "similarity"),
                "id": criterion.get("id", f"{criterion.get('name', 'similarity')}-{uuid.uuid4()}"),
                "input": criterion.get("input", "{{sample.output}}"),
                "reference": criterion.get("reference", "{{item.expected}}"),
                "threshold": criterion.get("threshold", 0.8),
            }
            processed_criteria.append(processed)
        else:
            processed_criteria.append(criterion)
    await store.create_eval(
        id=eval_id,
        name=name,
        data_source_config=processed_data_source.model_dump(),
        testing_criteria=processed_criteria,
        metadata=metadata,
    )
    return _reconstruct_eval_object(await store.get_eval(eval_id))


@router.get("/{eval_id}", response_model=EvalObject)
async def get_evaluation(
    eval_id: str,
) -> EvalObject:
    """Get evaluation definition."""
    store = EvalStore()
    eval = await store.get_eval(eval_id)
    if not eval:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _reconstruct_eval_object(eval)


@router.delete("/{eval_id}", response_model=EvalDeleteResponse)
async def delete_evaluation(
    eval_id: str,
) -> EvalDeleteResponse:
    """Delete an evaluation."""
    store = EvalStore()
    deleted = await store.delete_eval(eval_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return EvalDeleteResponse(id=eval_id, object="eval", deleted=True)


@router.get("", response_model=EvalListResponse)
async def list_evaluations(
    limit: int = Query(20, description="Number of evaluations to return"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
    order: Optional[str] = Query("desc", description="Sort order (asc/desc)"),
) -> EvalListResponse:
    """List evaluation definitions - OpenAI SDK compatible."""
    store = EvalStore()
    evals = await store.list_evals(limit=limit)
    eval_objects = [_reconstruct_eval_object(eval) for eval in evals]
    return EvalListResponse(
        object="list",
        data=eval_objects,
        has_more=False,
        first_id=eval_objects[0].id if eval_objects else None,
        last_id=eval_objects[-1].id if eval_objects else None,
    )


@router.post("/{eval_id}/runs", response_model=EvalRunResponse)
async def create_eval_run(
    eval_id: str,
    request: EvalRunCreateRequest,
) -> EvalRunResponse:
    """Create an evaluation run - OpenAI compatible."""
    store = EvalStore()
    eval_def = await store.get_eval(eval_id)
    if not eval_def:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    run_id = f"evalrun_{uuid.uuid4().hex}"
    model = request.data_source.model
    eval_run = await store.create_eval_run(
        id=run_id,
        eval_id=eval_id,
        name=request.name,
        model=model,
        data_source=request.data_source.dict(),
    )
    metadata = {"data_source": request.data_source.dict(), "eval_def_id": eval_id, "run_name": request.name}
    if request.data_source.max_samples is not None:
        metadata["max_samples"] = request.data_source.max_samples
    if eval_def.data_source_config:
        eval_metadata = eval_def.data_source_config.get("metadata", {})
        if eval_metadata.get("source") == "inline":
            content_str = eval_metadata.get("content", "[]")
            metadata["inline_content"] = json.loads(content_str) if isinstance(content_str, str) else content_str
    job_id = await EvalJob.dispatch(
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


@router.get("/{eval_id}/runs", response_model=List[EvalRunResponse])
async def list_eval_runs(
    eval_id: str,
    limit: int = 20,
) -> List[EvalRunResponse]:
    """List runs for a specific evaluation."""
    store = EvalStore()
    eval_def = await store.get_eval(eval_id)
    if not eval_def:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    runs = await store.list_eval_runs(eval_id=eval_id, limit=limit)
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
            data_source=run.data_source,
            results=run.results,
            error=run.error_message,
        )
        for run in runs
    ]


@router.get("/{eval_id}/runs/{run_id}", response_model=EvalRunResponse)
async def get_eval_run(
    eval_id: str,
    run_id: str,
) -> EvalRunResponse:
    """Get a specific evaluation run."""
    store = EvalStore()
    run = await store.get_eval_run(run_id)
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
        data_source=run.data_source,
        results=run.results,
        error=run.error_message,
    )


@router.get("/{eval_id}/runs/{run_id}/samples")
async def list_eval_samples(
    eval_id: str,
    run_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    only_failed: bool = Query(False, description="Only return failed samples"),
) -> Dict[str, Any]:
    """Get evaluation samples for a specific run."""
    store = EvalStore()
    run = await store.get_eval_run(run_id)
    if not run or run.eval_id != eval_id:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    samples = await store.get_eval_samples(eval_run_id=run_id, limit=limit, offset=offset, only_failed=only_failed)
    counts = await store.count_eval_samples(run_id)
    return {
        "object": "list",
        "data": [
            {
                "id": sample.id,
                "object": "eval.sample",
                "eval_run_id": sample.eval_run_id,
                "sample_index": sample.sample_index,
                "input": sample.input,
                "expected_output": sample.expected_output,
                "actual_output": sample.actual_output,
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


@router.get("/{eval_id}/runs/{run_id}/samples/{sample_id}")
async def get_eval_sample(
    eval_id: str,
    run_id: str,
    sample_id: str,
) -> Dict[str, Any]:
    """Get a specific evaluation sample."""
    store = EvalStore()
    run = await store.get_eval_run(run_id)
    if not run or run.eval_id != eval_id:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    sample = await store.get_eval_sample(sample_id)
    if not sample or sample.eval_run_id != run_id:
        raise HTTPException(status_code=404, detail="Evaluation sample not found")
    return {
        "id": sample.id,
        "object": "eval.sample",
        "eval_run_id": sample.eval_run_id,
        "sample_index": sample.sample_index,
        "input": sample.input,
        "expected_output": sample.expected_output,
        "actual_output": sample.actual_output,
        "score": sample.score,
        "passed": sample.passed,
        "response_time": sample.response_time,
        "tokens_used": sample.tokens_used,
        "metadata": sample.meta,
        "created_at": sample.created_at,
    }
