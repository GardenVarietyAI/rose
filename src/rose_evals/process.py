"""Evaluation job processor."""

import logging
from typing import Any, Dict

from rose_core.models import cleanup_model_memory
from rose_core.webhook import post_webhook

from .evaluators.simple_evaluator import SimpleEvaluator

logger = logging.getLogger(__name__)


def process_eval_job(job_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single evaluation job."""
    eval_id = payload.get("eval_id")
    model = payload.get("model")
    metadata = payload.get("metadata", {})

    if not eval_id or not model:
        raise ValueError(f"Missing required fields: eval_id={eval_id}, model={model}")

    logger.info(f"Starting eval job {job_id} for eval {eval_id} with model {model}")

    post_webhook("job.running", "eval", job_id, eval_id)

    try:
        data_source_config = metadata.get("eval_data_source_config", {})
        testing_criteria = metadata.get("eval_testing_criteria", [])
        run_data_source = metadata.get("data_source", {})
        eval_name = metadata.get("eval", metadata.get("run_name"))

        evaluator = SimpleEvaluator()
        result = evaluator.run_evaluation(
            eval_run_id=eval_id,
            model=model,
            data_source_config=data_source_config,
            testing_criteria=testing_criteria,
            run_data_source=run_data_source,
            eval_name=eval_name,
        )

        post_webhook(
            "job.completed",
            "eval",
            job_id,
            eval_id,
            {"results": result.get("results", {}), "samples": result.get("samples", [])},
        )

        logger.info(f"Eval job {job_id} completed successfully")
        return {"eval_id": eval_id, "status": "completed", "results": result}

    except Exception as exc:
        logger.exception(f"Eval job {job_id} failed")
        post_webhook("job.failed", "eval", job_id, eval_id, {"error": {"message": str(exc), "code": "job_error"}})
        raise
    finally:
        cleanup_model_memory()
