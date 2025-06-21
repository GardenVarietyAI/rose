"""Simplified evaluator that orchestrates evaluation components."""

import logging
import time
from typing import Any, Dict, List, Optional

from .data_source_loader import DataSourceLoader
from .grader import Grader
from .model_runner import ModelRunner

logger = logging.getLogger(__name__)


class SimpleEvaluator:
    """Simplified evaluator that uses focused components."""

    def run_evaluation(
        self,
        eval_run_id: str,
        model: str,
        data_source_config: Dict[str, Any],
        testing_criteria: List[Dict[str, Any]],
        run_data_source: Optional[Dict[str, Any]] = None,
        eval_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run an evaluation.

        Args:
            eval_run_id: Evaluation run ID
            model: Model to evaluate
            data_source_config: Eval's data source configuration
            testing_criteria: Eval's testing criteria
            run_data_source: Run-specific overrides (model, sampling_params, etc.)

        Returns:
            Dict with results and samples
        """
        logger.info(f"Starting evaluation {eval_run_id} for model {model}")

        try:
            # Use run-specific params if provided
            if run_data_source:
                model = run_data_source.get("model", model)
                sampling_params = run_data_source.get("sampling_params")
                max_samples = run_data_source.get("max_samples")
            else:
                sampling_params = None
                max_samples = None

            # Load dataset
            loader = DataSourceLoader()
            samples = loader.load_dataset(data_source_config, max_samples)

            if not samples:
                raise ValueError("No samples loaded from data source")

            logger.info(f"Loaded {len(samples)} samples for evaluation")

            # Initialize components
            runner = ModelRunner(model)
            grader = Grader(testing_criteria)

            # Process samples
            sample_results = []
            all_scores = []

            for idx, sample in enumerate(samples):
                start_time = time.time()

                # Generate model output
                output = runner.generate(sample["input"], sampling_params)

                response_time = time.time() - start_time

                # Grade the output
                scores = grader.grade_sample(
                    actual_output=output, expected_output=sample["expected"], eval_name=eval_name
                )

                all_scores.append(scores)
                max_score = max(scores.values()) if scores else 0.0
                passed = max_score >= 0.5  # Default threshold

                sample_result = {
                    "sample_index": idx,
                    "input": sample["input"],
                    "ideal": sample["expected"],
                    "completion": output,
                    "score": max_score,
                    "passed": passed,
                    "response_time": response_time,
                    "tokens_used": len(sample["input"].split()) + len(output.split()),
                    "metadata": {"scores": scores, "eval_name": eval_name},
                }

                sample_results.append(sample_result)

                if idx % 10 == 0:
                    logger.info(f"Progress: {idx + 1}/{len(samples)} samples")

            aggregate_results = self._aggregate_results(all_scores, sample_results)

            logger.info(f"Evaluation {eval_run_id} completed")

            return {"results": aggregate_results, "samples": sample_results}

        except Exception as e:
            logger.error(f"Evaluation {eval_run_id} failed: {str(e)}")
            raise

    def _aggregate_results(
        self, all_scores: List[Dict[str, float]], sample_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate evaluation results."""
        if not all_scores:
            return {"error": "No results to aggregate"}

        # Get all unique metrics
        all_metrics: set[Any] = set()
        for scores in all_scores:
            all_metrics.update(scores.keys())

        # Calculate per-metric stats
        metrics = {}
        for metric in all_metrics:
            values = [scores.get(metric, 0.0) for scores in all_scores]
            passed_for_metric = sum(1 for v in values if v >= 0.5)

            metrics[metric] = {"mean": sum(values) / len(values), "total": len(values), "passed": passed_for_metric}

        # Overall stats
        total_passed = sum(1 for result in sample_results if result["passed"])

        return {
            "metrics": metrics,
            "total_samples": len(sample_results),
            "passed_samples": total_passed,
            "failed_samples": len(sample_results) - total_passed,
            "pass_rate": total_passed / len(sample_results) if sample_results else 0.0,
        }
