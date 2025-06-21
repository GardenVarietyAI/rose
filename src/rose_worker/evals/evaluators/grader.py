"""Grader for evaluations - handles testing criteria evaluation."""

import logging
import re
from typing import Any, Dict, List, Optional

from .scorers import exact_match, f1_score
from .scorers.common import normalize_answer

logger = logging.getLogger(__name__)


_SCORERS = {
    "exact_match": exact_match.score,
    "f1": f1_score.score,
}


class Grader:
    """Evaluates model outputs against testing criteria."""

    def __init__(self, testing_criteria: List[Dict[str, Any]]):
        """Initialize with testing criteria.

        Args:
            testing_criteria: List of grader configs, each with:
                - type: "text_similarity", "exact_match", etc.
                - name: Name of the grader
                - input: Template for input (e.g. "{{item.prompt}}")
                - reference: Template for expected (e.g. "{{item.expected}}")
                - evaluation_metric: "cosine", "f1", "exact", etc.
                - pass_threshold: float (0-1)
        """
        self.criteria = testing_criteria or []

        # Default to simple F1 scoring if no criteria provided
        if not self.criteria:
            self.criteria = [
                {"type": "text_similarity", "name": "default_f1", "evaluation_metric": "f1", "pass_threshold": 0.5}
            ]

    def grade_sample(
        self, actual_output: str, expected_output: str, eval_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Grade a single sample and return scores.

        Args:
            actual_output: Model's generated output
            expected_output: Expected/reference output
            eval_name: Optional eval name for dataset-specific scoring

        Returns:
            Dict of metric_name -> score
        """
        scores = {}

        # Apply each criterion
        for criterion in self.criteria:
            score = self._evaluate_criterion(actual_output, expected_output, criterion)

            criterion_name = criterion.get("name", criterion["type"])
            scores[criterion_name] = score

        # Add dataset-specific scoring if applicable
        if eval_name:
            extra_scores = self._dataset_specific_scores(actual_output, expected_output, eval_name)
            scores.update(extra_scores)

        return scores

    def _evaluate_criterion(self, actual: str, expected: str, criterion: Dict[str, Any]) -> float:
        """Evaluate a single testing criterion."""
        criterion_type = criterion.get("type", "text_similarity")
        metric = criterion.get("evaluation_metric", "f1")

        if criterion_type == "text_similarity":
            if metric in _SCORERS:
                return _SCORERS[metric](actual, expected)
            elif metric == "cosine":
                # Simple similarity for now
                return self._simple_similarity(actual, expected)
            else:
                logger.warning(f"Unknown metric {metric}, using F1")
                return _SCORERS["f1"](actual, expected)

        elif criterion_type == "exact_match":
            return _SCORERS["exact_match"](actual, expected)

        else:
            logger.warning(f"Unknown criterion type {criterion_type}")
            return 0.0

    def _simple_similarity(self, actual: str, expected: str) -> float:
        """Simple similarity score as placeholder."""
        actual_norm = normalize_answer(actual)
        expected_norm = normalize_answer(expected)

        if actual_norm == expected_norm:
            return 1.0
        elif expected_norm in actual_norm or actual_norm in expected_norm:
            return 0.8
        else:
            # Check for number matches
            actual_nums = set(re.findall(r"-?\d+\.?\d*", actual))
            expected_nums = set(re.findall(r"-?\d+\.?\d*", expected))

            if expected_nums and actual_nums & expected_nums:
                return 0.6

        return 0.0

    def _dataset_specific_scores(self, actual: str, expected: str, eval_name: str) -> Dict[str, float]:
        """Add dataset-specific scoring metrics."""
        extra_scores = {}

        if eval_name == "gsm8k":
            # Check for number matches
            expected_nums = re.findall(r"-?\d+\.?\d*", expected)
            actual_nums = re.findall(r"-?\d+\.?\d*", actual)

            if expected_nums and actual_nums:
                extra_scores["number_match"] = 1.0 if any(n in actual_nums for n in expected_nums) else 0.0

        # Always add substring match
        expected_norm = normalize_answer(expected)
        actual_norm = normalize_answer(actual)
        extra_scores["substring_match"] = 0.8 if expected_norm in actual_norm or actual_norm in expected_norm else 0.0

        return extra_scores
