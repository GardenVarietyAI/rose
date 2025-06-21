"""Fuzzy match scoring function.
Inspired by OpenAI Evals fuzzy match implementation:
https://github.com/openai/evals/blob/main/evals/elsuite/basic/fuzzy_match.py
MIT License
Copyright (c) 2023 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from difflib import SequenceMatcher

from .common import normalize_answer


def score(prediction: str, ground_truth: str, threshold: float = 0.8) -> float:
    """Calculate fuzzy match score between prediction and ground truth.

    Uses SequenceMatcher to compute similarity ratio between normalized strings.

    Args:
        prediction: Model's prediction
        ground_truth: Expected answer
        threshold: Minimum similarity ratio to consider a match (0.0 to 1.0)
    Returns:
        1.0 if similarity >= threshold, 0.0 otherwise
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    # Handle empty strings
    if not norm_pred and not norm_gt:
        return 1.0
    if not norm_pred or not norm_gt:
        return 0.0

    similarity = SequenceMatcher(None, norm_pred, norm_gt).ratio()
    return 1.0 if similarity >= threshold else 0.0


def score_batch(predictions: list[str], ground_truths: list[str], threshold: float = 0.8) -> float:
    """Calculate average fuzzy match score for a batch.

    Args:
        predictions: List of model predictions
        ground_truths: List of expected answers
        threshold: Minimum similarity ratio to consider a match (0.0 to 1.0)
    Returns:
        Average fuzzy match score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    if not predictions:
        return 0.0
    scores = [score(pred, gt, threshold) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(scores)


def score_with_ratio(prediction: str, ground_truth: str) -> tuple[float, float]:
    """Calculate fuzzy match score and similarity ratio.

    Args:
        prediction: Model's prediction
        ground_truth: Expected answer
    Returns:
        Tuple of (binary score at 0.8 threshold, raw similarity ratio)
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    # Handle empty strings
    if not norm_pred and not norm_gt:
        return 1.0, 1.0
    if not norm_pred or not norm_gt:
        return 0.0, 0.0

    similarity = SequenceMatcher(None, norm_pred, norm_gt).ratio()
    return (1.0 if similarity >= 0.8 else 0.0), similarity
