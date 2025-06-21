"""Includes/contains scoring function.
Inspired by OpenAI Evals includes implementation:
https://github.com/openai/evals/blob/main/evals/elsuite/basic/includes.py
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

from .common import normalize_answer


def score(prediction: str, ground_truth: str, normalize: bool = True) -> float:
    """Check if ground truth is included in prediction.

    Args:
        prediction: Model's prediction
        ground_truth: Expected answer to find in prediction
        normalize: Whether to normalize strings before checking
    Returns:
        1.0 if ground truth is found in prediction, 0.0 otherwise
    """
    if normalize:
        pred = normalize_answer(prediction)
        gt = normalize_answer(ground_truth)
    else:
        pred = prediction
        gt = ground_truth

    # Handle empty ground truth
    if not gt:
        return 1.0

    return 1.0 if gt in pred else 0.0


def score_batch(predictions: list[str], ground_truths: list[str], normalize: bool = True) -> float:
    """Calculate average includes score for a batch.

    Args:
        predictions: List of model predictions
        ground_truths: List of expected answers
        normalize: Whether to normalize strings before checking
    Returns:
        Average includes score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    if not predictions:
        return 0.0
    scores = [score(pred, gt, normalize) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(scores)


def score_any(prediction: str, ground_truths: list[str], normalize: bool = True) -> float:
    """Check if any of the ground truths is included in prediction.

    Args:
        prediction: Model's prediction
        ground_truths: List of acceptable answers to find in prediction
        normalize: Whether to normalize strings before checking
    Returns:
        1.0 if any ground truth is found in prediction, 0.0 otherwise
    """
    if normalize:
        pred = normalize_answer(prediction)
        gts = [normalize_answer(gt) for gt in ground_truths]
    else:
        pred = prediction
        gts = ground_truths

    return 1.0 if any(gt in pred for gt in gts if gt) else 0.0


def score_all(prediction: str, ground_truths: list[str], normalize: bool = True) -> float:
    """Check if all ground truths are included in prediction.

    Args:
        prediction: Model's prediction
        ground_truths: List of required answers that must all be in prediction
        normalize: Whether to normalize strings before checking
    Returns:
        1.0 if all ground truths are found in prediction, 0.0 otherwise
    """
    if normalize:
        pred = normalize_answer(prediction)
        gts = [normalize_answer(gt) for gt in ground_truths]
    else:
        pred = prediction
        gts = ground_truths

    # Handle empty list
    if not gts:
        return 1.0

    return 1.0 if all(gt in pred for gt in gts if gt) else 0.0
