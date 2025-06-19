"""Exact match scoring function.
Adapted from the official SQuAD evaluation script:
https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
MIT License
Copyright (c) 2016 Pranav Rajpurkar, Stanford NLP
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


def score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score between prediction and ground truth.

    Args:
        prediction: Model's prediction
        ground_truth: Expected answer
    Returns:
        1.0 if normalized strings match exactly, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def score_batch(predictions: list[str], ground_truths: list[str]) -> float:
    """Calculate average exact match score for a batch.

    Args:
        predictions: List of model predictions
        ground_truths: List of expected answers
    Returns:
        Average exact match score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    if not predictions:
        return 0.0
    scores = [score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(scores)
