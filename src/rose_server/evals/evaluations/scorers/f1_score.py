"""F1 scoring function for token-level overlap.
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
from collections import Counter

from .common import normalize_answer


def score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth.

    F1 score measures token-level overlap between the prediction and ground truth.
    It's the harmonic mean of precision and recall.
    Args:
        prediction: Model's prediction
        ground_truth: Expected answer
    Returns:
        F1 score between 0.0 and 1.0
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def score_batch(predictions: list[str], ground_truths: list[str]) -> float:
    """Calculate average F1 score for a batch.

    Args:
        predictions: List of model predictions
        ground_truths: List of expected answers
    Returns:
        Average F1 score (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    if not predictions:
        return 0.0
    scores = [score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(scores)