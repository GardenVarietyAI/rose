"""Numeric comparison scoring functions.
Inspired by OpenAI Evals numeric matching:
https://github.com/openai/evals
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

import re
from typing import Optional, Union

# Precompiled regex pattern for number extraction
NUMBER_EXTRACT_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def extract_number(text: str) -> Optional[float]:
    """Extract a numeric value from text.

    Handles integers, decimals, negative numbers, and scientific notation.

    Args:
        text: Text containing a number
    Returns:
        Extracted number or None if not found
    """
    if not text:
        return None

    # Find all numbers in the text
    matches = NUMBER_EXTRACT_PATTERN.findall(text)
    if matches:
        try:
            # Return the first valid number found
            return float(matches[0])
        except ValueError:
            pass

    return None


def score_exact(prediction: str, ground_truth: Union[str, float]) -> float:
    """Score numeric values for exact match.

    Args:
        prediction: Model's prediction
        ground_truth: Expected numeric value
    Returns:
        1.0 if numbers match exactly, 0.0 otherwise
    """
    pred_num = extract_number(str(prediction))

    if isinstance(ground_truth, str):
        gt_num = extract_number(ground_truth)
    else:
        gt_num = float(ground_truth)

    if pred_num is None or gt_num is None:
        return 0.0

    return 1.0 if pred_num == gt_num else 0.0


def score_tolerance(prediction: str, ground_truth: Union[str, float], tolerance: float = 1e-6) -> float:
    """Score numeric values with absolute tolerance.

    Args:
        prediction: Model's prediction
        ground_truth: Expected numeric value
        tolerance: Maximum allowed absolute difference
    Returns:
        1.0 if within tolerance, 0.0 otherwise
    """
    pred_num = extract_number(str(prediction))

    if isinstance(ground_truth, str):
        gt_num = extract_number(ground_truth)
    else:
        gt_num = float(ground_truth)

    if pred_num is None or gt_num is None:
        return 0.0

    return 1.0 if abs(pred_num - gt_num) <= tolerance else 0.0


def score_relative(prediction: str, ground_truth: Union[str, float], relative_tolerance: float = 0.01) -> float:
    """Score numeric values with relative tolerance.

    Args:
        prediction: Model's prediction
        ground_truth: Expected numeric value
        relative_tolerance: Maximum allowed relative difference (e.g., 0.01 for 1%)
    Returns:
        1.0 if within relative tolerance, 0.0 otherwise
    """
    pred_num = extract_number(str(prediction))

    if isinstance(ground_truth, str):
        gt_num = extract_number(ground_truth)
    else:
        gt_num = float(ground_truth)

    if pred_num is None or gt_num is None:
        return 0.0

    # Handle zero ground truth
    if gt_num == 0:
        return 1.0 if pred_num == 0 else 0.0

    relative_error = abs(pred_num - gt_num) / abs(gt_num)
    return 1.0 if relative_error <= relative_tolerance else 0.0


def score_batch_exact(predictions: list[str], ground_truths: list[Union[str, float]]) -> float:
    """Calculate average exact numeric match score for a batch."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    if not predictions:
        return 0.0
    scores = [score_exact(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(scores)


def score_batch_tolerance(
    predictions: list[str], ground_truths: list[Union[str, float]], tolerance: float = 1e-6
) -> float:
    """Calculate average numeric score with tolerance for a batch."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    if not predictions:
        return 0.0
    scores = [score_tolerance(pred, gt, tolerance) for pred, gt in zip(predictions, ground_truths)]
    return sum(scores) / len(scores)
