"""Perplexity metric computation."""

import math
from typing import Union


def compute_perplexity(loss: Union[float, int]) -> float:
    """Compute perplexity from loss value.

    Args:
        loss: The loss value (typically cross-entropy loss)

    Returns:
        Perplexity value. Capped at 1e10 to avoid overflow.
    """
    # Cap loss to prevent overflow in exp()
    # Loss > 23 would give perplexity > 1e10
    capped_loss = min(loss, 23.0)
    return math.exp(capped_loss)
