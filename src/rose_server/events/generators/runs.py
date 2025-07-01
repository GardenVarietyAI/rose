"""Runs event generator for assistant execution."""

from .base import BaseEventGenerator


class RunsGenerator(BaseEventGenerator):
    """Generate events for assistant runs. Event-based streaming using the same infrastructure as chat completions."""

    # All functionality is handled in the base class
    pass
