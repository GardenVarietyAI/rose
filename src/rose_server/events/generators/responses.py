"""Responses event generator for the /v1/responses API."""

from .base import BaseEventGenerator


class ResponsesGenerator(BaseEventGenerator):
    """Generate events for the Responses API.

    The Responses API is similar to chat completions but with:
    - Different response format (response.* events)
    - Support for instructions in addition to messages
    - Store functionality for persisting responses
    """

    # All functionality is handled in the base class
    pass
