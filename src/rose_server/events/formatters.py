"""Event formatters utility functions."""
import json

from sse_starlette.sse import EventSourceResponse


async def create_sse_response(event_generator, formatter) -> EventSourceResponse:
    """Create an SSE response using sse-starlette from an event generator."""

    async def generate():
        """Generate SSE events from LLM events."""
        async for event in event_generator:
            formatted = formatter.format_event(event)
            if formatted:
                yield {
                    "data": formatted.json() if hasattr(formatted, "json") else json.dumps(formatted),
                    "event": "chunk",
                }
        yield {"data": "[DONE]", "event": "done"}
    return EventSourceResponse(generate(), media_type="text/plain")