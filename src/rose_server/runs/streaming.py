"""Streaming utilities for runs execution."""
import asyncio
import json
import time
from typing import AsyncGenerator
from rose_server.schemas.runs import RunStep

async def stream_run_status(run_id: str, status: str, **kwargs) -> str:
    """Create a streaming event for run status updates."""
    event_data = {
        "id": run_id,
        "object": f"thread.run.{status}",
        "delta": {"status": status, **kwargs},
    }
    return f"event: thread.run.{status}\ndata: {json.dumps(event_data)}\n\n"

async def stream_message_created(message_id: str, thread_id: str, assistant_id: str = None, run_id: str = None) -> str:
    """Create a streaming event for message creation."""
    event_data = {
        "id": message_id,
        "object": "thread.message",
        "created_at": int(time.time()),
        "thread_id": thread_id,
        "role": "assistant",
        "content": [{"type": "text", "text": {"value": "", "annotations": []}}],
        "assistant_id": assistant_id,
        "run_id": run_id,
        "file_ids": [],
        "metadata": {},
    }
    return f"event: thread.message.created\ndata: {json.dumps(event_data)}\n\n"

async def stream_message_in_progress(
    message_id: str, thread_id: str = None, assistant_id: str = None, run_id: str = None
) -> str:
    """Create a streaming event for message in progress."""
    event_data = {
        "id": message_id,
        "object": "thread.message.in_progress",
        "created_at": int(time.time()),
        "thread_id": thread_id,
        "role": "assistant",
        "content": [{"type": "text", "text": {"value": "", "annotations": []}}],
        "assistant_id": assistant_id,
        "run_id": run_id,
        "file_ids": [],
        "metadata": {},
    }
    return f"event: thread.message.in_progress\ndata: {json.dumps(event_data)}\n\n"

async def stream_message_chunk(run_id: str, message_id: str, text: str, index: int = 0) -> str:
    """Create a streaming event for message content delta."""
    event_data = {
        "id": message_id,
        "object": "thread.message.delta",
        "delta": {"content": [{"index": index, "type": "text", "text": {"value": text}}]},
    }
    return f"event: thread.message.delta\ndata: {json.dumps(event_data)}\n\n"

async def stream_message_completed(
    message_id: str,
    full_content: str = "",
    thread_id: str = None,
    assistant_id: str = None,
    run_id: str = None,
) -> str:
    """Create a streaming event for message completion."""
    event_data = {
        "id": message_id,
        "object": "thread.message.completed",
        "created_at": int(time.time()),
        "thread_id": thread_id,
        "role": "assistant",
        "content": [{"type": "text", "text": {"value": full_content, "annotations": []}}],
        "assistant_id": assistant_id,
        "run_id": run_id,
        "file_ids": [],
        "metadata": {},
    }
    return f"event: thread.message.completed\ndata: {json.dumps(event_data)}\n\n"

async def stream_run_step_event(event_type: str, step: RunStep) -> str:
    """Create a streaming event for run step updates."""
    event_data = step.dict()
    return f"event: thread.run.step.{event_type}\ndata: {json.dumps(event_data)}\n\n"

async def stream_agent_response(run_id: str, message_id: str, response_text: str) -> AsyncGenerator[str, None]:
    """Stream agent response word by word."""
    words = response_text.split()
    for i, word in enumerate(words):
        delta_text = word + (" " if i < len(words) - 1 else "")
        yield await stream_message_chunk(run_id, message_id, delta_text)
        await asyncio.sleep(0.05)