"""Streaming utilities for runs execution."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from sse_starlette import ServerSentEvent

from rose_server.runs.steps.store import update_run_step
from rose_server.runs.store import update_run
from rose_server.schemas.runs import RunStepResponse


@dataclass
class MessageData:
    """Common structure for message events."""

    id: str
    thread_id: Optional[str]
    role: str = "assistant"
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    content: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"type": "text", "text": {"value": "", "annotations": []}}]
    )
    file_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, object_type: str = "thread.message") -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": object_type,
            "created_at": int(time.time()),
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "assistant_id": self.assistant_id,
            "run_id": self.run_id,
            "file_ids": self.file_ids,
            "metadata": self.metadata,
        }


async def stream_run_step_event(event_type: str, step: RunStepResponse) -> ServerSentEvent:
    """Create a streaming event for run step updates."""
    event_data = step.dict()
    return ServerSentEvent(data=json.dumps(event_data), event=f"thread.run.step.{event_type}")


async def stream_run_status(run_id: str, status: str, **kwargs: Dict[str, Any]) -> ServerSentEvent:
    """Create a streaming event for run status updates."""
    event_data = {
        "id": run_id,
        "object": f"thread.run.{status}",
        "delta": {"status": status, **kwargs},
    }
    return ServerSentEvent(data=json.dumps(event_data), event=f"thread.run.{status}")


async def stream_message_completed(
    message_id: str,
    full_content: str = "",
    thread_id: Optional[str] = None,
    assistant_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> ServerSentEvent:
    """Create a streaming event for message completion."""
    msg = MessageData(
        id=message_id,
        thread_id=thread_id,
        assistant_id=assistant_id,
        run_id=run_id,
        content=[{"type": "text", "text": {"value": full_content, "annotations": []}}],
    )
    return ServerSentEvent(data=json.dumps(msg.to_dict("thread.message.completed")), event="thread.message.completed")


async def fail_run(
    run_id: str,
    step: Optional[RunStepResponse],
    code: str,
    message: str,
) -> AsyncGenerator[ServerSentEvent, None]:
    err = {"code": code, "message": message}
    if step:
        await update_run_step(step.id, status="failed", last_error=err)
    await update_run(run_id, status="failed", last_error=err)
    status_evt = await stream_run_status(run_id, "failed", last_error=err)
    step_evt = await stream_run_step_event("failed", step) if step else ""
    if step:
        yield step_evt
    yield status_evt
