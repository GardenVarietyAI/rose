"""API router for threads endpoints."""

import logging
import time
from typing import Any, Dict, Union

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from rose_server.assistants.store import get_assistant
from rose_server.entities.messages import Message
from rose_server.entities.threads import Thread
from rose_server.schemas.runs import RunResponse
from rose_server.schemas.threads import ThreadAndRunCreateRequest, ThreadCreateRequest, ThreadResponse
from rose_server.threads.messages.router import router as messages_router
from rose_server.threads.messages.store import create_message
from rose_server.threads.runs import execute_assistant_run_streaming, get_run
from rose_server.threads.runs.router import router as runs_router
from rose_server.threads.runs.store import create_run
from rose_server.threads.store import create_thread, delete_thread, get_thread, list_threads, update_thread

router = APIRouter(prefix="/v1/threads")
logger = logging.getLogger(__name__)

router.include_router(messages_router)
router.include_router(runs_router)


@router.get("")
async def index(
    limit: int = Query(default=20, description="Number of threads to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
) -> JSONResponse:
    """List all threads."""
    threads = await list_threads(limit=limit, order=order)
    thread_data = [ThreadResponse(**thread.model_dump()).model_dump() for thread in threads]
    return JSONResponse(
        content={
            "object": "list",
            "data": thread_data,
            "first_id": thread_data[0]["id"] if thread_data else None,
            "last_id": thread_data[-1]["id"] if thread_data else None,
            "has_more": False,
        }
    )


@router.post("", response_model=ThreadResponse)
async def create(request: ThreadCreateRequest = Body(...)) -> ThreadResponse:
    """Create a new conversation thread."""

    thread = await create_thread(Thread(meta=request.metadata, tool_resources=None))

    if request.messages:
        for msg_data in request.messages:
            role = msg_data.get("role", "user")
            content = msg_data.get("content", "")
            formatted_content = [{"type": "text", "text": {"value": content, "annotations": []}}]

            await create_message(
                Message(
                    thread_id=thread.id,
                    role=role,
                    content=formatted_content,
                    meta=msg_data.get("metadata", {}),
                    completed_at=int(time.time()),
                )
            )

    return ThreadResponse(**thread.model_dump())


@router.post("/runs", response_model=None)
async def create_thread_and_run(
    request: ThreadAndRunCreateRequest = Body(...),
) -> Union[RunResponse, EventSourceResponse]:
    """Create a thread and immediately run it with an assistant."""
    try:
        assistant_id = request.assistant_id
        assistant = await get_assistant(assistant_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        thread_params = request.thread
        messages = thread_params.get("messages", [])
        thread_metadata = thread_params.get("metadata", {})
        thread = await create_thread(Thread(meta=thread_metadata, tool_resources=None))

        if messages:
            for msg_data in messages:
                role = msg_data.get("role", "user")
                content = msg_data.get("content", "")
                formatted_content = [{"type": "text", "text": {"value": content, "annotations": []}}]

                await create_message(
                    Message(
                        thread_id=thread.id,
                        role=role,
                        content=formatted_content,
                        meta=msg_data.get("metadata", {}),
                        completed_at=int(time.time()),
                    )
                )

        run = await create_run(
            thread_id=thread.id,
            assistant_id=request.assistant_id,
            model=request.model or assistant.model,
            instructions=request.instructions or assistant.instructions,
            additional_instructions=None,
            additional_messages=None,
            tools=[
                tool.model_dump() if hasattr(tool, "model_dump") else tool
                for tool in (request.tools or assistant.tools or [])
            ],
            metadata=request.metadata or {},
            temperature=(request.temperature if request.temperature is not None else assistant.temperature),
            top_p=request.top_p if request.top_p is not None else assistant.top_p,
            max_prompt_tokens=request.max_prompt_tokens,
            max_completion_tokens=request.max_completion_tokens,
            truncation_strategy=request.truncation_strategy,
            tool_choice=request.tool_choice,
            parallel_tool_calls=(request.parallel_tool_calls if request.parallel_tool_calls is not None else True),
            response_format=request.response_format,
            stream=request.stream,
        )

        if request.stream:
            return EventSourceResponse(execute_assistant_run_streaming(run, assistant))
        else:
            events = []
            async for event in execute_assistant_run_streaming(run, assistant):
                events.append(event)
            updated_run = await get_run(run.id)
            return RunResponse(**updated_run.model_dump())
    except Exception as e:
        logger.error(f"Error in create_thread_and_run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{thread_id}", response_model=ThreadResponse)
async def get(thread_id: str) -> ThreadResponse:
    """Retrieve a thread by ID."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(**thread.model_dump())


@router.post("/{thread_id}", response_model=ThreadResponse)
async def update(thread_id: str, request: Dict[str, Any] = Body(...)) -> ThreadResponse:
    """Update a thread's metadata."""
    metadata = request.get("metadata", {})
    thread = await update_thread(thread_id, metadata)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(**thread.model_dump())


@router.delete("/{thread_id}")
async def delete(thread_id: str) -> Dict[str, Any]:
    """Delete a thread."""
    success = await delete_thread(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {
        "id": thread_id,
        "object": "thread.deleted",
        "deleted": True,
    }
