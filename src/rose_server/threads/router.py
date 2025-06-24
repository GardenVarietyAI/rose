"""API router for threads endpoints."""

import logging
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from rose_server.assistants.store import get_assistant
from rose_server.database import Thread, current_timestamp
from rose_server.messages.store import create_message
from rose_server.runs.executor import execute_assistant_run_streaming
from rose_server.runs.store import RunsStore
from rose_server.schemas.runs import CreateRunRequest
from rose_server.schemas.threads import ThreadCreateRequest, ThreadResponse
from rose_server.threads.store import (
    create_thread,
    delete_thread,
    get_thread,
    list_threads as list_threads_store,
    update_thread,
)

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.get("/threads")
async def list_threads(
    limit: int = Query(default=20, description="Number of threads to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
) -> JSONResponse:
    """List all threads."""
    threads = await list_threads_store(limit=limit, order=order)
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


@router.post("/threads", response_model=ThreadResponse)
async def create(request: ThreadCreateRequest = Body(...)):
    """Create a new conversation thread."""
    thread = Thread(
        id=f"thread_{uuid.uuid4().hex}",
        created_at=current_timestamp(),
        meta=request.metadata,
        tool_resources=None,
    )
    thread = await create_thread(thread)
    if request.messages:
        for msg_data in request.messages:
            role = msg_data.get("role", "user")
            content = msg_data.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            await create_message(
                thread_id=thread.id,
                role=role,
                content=content,
                metadata=msg_data.get("metadata", {}),
            )
    return ThreadResponse(**thread.model_dump())


@router.post("/threads/runs")
async def create_thread_and_run(request: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Create a thread and immediately run it with an assistant."""
    try:
        assistant_id = request.get("assistant_id")
        if not assistant_id:
            return JSONResponse(status_code=400, content={"error": "assistant_id is required"})

        assistant = await get_assistant(assistant_id)
        if not assistant:
            return JSONResponse(status_code=404, content={"error": "Assistant not found"})

        thread_params = request.get("thread", {})
        messages = thread_params.get("messages", [])
        thread_metadata = thread_params.get("metadata", {})
        thread = Thread(
            id=f"thread_{uuid.uuid4().hex}",
            created_at=current_timestamp(),
            meta=thread_metadata,
            tool_resources=None,
        )
        thread = await create_thread(thread)

        if messages:
            for msg_data in messages:
                role = msg_data.get("role", "user")
                content = msg_data.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                await create_message(
                    thread_id=thread.id,
                    role=role,
                    content=content,
                    metadata=msg_data.get("metadata", {}),
                )

        run_request = CreateRunRequest(
            assistant_id=assistant_id,
            model=request.get("model"),
            instructions=request.get("instructions"),
            tools=request.get("tools"),
            metadata=request.get("metadata", {}),
            temperature=request.get("temperature"),
            stream=request.get("stream", False),
        )

        runs_store = RunsStore()
        run = await runs_store.create_run(thread.id, run_request)

        if not run.model or run.model == "zephyr":
            run.model = assistant.model

        if not run.instructions:
            run.instructions = assistant.instructions

        if not run.tools:
            run.tools = assistant.tools

        if run_request.stream:
            return StreamingResponse(
                execute_assistant_run_streaming(run, assistant),
                media_type="text/event-stream",
            )
        else:
            events = []
            async for event in execute_assistant_run_streaming(run, assistant):
                events.append(event)
            updated_run = await runs_store.get_run(run.id)
            return JSONResponse(content=updated_run.dict())
    except Exception as e:
        logger.error(f"Error in create_thread_and_run: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get(thread_id: str):
    """Retrieve a thread by ID."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(**thread.model_dump())


@router.post("/threads/{thread_id}", response_model=ThreadResponse)
async def update(thread_id: str, request: Dict[str, Any] = Body(...)):
    """Update a thread's metadata."""
    metadata = request.get("metadata", {})
    thread = await update_thread(thread_id, metadata)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(**thread.model_dump())


@router.delete("/threads/{thread_id}")
async def delete(thread_id: str):
    """Delete a thread."""
    success = await delete_thread(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"id": thread_id, "object": "thread.deleted", "deleted": True}
