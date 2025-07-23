"""API router for threads endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from rose_server.assistants.store import get_assistant
from rose_server.database import current_timestamp
from rose_server.entities.messages import Message
from rose_server.entities.runs import Run
from rose_server.entities.threads import Thread
from rose_server.runs.executor import execute_assistant_run_streaming
from rose_server.runs.store import create_run, get_run
from rose_server.schemas.runs import RunCreateRequest, RunResponse
from rose_server.schemas.threads import ThreadCreateRequest, ThreadResponse
from rose_server.threads.messages.router import router as messages_router
from rose_server.threads.messages.store import create_message
from rose_server.threads.store import (
    create_thread,
    delete_thread,
    get_thread,
    list_threads as list_threads_store,
    update_thread,
)
from rose_server.vector_stores.deps import VectorManager

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

# Include the messages router
router.include_router(messages_router)


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
async def create(request: ThreadCreateRequest = Body(...)) -> ThreadResponse:
    """Create a new conversation thread."""
    thread = Thread(
        created_at=current_timestamp(),
        meta=request.metadata,
        tool_resources=None,
    )

    thread = await create_thread(thread)

    if request.messages:
        for msg_data in request.messages:
            role = msg_data.get("role", "user")
            content = msg_data.get("content", "")

            # Format content to match OpenAI structure
            if isinstance(content, str):
                formatted_content = [
                    {
                        "type": "text",
                        "text": {
                            "value": content,
                            "annotations": [],
                        },
                    }
                ]
            else:
                formatted_content = content

            # Create message entity
            message = Message(
                thread_id=thread.id,
                role=role,
                content=formatted_content,
                created_at=current_timestamp(),
                meta=msg_data.get("metadata", {}),
                completed_at=current_timestamp(),
            )
            await create_message(message)

    return ThreadResponse(**thread.model_dump())


@router.post("/threads/runs")
async def create_thread_and_run(
    request: Dict[str, Any] = Body(...), vector: VectorManager = VectorManager
) -> JSONResponse:
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
            created_at=current_timestamp(),
            meta=thread_metadata,
            tool_resources=None,
        )
        thread = await create_thread(thread)

        if messages:
            for msg_data in messages:
                role = msg_data.get("role", "user")
                content = msg_data.get("content", "")

                # Format content to match OpenAI structure
                if isinstance(content, str):
                    formatted_content = [
                        {
                            "type": "text",
                            "text": {
                                "value": content,
                                "annotations": [],
                            },
                        }
                    ]
                else:
                    formatted_content = content

                # Create message entity
                message = Message(
                    thread_id=thread.id,
                    role=role,
                    content=formatted_content,
                    created_at=current_timestamp(),
                    meta=msg_data.get("metadata", {}),
                    completed_at=current_timestamp(),
                )
                await create_message(message)

        run_request = RunCreateRequest(
            assistant_id=assistant_id,
            model=request.get("model"),
            instructions=request.get("instructions"),
            tools=request.get("tools"),
            metadata=request.get("metadata", {}),
            temperature=request.get("temperature"),
            stream=request.get("stream", False),
        )

        run = Run(
            created_at=current_timestamp(),
            thread_id=thread.id,
            assistant_id=run_request.assistant_id,
            status="queued",
            model=run_request.model if run_request.model is not None else assistant.model,
            instructions=run_request.instructions or assistant.instructions,
            tools=[
                tool.model_dump() if hasattr(tool, "model_dump") else tool
                for tool in (run_request.tools or assistant.tools or [])
            ],
            meta=run_request.metadata or {},
            temperature=run_request.temperature if run_request.temperature is not None else assistant.temperature,
            top_p=run_request.top_p if run_request.top_p is not None else assistant.top_p,
            max_prompt_tokens=run_request.max_prompt_tokens,
            max_completion_tokens=run_request.max_completion_tokens,
            truncation_strategy=run_request.truncation_strategy,
            tool_choice=run_request.tool_choice,
            parallel_tool_calls=run_request.parallel_tool_calls
            if run_request.parallel_tool_calls is not None
            else True,
            response_format=run_request.response_format,
        )

        run = await create_run(run)

        if run_request.stream:
            return EventSourceResponse(execute_assistant_run_streaming(run, assistant, vector))
        else:
            events = []
            async for event in execute_assistant_run_streaming(run, assistant, vector):
                events.append(event)
            updated_run = await get_run(run.id)
            return JSONResponse(content=RunResponse(**updated_run.model_dump()).model_dump())
    except Exception as e:
        logger.error(f"Error in create_thread_and_run: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get(thread_id: str) -> ThreadResponse:
    """Retrieve a thread by ID."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(**thread.model_dump())


@router.post("/threads/{thread_id}", response_model=ThreadResponse)
async def update(thread_id: str, request: Dict[str, Any] = Body(...)) -> ThreadResponse:
    """Update a thread's metadata."""
    metadata = request.get("metadata", {})
    thread = await update_thread(thread_id, metadata)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(**thread.model_dump())


@router.delete("/threads/{thread_id}")
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
