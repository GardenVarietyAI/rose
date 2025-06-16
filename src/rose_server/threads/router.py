"""API router for threads endpoints."""
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse, StreamingResponse

from rose_server.assistants.store import get_assistant_store
from rose_server.runs.executor import execute_assistant_run_streaming
from rose_server.runs.store import RunsStore
from rose_server.schemas.assistants import CreateRunRequest
from rose_server.schemas.threads import (
    CreateMessageRequest,
    CreateThreadRequest,
)
from rose_server.threads.store import ThreadStore

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)
@router.get("/threads")

async def list_threads(
    limit: int = Query(default=20, description="Number of threads to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
) -> JSONResponse:
    """List all threads."""
    try:
        thread_store = ThreadStore()
        threads = await thread_store.list_threads(limit=limit, order=order)
        thread_data = [thread.model_dump() for thread in threads]
        return JSONResponse(
            content={
                "object": "list",
                "data": thread_data,
                "first_id": thread_data[0]["id"] if thread_data else None,
                "last_id": thread_data[-1]["id"] if thread_data else None,
                "has_more": False,
            }
        )
    except Exception as e:
        logger.error(f"Error listing threads: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error listing threads: {str(e)}"})
@router.post("/threads/runs")

async def create_thread_and_run(request: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Create a thread and immediately run it with an assistant."""
    try:
        assistant_id = request.get("assistant_id")
        if not assistant_id:
            return JSONResponse(status_code=400, content={"error": "assistant_id is required"})
        thread_params = request.get("thread", {})
        messages = thread_params.get("messages", [])
        thread_metadata = thread_params.get("metadata", {})
        thread_store = ThreadStore()
        thread = await thread_store.create_thread(metadata=thread_metadata)
        if messages:
            for msg_data in messages:
                role = msg_data.get("role", "user")
                content = msg_data.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                await thread_store.create_message(
                    thread_id=thread.id,
                    role=role,
                    content=content,
                    metadata=msg_data.get("metadata", {}),
                )
        assistant_store = get_assistant_store()
        assistant = await assistant_store.get_assistant(assistant_id)
        if not assistant:
            return JSONResponse(status_code=404, content={"error": "Assistant not found"})
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
                execute_assistant_run_streaming(run, thread_store, assistant),
                media_type="text/event-stream",
            )
        else:
            events = []
            async for event in execute_assistant_run_streaming(run, thread_store, assistant):
                events.append(event)
            updated_run = await runs_store.get_run(run.id)
            return JSONResponse(content=updated_run.dict())
    except Exception as e:
        logger.error(f"Error in create_thread_and_run: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
@router.post("/threads")

async def create_thread(request: CreateThreadRequest = Body(...)) -> JSONResponse:
    """Create a new conversation thread."""
    try:
        thread_store = ThreadStore()
        thread = await thread_store.create_thread(metadata=request.metadata)
        if request.messages:
            for msg_data in request.messages:
                role = msg_data.get("role", "user")
                content = msg_data.get("content", "")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                await thread_store.create_message(
                    thread_id=thread.id,
                    role=role,
                    content=content,
                    metadata=msg_data.get("metadata", {}),
                )
        return JSONResponse(content=thread.model_dump())
    except Exception as e:
        logger.error(f"Error creating thread: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error creating thread: {str(e)}"})
@router.get("/threads/{thread_id}")

async def get_thread(thread_id: str) -> JSONResponse:
    """Retrieve a thread by ID."""
    try:
        thread_store = ThreadStore()
        thread = await thread_store.get_thread(thread_id)
        if not thread:
            return JSONResponse(status_code=404, content={"error": "Thread not found"})
        return JSONResponse(content=thread.model_dump())
    except Exception as e:
        logger.error(f"Error retrieving thread: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving thread: {str(e)}"})
@router.post("/threads/{thread_id}")

async def update_thread(thread_id: str, request: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Update a thread's metadata."""
    try:
        thread_store = ThreadStore()
        metadata = request.get("metadata", {})
        thread = await thread_store.update_thread(thread_id, metadata)
        if not thread:
            return JSONResponse(status_code=404, content={"error": "Thread not found"})
        return JSONResponse(content=thread.model_dump())
    except Exception as e:
        logger.error(f"Error updating thread: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error updating thread: {str(e)}"})
@router.delete("/threads/{thread_id}")

async def delete_thread(thread_id: str) -> JSONResponse:
    """Delete a thread."""
    try:
        thread_store = ThreadStore()
        success = await thread_store.delete_thread(thread_id)
        if not success:
            return JSONResponse(status_code=404, content={"error": "Thread not found"})
        return JSONResponse(content={"id": thread_id, "object": "thread.deleted", "deleted": True})
    except Exception as e:
        logger.error(f"Error deleting thread: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error deleting thread: {str(e)}"})
@router.post("/threads/{thread_id}/messages")

async def create_message(
    thread_id: str,
    request: CreateMessageRequest = Body(...),
    enable_embedding: bool = Query(default=True, description="Whether to embed message to vector store"),
) -> JSONResponse:
    """Create a message in a thread with optional vector embedding."""
    try:
        thread_store = ThreadStore()
        if not await thread_store.get_thread(thread_id):
            return JSONResponse(status_code=404, content={"error": "Thread not found"})
        content = request.content
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        embedding_func = None
        if enable_embedding:
            from rose_server.embeddings import generate_embeddings
            embedding_func = generate_embeddings
        message = await thread_store.create_message(
            thread_id=thread_id,
            role=request.role,
            content=content,
            metadata=request.metadata,
            enable_embedding=enable_embedding,
            embedding_func=embedding_func,
        )
        return JSONResponse(content=message.model_dump())
    except Exception as e:
        logger.error(f"Error creating message: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error creating message: {str(e)}"})
@router.get("/threads/{thread_id}/messages")

async def list_messages(
    thread_id: str,
    limit: int = Query(default=20, description="Number of messages to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
    after: Optional[str] = Query(default=None, description="Cursor for pagination"),
    before: Optional[str] = Query(default=None, description="Cursor for pagination"),
) -> JSONResponse:
    """List messages in a thread."""
    try:
        thread_store = ThreadStore()
        if not await thread_store.get_thread(thread_id):
            return JSONResponse(status_code=404, content={"error": "Thread not found"})
        messages = await thread_store.get_messages(thread_id, limit=limit, order=order)
        message_data = [msg.model_dump() for msg in messages]
        return JSONResponse(
            content={
                "object": "list",
                "data": message_data,
                "first_id": message_data[0]["id"] if message_data else None,
                "last_id": message_data[-1]["id"] if message_data else None,
                "has_more": False,
            }
        )
    except Exception as e:
        logger.error(f"Error listing messages: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error listing messages: {str(e)}"})
@router.get("/threads/{thread_id}/messages/{message_id}")

async def get_message(thread_id: str, message_id: str) -> JSONResponse:
    """Retrieve a specific message from a thread."""
    try:
        thread_store = ThreadStore()
        if not await thread_store.get_thread(thread_id):
            return JSONResponse(status_code=404, content={"error": "Thread not found"})
        message = await thread_store.get_message(thread_id, message_id)
        if not message:
            return JSONResponse(status_code=404, content={"error": "Message not found"})
        return JSONResponse(content=message.model_dump())
    except Exception as e:
        logger.error(f"Error retrieving message: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving message: {str(e)}"})