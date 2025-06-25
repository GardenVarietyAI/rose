"""API router for assistants endpoints."""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from rose_server.assistants.store import (
    create_assistant,
    delete_assistant,
    get_assistant,
    list_assistants,
    update_assistant,
)
from rose_server.database import current_timestamp
from rose_server.entities.assistants import Assistant
from rose_server.schemas.assistants import AssistantCreateRequest, AssistantResponse, AssistantUpdateRequest

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.post("/assistants", response_model=AssistantResponse)
async def create(request: AssistantCreateRequest):
    """Create a new assistant."""
    # Convert tools to dicts for JSON storage
    tools = [tool.model_dump() for tool in request.tools] if request.tools else []

    assistant = Assistant(
        id=f"asst_{uuid.uuid4().hex}",
        created_at=current_timestamp(),
        name=request.name,
        description=request.description,
        model=request.model,
        instructions=request.instructions,
        tools=tools,
        tool_resources=request.tool_resources or {},
        meta=request.metadata or {},
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 1.0,
        response_format=request.response_format,
    )
    assistant = await create_assistant(assistant)
    return AssistantResponse(**assistant.model_dump())


@router.get("/assistants")
async def index(
    limit: int = Query(default=20, description="Number of assistants to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
    after: Optional[str] = Query(default=None, description="Cursor for pagination after this object ID"),
    before: Optional[str] = Query(default=None, description="Cursor for pagination before this object ID"),
) -> JSONResponse:
    """List assistants."""
    try:
        assistants = await list_assistants(limit=limit, order=order)
        assistant_responses = [AssistantResponse(**assistant.model_dump()) for assistant in assistants]
        return JSONResponse(
            content={
                "object": "list",
                "data": [resp.model_dump() for resp in assistant_responses],
                "first_id": assistant_responses[0].id if assistant_responses else None,
                "last_id": assistant_responses[-1].id if assistant_responses else None,
                "has_more": False,
            }
        )
    except Exception as e:
        logger.error(f"Error listing assistants: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error listing assistants: {str(e)}"})


@router.get("/assistants/{assistant_id}", response_model=AssistantResponse)
async def get(assistant_id: str) -> AssistantResponse:
    """Retrieve an assistant by ID."""
    assistant = await get_assistant(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return AssistantResponse(**assistant.model_dump())


@router.post("/assistants/{assistant_id}", response_model=AssistantResponse)
async def update(assistant_id: str, request: AssistantUpdateRequest) -> AssistantResponse:
    """Update an assistant."""
    # Build updates dict
    updates = request.model_dump(exclude_unset=True)
    if "metadata" in updates:
        updates["meta"] = updates.pop("metadata")
    # tools are already dicts after model_dump()

    assistant = await update_assistant(assistant_id, updates)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return AssistantResponse(**assistant.model_dump())


@router.delete("/assistants/{assistant_id}")
async def remove(assistant_id: str) -> JSONResponse:
    """Delete an assistant."""
    try:
        success = await delete_assistant(assistant_id)
        if not success:
            return JSONResponse(status_code=404, content={"error": "Assistant not found"})
        return JSONResponse(content={"id": assistant_id, "object": "assistant.deleted", "deleted": True})
    except Exception as e:
        logger.error(f"Error deleting assistant: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error deleting assistant: {str(e)}"})
