"""API router for assistants endpoints."""
import logging
from typing import Optional
from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse
from rose_server.assistants.store import get_assistant_store
from rose_server.schemas.assistants import CreateAssistantRequest, UpdateAssistantRequest
from rose_server.tools import Tool, validate_tools
router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)
@router.post("/assistants")

async def create_assistant(request: CreateAssistantRequest = Body(...)) -> JSONResponse:
    """Create a new assistant."""
    try:
        if request.tools:
            try:
                tool_dicts = []
                for tool in request.tools:
                    if isinstance(tool, dict):
                        tool_dicts.append(tool)
                    else:
                        tool_dicts.append(tool.model_dump() if hasattr(tool, 'model_dump') else tool.dict())
                validate_tools(tool_dicts)
                request.tools = [Tool(**tool) if isinstance(tool, dict) else tool for tool in request.tools]
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid tools configuration: {str(e)}"})
        assistant_store = get_assistant_store()
        assistant = await assistant_store.create_assistant(request)
        return JSONResponse(content=assistant.model_dump())
    except Exception as e:
        logger.error(f"Error creating assistant: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error creating assistant: {str(e)}"})
@router.get("/assistants")

async def list_assistants(
    limit: int = Query(default=20, description="Number of assistants to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
    after: Optional[str] = Query(default=None, description="Cursor for pagination after this object ID"),
    before: Optional[str] = Query(default=None, description="Cursor for pagination before this object ID"),
) -> JSONResponse:
    """List assistants."""
    try:
        assistant_store = get_assistant_store()
        assistants = await assistant_store.list_assistants(limit=limit, order=order)
        assistant_data = [assistant.model_dump() for assistant in assistants]
        return JSONResponse(
            content={
                "object": "list",
                "data": assistant_data,
                "first_id": assistant_data[0]["id"] if assistant_data else None,
                "last_id": assistant_data[-1]["id"] if assistant_data else None,
                "has_more": False,
            }
        )
    except Exception as e:
        logger.error(f"Error listing assistants: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error listing assistants: {str(e)}"})
@router.get("/assistants/{assistant_id}")

async def get_assistant(assistant_id: str) -> JSONResponse:
    """Retrieve an assistant by ID."""
    try:
        assistant_store = get_assistant_store()
        assistant = await assistant_store.get_assistant(assistant_id)
        if not assistant:
            return JSONResponse(status_code=404, content={"error": "Assistant not found"})
        return JSONResponse(content=assistant.dict())
    except Exception as e:
        logger.error(f"Error retrieving assistant: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving assistant: {str(e)}"})
@router.post("/assistants/{assistant_id}")

async def update_assistant(assistant_id: str, request: UpdateAssistantRequest = Body(...)) -> JSONResponse:
    """Update an assistant."""
    try:
        if request.tools is not None:
            try:
                tool_dicts = []
                for tool in request.tools:
                    if isinstance(tool, dict):
                        tool_dicts.append(tool)
                    else:
                        tool_dicts.append(tool.model_dump() if hasattr(tool, 'model_dump') else tool.dict())
                validate_tools(tool_dicts)
                request.tools = [Tool(**tool) if isinstance(tool, dict) else tool for tool in request.tools]
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid tools configuration: {str(e)}"})
        assistant_store = get_assistant_store()
        assistant = await assistant_store.update_assistant(assistant_id, request)
        if not assistant:
            return JSONResponse(status_code=404, content={"error": "Assistant not found"})
        return JSONResponse(content=assistant.dict())
    except Exception as e:
        logger.error(f"Error updating assistant: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error updating assistant: {str(e)}"})
@router.delete("/assistants/{assistant_id}")

async def delete_assistant(assistant_id: str) -> JSONResponse:
    """Delete an assistant."""
    try:
        assistant_store = get_assistant_store()
        success = await assistant_store.delete_assistant(assistant_id)
        if not success:
            return JSONResponse(status_code=404, content={"error": "Assistant not found"})
        return JSONResponse(content={"id": assistant_id, "object": "assistant.deleted", "deleted": True})
    except Exception as e:
        logger.error(f"Error deleting assistant: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error deleting assistant: {str(e)}"})