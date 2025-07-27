"""API router for threads endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse

from rose_server.database import current_timestamp
from rose_server.entities.messages import Message
from rose_server.schemas.messages import MessageCreateRequest, MessageResponse
from rose_server.threads.messages.store import create_message, get_message, get_messages

router = APIRouter(prefix="/{thread_id}/messages")
logger = logging.getLogger(__name__)


@router.post("", response_model=MessageResponse)
async def create(
    thread_id: str,
    request: MessageCreateRequest = Body(...),
    enable_embedding: bool = Query(default=True, description="Whether to embed message to vector store"),
) -> MessageResponse:
    """Create a message in a thread with optional vector embedding."""
    try:
        if isinstance(request.content, str):
            content = [
                {
                    "type": "text",
                    "text": {
                        "value": request.content,
                        "annotations": [],
                    },
                }
            ]
        else:
            content = request.content

        message = Message(
            thread_id=thread_id,
            role=request.role,
            content=content,
            created_at=current_timestamp(),
            meta=request.metadata,
            completed_at=current_timestamp(),
        )

        message = await create_message(message)
        if not message:
            raise HTTPException(status_code=500, detail="Failed to create message")

        return MessageResponse(**message.model_dump())
    except Exception as e:
        logger.error(f"Error creating message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating message: {str(e)}")


@router.get("")
async def list_messages(
    thread_id: str,
    limit: int = Query(default=20, description="Number of messages to retrieve"),
    order: str = Query(default="desc", description="Sort order (asc or desc)"),
    after: Optional[str] = Query(default=None, description="Cursor for pagination"),
    before: Optional[str] = Query(default=None, description="Cursor for pagination"),
) -> JSONResponse:
    """List messages in a thread."""
    try:
        messages = await get_messages(thread_id, limit=limit, order=order)
        message_data = [MessageResponse(**msg.model_dump()).model_dump() for msg in messages]
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
        raise HTTPException(status_code=500, detail=f"Error listing messages: {str(e)}")


@router.get("/{message_id}", response_model=MessageResponse)
async def get_message_for_thread(thread_id: str, message_id: str) -> MessageResponse:
    """Retrieve a specific message from a thread."""
    try:
        message = await get_message(thread_id, message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        return MessageResponse(**message.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving message: {str(e)}")
