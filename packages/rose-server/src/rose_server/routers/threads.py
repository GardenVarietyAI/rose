import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from rose_server.models.messages import Message
from sqlmodel import col, select

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["threads"])

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class ThreadResponse(BaseModel):
    thread_id: str
    messages: list[Message]


@router.get("/threads/{thread_id}", response_model=None)
async def get_thread(
    request: Request,
    thread_id: str,
) -> ThreadResponse | Response:
    async with request.app.state.get_db_session(read_only=True) as session:
        statement = select(Message).where(Message.thread_id == thread_id).order_by(col(Message.created_at))
        result = await session.execute(statement)
        messages = list(result.scalars().all())

    if not messages:
        raise HTTPException(status_code=404, detail="Thread not found")

    response_data = ThreadResponse(
        thread_id=thread_id,
        messages=messages,
    )

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return templates.TemplateResponse(
            "thread.html",
            {"request": request, "thread_id": thread_id, "messages": messages},
        )

    return response_data
