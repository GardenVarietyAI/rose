import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from htpy.starlette import HtpyResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select, update
from starlette.responses import RedirectResponse

from rose_server.dependencies import get_db_session, get_readonly_db_session
from rose_server.models.message_types import FactsheetMessage, FactsheetMeta
from rose_server.models.messages import Message
from rose_server.schemas.factsheets import CreateFactsheetRequest
from rose_server.services.factsheets import (
    get_factsheet_message,
    get_latest_factsheet_revision,
    list_factsheets_messages,
    resolve_factsheet_uuid_to_root,
    validate_hashtag_unique,
)
from rose_server.views.pages.factsheet import render_factsheet_form_page, render_factsheets_page

router = APIRouter(prefix="/v1", tags=["factsheets"])


async def list_factsheet_picker_options(session: AsyncSession) -> list[tuple[str, str, str]]:
    factsheets = await list_factsheets_messages(session)
    options: list[tuple[str, str, str]] = []
    for factsheet in factsheets:
        try:
            factsheet_message = FactsheetMessage(message=factsheet)
        except ValidationError as e:
            raise HTTPException(status_code=500, detail="Invalid fact sheet") from e
        options.append((factsheet_message.factsheet_id, factsheet_message.tag, factsheet_message.title))
    return options


@router.get("/factsheets", response_model=None)
async def list_factsheets(
    request: Request,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> Any:
    factsheets = await list_factsheets_messages(session)

    if "text/html" in request.headers.get("accept", ""):
        return HtpyResponse(render_factsheets_page(factsheets=factsheets))

    return factsheets


@router.get("/factsheets/create", response_model=None)
async def create_factsheet_page() -> HtpyResponse:
    return HtpyResponse(render_factsheet_form_page(factsheet=None))


@router.get("/factsheets/{factsheet_id}/edit", response_model=None)
async def edit_factsheet_page(
    factsheet_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> HtpyResponse:
    factsheet = await get_factsheet_message(session, factsheet_id)
    if factsheet is None:
        raise HTTPException(status_code=404, detail="Fact sheet not found")
    return HtpyResponse(render_factsheet_form_page(factsheet=factsheet))


@router.get("/factsheets/{factsheet_id}", response_model=None)
async def get_factsheet(
    factsheet_id: str,
    request: Request,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> Any:
    factsheet = await get_factsheet_message(session, factsheet_id)
    if factsheet is None:
        raise HTTPException(status_code=404, detail="Fact sheet not found")
    if "text/html" in request.headers.get("accept", ""):
        return HtpyResponse(render_factsheet_form_page(factsheet=factsheet))
    return factsheet


@router.post("/factsheets", response_model=None)
async def create_factsheet(
    request: Request,
    body: CreateFactsheetRequest = Depends(CreateFactsheetRequest.as_form),
    session: AsyncSession = Depends(get_db_session),
) -> Any:
    if not await validate_hashtag_unique(session, hashtag=body.tag):
        raise HTTPException(status_code=400, detail=f"Fact sheet with tag '{body.tag}' already exists")

    message = Message(thread_id=None, role="system", content=body.body, model=None)
    message.meta = FactsheetMeta(
        tag=body.tag,
        title=body.title,
        root_message_id=message.uuid,
        parent_message_id=None,
    ).model_dump()
    session.add(message)

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url=f"/v1/factsheets/{message.uuid}/edit", status_code=303)

    return message


@router.post("/factsheets/{factsheet_id}", response_model=None)
async def update_factsheet(
    request: Request,
    factsheet_id: str,
    body: CreateFactsheetRequest = Depends(CreateFactsheetRequest.as_form),
    session: AsyncSession = Depends(get_db_session),
) -> Any:
    root_id = await resolve_factsheet_uuid_to_root(session, factsheet_id)
    if root_id is None:
        raise HTTPException(status_code=404, detail="Fact sheet not found")

    current = await get_latest_factsheet_revision(session, root_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Fact sheet not found")

    if current.meta is None:
        raise HTTPException(status_code=400, detail="Fact sheet missing meta")
    if current.content is None:
        raise HTTPException(status_code=400, detail="Fact sheet missing content")

    try:
        current_meta = FactsheetMeta.model_validate(current.meta)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail="Fact sheet missing meta") from e

    current_body = CreateFactsheetRequest(tag=current_meta.tag, title=current_meta.title, body=current.content)
    if current_body == body:
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse(url=f"/v1/factsheets/{root_id}/edit", status_code=303)
        return current

    if not await validate_hashtag_unique(session, hashtag=body.tag, exclude_root_id=root_id):
        raise HTTPException(status_code=400, detail=f"Fact sheet with tag '{body.tag}' already exists")

    new_revision = Message(thread_id=None, role="system", content=body.body, model=None)
    new_revision.meta = FactsheetMeta(
        tag=body.tag,
        title=body.title,
        root_message_id=root_id,
        parent_message_id=current.uuid,
    ).model_dump()
    session.add(new_revision)

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url=f"/v1/factsheets/{root_id}/edit", status_code=303)

    return new_revision


@router.get("/factsheets/{factsheet_id}/revisions", response_model=None)
async def get_factsheet_revisions(
    factsheet_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> list[Message]:
    root_id = await resolve_factsheet_uuid_to_root(session, factsheet_id)
    if root_id is None:
        raise HTTPException(status_code=404, detail="Fact sheet not found")

    result = await session.execute(
        select(Message)
        .where(col(Message.object) == "factsheet", col(Message.root_message_id) == root_id)
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
    )
    return list(result.scalars().all())


@router.post("/factsheets/{factsheet_id}/delete", response_model=None)
async def delete_factsheet(factsheet_id: str, session: AsyncSession = Depends(get_db_session)) -> RedirectResponse:
    root_id = await resolve_factsheet_uuid_to_root(session, factsheet_id)
    if root_id is None:
        raise HTTPException(status_code=404, detail="Fact sheet not found")

    await session.execute(
        update(Message)
        .where(col(Message.object) == "factsheet", col(Message.root_message_id) == root_id)
        .values(deleted_at=int(time.time()))
    )
    return RedirectResponse(url="/v1/factsheets", status_code=303)
