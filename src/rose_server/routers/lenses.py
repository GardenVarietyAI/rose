import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from htpy.starlette import HtpyResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select, update
from starlette.responses import RedirectResponse

from rose_server.dependencies import get_db_session, get_readonly_db_session
from rose_server.models.message_types import LensMessage, LensMeta
from rose_server.models.messages import Message
from rose_server.schemas.lenses import CreateLensRequest
from rose_server.services.lenses import (
    get_latest_lens_revision,
    get_lens_message,
    list_lenses_messages,
    resolve_lens_uuid_to_root,
    validate_at_name_unique,
)
from rose_server.views.pages.lens import render_lens_form_page, render_lenses_page

router = APIRouter(prefix="/v1", tags=["lenses"])


async def list_lens_options(session: AsyncSession) -> list[tuple[str, str]]:
    lenses = await list_lenses_messages(session)
    options: list[tuple[str, str]] = []
    for lens in lenses:
        try:
            lens_message = LensMessage(message=lens)
        except ValidationError as e:
            raise HTTPException(status_code=500, detail="Invalid lens message") from e
        options.append((lens_message.lens_id, lens_message.label))
    return options


async def list_lens_autocomplete_options(session: AsyncSession) -> list[tuple[str, str]]:
    lenses = await list_lenses_messages(session)
    options: list[tuple[str, str]] = []
    for lens in lenses:
        try:
            lens_message = LensMessage(message=lens)
        except ValidationError as e:
            raise HTTPException(status_code=500, detail="Invalid lens message") from e
        options.append((lens_message.at_name, lens_message.label))
    return options


async def list_lens_picker_options(session: AsyncSession) -> list[tuple[str, str, str]]:
    lenses = await list_lenses_messages(session)
    options: list[tuple[str, str, str]] = []
    for lens in lenses:
        try:
            lens_message = LensMessage(message=lens)
        except ValidationError as e:
            raise HTTPException(status_code=500, detail="Invalid lens message") from e
        options.append((lens_message.lens_id, lens_message.at_name, lens_message.label))
    return options


@router.get("/lenses", response_model=None)
async def list_lenses(
    request: Request,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> Any:
    lenses = await list_lenses_messages(session)

    if "text/html" in request.headers.get("accept", ""):
        return HtpyResponse(render_lenses_page(lenses=lenses))

    return lenses


@router.get("/lenses/create", response_model=None)
async def create_lens_page() -> HtpyResponse:
    return HtpyResponse(render_lens_form_page(lens=None))


@router.get("/lenses/{lens_id}/edit", response_model=None)
async def edit_lens_page(
    lens_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> HtpyResponse:
    lens = await get_lens_message(session, lens_id)
    if lens is None:
        raise HTTPException(status_code=404, detail="Lens not found")
    return HtpyResponse(render_lens_form_page(lens=lens))


@router.post("/lenses", response_model=None)
async def create_lens(
    request: Request,
    body: CreateLensRequest = Depends(CreateLensRequest.as_form),
    session: AsyncSession = Depends(get_db_session),
) -> Any:
    if not await validate_at_name_unique(session, body.at_name):
        raise HTTPException(status_code=400, detail=f"Lens with at_name '{body.at_name}' already exists")

    message = Message(
        thread_id=None,
        role="system",
        content=body.system_prompt,
        model=None,
    )
    message.meta = LensMeta(
        at_name=body.at_name,
        label=body.label,
        root_message_id=message.uuid,
        parent_message_id=None,
    ).model_dump()
    session.add(message)

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url=f"/v1/lenses/{message.uuid}/edit", status_code=303)

    return message


@router.post("/lenses/{lens_id}", response_model=None)
async def update_lens(
    request: Request,
    lens_id: str,
    body: CreateLensRequest = Depends(CreateLensRequest.as_form),
    session: AsyncSession = Depends(get_db_session),
) -> Any:
    root_id = await resolve_lens_uuid_to_root(session, lens_id)
    if root_id is None:
        raise HTTPException(status_code=404, detail="Lens not found")

    current_lens = await get_latest_lens_revision(session, root_id)
    if current_lens is None:
        raise HTTPException(status_code=404, detail="Lens not found")

    if current_lens.meta is None:
        raise HTTPException(status_code=400, detail="Lens missing meta")

    if not await validate_at_name_unique(session, body.at_name, exclude_root_id=root_id):
        raise HTTPException(status_code=400, detail=f"Lens with at_name '{body.at_name}' already exists")

    new_revision = Message(
        thread_id=None,
        role="system",
        content=body.system_prompt,
        model=None,
    )
    new_revision.meta = LensMeta(
        at_name=body.at_name,
        label=body.label,
        root_message_id=root_id,
        parent_message_id=current_lens.uuid,
    ).model_dump()
    session.add(new_revision)

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url=f"/v1/lenses/{root_id}/edit", status_code=303)

    return new_revision


@router.get("/lenses/{lens_id}/revisions", response_model=None)
async def get_lens_revisions(
    lens_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> list[Message]:
    root_id = await resolve_lens_uuid_to_root(session, lens_id)
    if root_id is None:
        raise HTTPException(status_code=404, detail="Lens not found")

    result = await session.execute(
        select(Message)
        .where(col(Message.object) == "lens", col(Message.root_message_id) == root_id)
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
    )
    return list(result.scalars().all())


@router.post("/lenses/{lens_id}/delete", response_model=None)
async def delete_lens(lens_id: str, session: AsyncSession = Depends(get_db_session)) -> RedirectResponse:
    root_id = await resolve_lens_uuid_to_root(session, lens_id)
    if root_id is None:
        raise HTTPException(status_code=404, detail="Lens not found")

    await session.execute(
        update(Message)
        .where(col(Message.object) == "lens", col(Message.root_message_id) == root_id)
        .values(deleted_at=int(time.time()))
    )
    return RedirectResponse(url="/v1/lenses", status_code=303)
