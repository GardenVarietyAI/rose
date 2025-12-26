import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel, StringConstraints, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select, update
from starlette.responses import RedirectResponse

from rose_server.dependencies import get_db_session, get_readonly_db_session
from rose_server.models.messages import Message
from rose_server.views.pages.lens import render_lens_form_page, render_lenses_page

router = APIRouter(prefix="/v1", tags=["lenses"])

LENS_OBJECT = "lens"


class CreateLensRequest(BaseModel):
    at_name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9]+$"),
    ]
    label: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    system_prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    @field_validator("at_name")
    @classmethod
    def validate_at_name(cls, value: str) -> str:
        return value.lower()

    @classmethod
    def as_form(
        cls,
        at_name: str = Form(...),
        label: str = Form(...),
        system_prompt: str = Form(...),
    ) -> "CreateLensRequest":
        return cls(at_name=at_name, label=label, system_prompt=system_prompt)


async def list_lenses_messages(session: AsyncSession) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(col(Message.object) == LENS_OBJECT)
        .where(col(Message.deleted_at).is_(None))
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
    )
    return list(result.scalars().all())


async def list_lens_options(session: AsyncSession) -> list[tuple[str, str]]:
    lenses = await list_lenses_messages(session)
    options: list[tuple[str, str]] = []
    for lens in lenses:
        meta = lens.meta
        if meta is None:
            raise HTTPException(status_code=500, detail="Lens missing meta")
        options.append((lens.uuid, str(meta["label"])))
    return options


async def list_lens_autocomplete_options(session: AsyncSession) -> list[tuple[str, str]]:
    lenses = await list_lenses_messages(session)
    options: list[tuple[str, str]] = []
    for lens in lenses:
        meta = lens.meta
        if meta is None:
            raise HTTPException(status_code=500, detail="Lens missing meta")
        options.append((str(meta["at_name"]), str(meta["label"])))
    return options


async def list_lens_picker_options(session: AsyncSession) -> list[tuple[str, str, str]]:
    lenses = await list_lenses_messages(session)
    options: list[tuple[str, str, str]] = []
    for lens in lenses:
        meta = lens.meta
        if meta is None:
            raise HTTPException(status_code=500, detail="Lens missing meta")
        options.append((lens.uuid, str(meta["at_name"]), str(meta["label"])))
    return options


async def get_lens_message(session: AsyncSession, lens_id: str) -> Message | None:
    result = await session.execute(
        select(Message)
        .where(col(Message.uuid) == lens_id)
        .where(col(Message.object) == LENS_OBJECT)
        .where(col(Message.deleted_at).is_(None))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_lens_message_by_at_name(session: AsyncSession, at_name: str) -> Message | None:
    result = await session.execute(
        select(Message)
        .where(col(Message.object) == LENS_OBJECT)
        .where(col(Message.at_name) == at_name)
        .where(col(Message.deleted_at).is_(None))
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


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
    message = Message(
        thread_id=None,
        role="system",
        content=body.system_prompt,
        model=None,
    )
    message.meta = {
        "object": LENS_OBJECT,
        "lens_id": message.uuid,
        "label": body.label,
        "at_name": body.at_name,
    }
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
    lens = await get_lens_message(session, lens_id)
    if lens is None:
        raise HTTPException(status_code=404, detail="Lens not found")

    if lens.meta is None:
        raise HTTPException(status_code=400, detail="Lens missing meta")

    meta = dict(lens.meta)
    meta.update(
        {
            "object": LENS_OBJECT,
            "lens_id": lens_id,
            "label": body.label,
            "at_name": body.at_name,
        }
    )
    lens.meta = meta
    lens.content = body.system_prompt

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url=f"/v1/lenses/{lens_id}/edit", status_code=303)

    return lens


@router.post("/lenses/{lens_id}/delete", response_model=None)
async def delete_lens(lens_id: str, session: AsyncSession = Depends(get_db_session)) -> RedirectResponse:
    await session.execute(
        update(Message)
        .where(col(Message.uuid) == lens_id)
        .where(col(Message.object) == LENS_OBJECT)
        .values(deleted_at=int(time.time()))
    )
    return RedirectResponse(url="/v1/lenses", status_code=303)
