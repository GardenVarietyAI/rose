import time
from typing import Annotated, Any

import frontmatter
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select, update
from starlette.responses import PlainTextResponse, RedirectResponse

from rose_server.dependencies import get_db_session, get_readonly_db_session
from rose_server.models.message_types import LensMessage, LensMeta
from rose_server.models.messages import Message
from rose_server.schemas.lenses import CreateLensRequest, LensFrontmatter
from rose_server.services.lenses import (
    get_latest_lens_revision,
    get_lens_message,
    list_lenses_messages,
    resolve_lens_uuid_to_root,
    validate_at_name_unique,
)

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


@router.get("/lenses", response_model=None)
async def list_lenses(
    session: AsyncSession = Depends(get_readonly_db_session),
) -> list[Message]:
    return await list_lenses_messages(session)


@router.post("/lenses", response_model=None)
async def create_lens(
    body: CreateLensRequest,
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
    return message


@router.post("/lenses/upload", response_model=None)
async def upload_lens(
    file: Annotated[UploadFile, File()],
    session: AsyncSession = Depends(get_db_session),
) -> Any:
    file_content = await file.read()
    try:
        parsed = frontmatter.loads(file_content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse frontmatter: {e}") from e

    try:
        metadata = LensFrontmatter.model_validate(parsed.metadata)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid frontmatter: {e}") from e

    lens_uuid = metadata.uuid
    at_name = metadata.at_name
    label = metadata.label
    system_prompt = parsed.content.strip()

    if not system_prompt:
        raise HTTPException(status_code=400, detail="Lens content cannot be empty")

    root_id = await resolve_lens_uuid_to_root(session, lens_uuid)
    if root_id is None:
        if not await validate_at_name_unique(session, at_name):
            raise HTTPException(status_code=400, detail=f"Lens with at_name '{at_name}' already exists")

        message = Message(
            uuid=lens_uuid,
            thread_id=None,
            role="system",
            content=system_prompt,
            model=None,
        )
        message.meta = LensMeta(
            at_name=at_name,
            label=label,
            root_message_id=lens_uuid,
            parent_message_id=None,
        ).model_dump()
        session.add(message)
        return message

    current_lens = await get_latest_lens_revision(session, root_id)
    if current_lens is None:
        raise HTTPException(status_code=404, detail="Lens not found")

    if (
        current_lens.meta
        and current_lens.meta.get("at_name") == at_name
        and current_lens.meta.get("label") == label
        and current_lens.content == system_prompt
    ):
        return current_lens

    if not await validate_at_name_unique(session, at_name, exclude_root_id=root_id):
        raise HTTPException(status_code=400, detail=f"Lens with at_name '{at_name}' already exists")

    new_revision = Message(thread_id=None, role="system", content=system_prompt, model=None)
    new_revision.meta = LensMeta(
        at_name=at_name,
        label=label,
        root_message_id=root_id,
        parent_message_id=current_lens.uuid,
    ).model_dump()
    session.add(new_revision)
    return new_revision


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
    if current_lens.content is None:
        raise HTTPException(status_code=400, detail="Lens missing content")

    try:
        current_meta = LensMeta.model_validate(current_lens.meta)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail="Lens missing meta") from e

    current_body = CreateLensRequest(
        at_name=current_meta.at_name,
        label=current_meta.label,
        system_prompt=current_lens.content,
    )
    if current_body == body:
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse(url=f"/v1/lenses/{root_id}/edit", status_code=303)
        return current_lens

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


@router.get("/lenses/{lens_id}/download", response_model=None)
async def download_lens(
    lens_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> PlainTextResponse:
    lens = await get_lens_message(session, lens_id)
    if lens is None:
        raise HTTPException(status_code=404, detail="Lens not found")

    if lens.meta is None:
        raise HTTPException(status_code=400, detail="Lens missing meta")
    if lens.content is None:
        raise HTTPException(status_code=400, detail="Lens missing content")

    try:
        lens_meta = LensMeta.model_validate(lens.meta)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail="Lens missing meta") from e

    root_id = lens_meta.root_message_id
    post = frontmatter.Post(
        content=lens.content,
        uuid=root_id,
        at_name=lens_meta.at_name,
        label=lens_meta.label,
    )
    content = frontmatter.dumps(post)

    filename = f"{lens_meta.at_name}.md"
    return PlainTextResponse(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
