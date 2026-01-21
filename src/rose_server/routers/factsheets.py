import time
from typing import Annotated, Any

import frontmatter
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select, update
from starlette.responses import PlainTextResponse, RedirectResponse

from rose_server.dependencies import get_db_session, get_readonly_db_session
from rose_server.models.message_types import FactsheetMeta
from rose_server.models.messages import Message
from rose_server.schemas.factsheets import CreateFactsheetRequest, FactsheetFrontmatter
from rose_server.services.factsheets import (
    get_factsheet_message,
    get_latest_factsheet_revision,
    list_factsheets_messages,
    resolve_factsheet_uuid_to_root,
    validate_hashtag_unique,
)

router = APIRouter(prefix="/v1", tags=["factsheets"])


@router.get("/factsheets", response_model=None)
async def list_factsheets(
    session: AsyncSession = Depends(get_readonly_db_session),
) -> list[Message]:
    return await list_factsheets_messages(session)


@router.get("/factsheets/{factsheet_id}", response_model=None)
async def get_factsheet(
    factsheet_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> Any:
    factsheet = await get_factsheet_message(session, factsheet_id)
    if factsheet is None:
        raise HTTPException(status_code=404, detail="Fact sheet not found")
    return factsheet


@router.post("/factsheets", response_model=None)
async def create_factsheet(
    body: CreateFactsheetRequest,
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
    return message


@router.post("/factsheets/upload", response_model=None)
async def upload_factsheet(
    file: Annotated[UploadFile, File()],
    session: AsyncSession = Depends(get_db_session),
) -> Any:
    file_content = await file.read()
    try:
        parsed = frontmatter.loads(file_content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse frontmatter: {e}") from e

    try:
        metadata = FactsheetFrontmatter.model_validate(parsed.metadata)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid frontmatter: {e}") from e

    factsheet_uuid = metadata.uuid
    tag = metadata.tag
    title = metadata.title
    factsheet_body = parsed.content.strip()

    if not factsheet_body:
        raise HTTPException(status_code=400, detail="Factsheet content cannot be empty")

    root_id = await resolve_factsheet_uuid_to_root(session, factsheet_uuid)
    if root_id is None:
        if not await validate_hashtag_unique(session, hashtag=tag):
            raise HTTPException(status_code=400, detail=f"Fact sheet with tag '{tag}' already exists")

        message = Message(
            uuid=factsheet_uuid,
            thread_id=None,
            role="system",
            content=factsheet_body,
            model=None,
        )
        message.meta = FactsheetMeta(
            tag=tag,
            title=title,
            root_message_id=factsheet_uuid,
            parent_message_id=None,
        ).model_dump()
        session.add(message)
        return message

    current_factsheet = await get_latest_factsheet_revision(session, root_id)
    if current_factsheet is None:
        raise HTTPException(status_code=404, detail="Fact sheet not found")

    if (
        current_factsheet.meta
        and current_factsheet.meta.get("tag") == tag
        and current_factsheet.meta.get("title") == title
        and current_factsheet.content == factsheet_body
    ):
        return current_factsheet

    if not await validate_hashtag_unique(session, hashtag=tag, exclude_root_id=root_id):
        raise HTTPException(status_code=400, detail=f"Fact sheet with tag '{tag}' already exists")

    new_revision = Message(thread_id=None, role="system", content=factsheet_body, model=None)
    new_revision.meta = FactsheetMeta(
        tag=tag,
        title=title,
        root_message_id=root_id,
        parent_message_id=current_factsheet.uuid,
    ).model_dump()
    session.add(new_revision)
    return new_revision


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


@router.get("/factsheets/{factsheet_id}/download", response_model=None)
async def download_factsheet(
    factsheet_id: str,
    session: AsyncSession = Depends(get_readonly_db_session),
) -> PlainTextResponse:
    factsheet = await get_factsheet_message(session, factsheet_id)
    if factsheet is None:
        raise HTTPException(status_code=404, detail="Fact sheet not found")

    if factsheet.meta is None:
        raise HTTPException(status_code=400, detail="Fact sheet missing meta")
    if factsheet.content is None:
        raise HTTPException(status_code=400, detail="Fact sheet missing content")

    try:
        factsheet_meta = FactsheetMeta.model_validate(factsheet.meta)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail="Fact sheet missing meta") from e

    root_id = factsheet_meta.root_message_id
    post = frontmatter.Post(
        content=factsheet.content,
        uuid=root_id,
        tag=factsheet_meta.tag,
        title=factsheet_meta.title,
    )
    content = frontmatter.dumps(post)

    filename = f"{factsheet_meta.tag}.md"
    return PlainTextResponse(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
