from fastapi import APIRouter, Depends
from htpy.starlette import HtpyResponse
from sqlalchemy.ext.asyncio import AsyncSession

from rose_server.dependencies import get_db_session
from rose_server.schemas.importer import ImportRequest, ImportResponse
from rose_server.services.importer import import_conversations
from rose_server.views.pages.importer import render_import_page

router = APIRouter(prefix="/v1", tags=["import"])


@router.get("/import")
async def import_page() -> HtpyResponse:
    return HtpyResponse(render_import_page())


@router.post("/import")
async def import_file(
    body: ImportRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ImportResponse:
    return await import_conversations(body.conversations, body.import_source, session)
