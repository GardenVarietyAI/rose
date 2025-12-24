from typing import Any

from fastapi import APIRouter
from htpy.starlette import HtpyResponse

from rose_server.views.pages.nav import render_nav

router = APIRouter(prefix="/v1", tags=["nav"])


@router.get("/nav")
async def nav_page() -> Any:
    return HtpyResponse(render_nav())
