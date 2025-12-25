from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell

from rose_server.dependencies import (
    get_db_session,
    get_readonly_db_session,
    get_spell_checker,
)
from rose_server.models.search_events import SearchEvent
from rose_server.routers.lenses import list_lens_options
from rose_server.services.search import run_search
from rose_server.views.pages.search import render_search

router = APIRouter(prefix="/v1", tags=["search"])


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    excerpt: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    index: str
    query: str
    hits: list[SearchHit]


class SearchRequest(BaseModel):
    q: str = ""
    limit: int = 10
    exact: bool = False
    lens_id: str | None = None


def _convert_hits(hits: list[Any]) -> list[SearchHit]:
    return [SearchHit.model_validate(hit, from_attributes=True) for hit in hits]


def _record_search_event(result: Any, write_session: AsyncSession) -> None:
    if not result.search_mode or not result.fts_query:
        return

    original_query = result.original_query if result.original_query != result.fts_query else None
    search_event = SearchEvent(
        event_type="search",
        search_mode=result.search_mode,
        query=result.fts_query,
        original_query=original_query,
        result_count=len(result.hits),
    )
    write_session.add(search_event)


@router.get("/search")
async def search_messages(
    request: Request,
    q: str = Query("", description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    exact: bool = Query(False, description="Skip spell correction"),
    lens_id: str | None = Query(None, description="Filter by lens id"),
    read_session: AsyncSession = Depends(get_readonly_db_session),
    write_session: AsyncSession = Depends(get_db_session),
    spell_checker: SymSpell | None = Depends(get_spell_checker),
) -> Any:
    result = await run_search(
        read_session=read_session,
        q=q,
        limit=limit,
        exact=exact,
        lens_id=lens_id,
        spell_checker=spell_checker,
    )

    _record_search_event(result, write_session)

    response_data = SearchResponse(
        index="messages",
        query=result.query,
        hits=_convert_hits(result.hits),
    )

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        lenses = await list_lens_options(read_session)
        hits = _convert_hits(result.hits)
        display_query = result.original_query or result.query
        return HtpyResponse(
            render_search(
                query=display_query,
                hits=hits,
                corrected_query=result.corrected_query,
                original_query=result.original_query,
                lenses=lenses,
                selected_lens_id=result.lens_id,
            )
        )

    return response_data


@router.post("/search")
async def search_messages_post(
    request: Request,
    body: SearchRequest,
    read_session: AsyncSession = Depends(get_readonly_db_session),
    write_session: AsyncSession = Depends(get_db_session),
    spell_checker: SymSpell | None = Depends(get_spell_checker),
) -> Any:
    result = await run_search(
        read_session=read_session,
        q=body.q,
        limit=body.limit,
        exact=body.exact,
        lens_id=body.lens_id,
        spell_checker=spell_checker,
    )

    _record_search_event(result, write_session)

    response_data = SearchResponse(
        index="messages",
        query=result.query,
        hits=_convert_hits(result.hits),
    )

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        lenses = await list_lens_options(read_session)
        hits = _convert_hits(result.hits)
        display_query = result.original_query or result.query
        return HtpyResponse(
            render_search(
                query=display_query,
                hits=hits,
                corrected_query=result.corrected_query,
                original_query=result.original_query,
                lenses=lenses,
                selected_lens_id=result.lens_id,
            )
        )

    return response_data
