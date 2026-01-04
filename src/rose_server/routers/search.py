from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from htpy.starlette import HtpyResponse
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell

from rose_server.dependencies import (
    get_db_session,
    get_readonly_db_session,
    get_spell_checker,
)
from rose_server.models.search_events import SearchEvent
from rose_server.routers.lenses import list_lens_picker_options
from rose_server.schemas.search import SearchHit, SearchRequest, SearchResponse
from rose_server.services.search import SearchResult, run_search
from rose_server.views.pages.search import render_search, render_search_root

router = APIRouter(prefix="/v1", tags=["search"])


def _convert_hits(hits: list[Any]) -> list[SearchHit]:
    return [SearchHit.model_validate(hit, from_attributes=True) for hit in hits]


def _record_search_event(result: SearchResult, write_session: AsyncSession) -> None:
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

    converted_hits = _convert_hits(result.hits)
    response_data = SearchResponse(
        index="messages",
        query=result.query,
        hits=converted_hits,
    )

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        lenses = await list_lens_picker_options(read_session)
        return HtpyResponse(
            render_search(
                query=result.query,
                hits=converted_hits,
                corrected_query=result.corrected_query,
                original_query=result.original_query,
                lenses=lenses,
                selected_lens_id=result.lens_id,
                limit=limit,
                exact=exact,
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

    converted_hits = _convert_hits(result.hits)
    response_data = SearchResponse(
        index="messages",
        query=result.query,
        hits=converted_hits,
    )

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        lenses = await list_lens_picker_options(read_session)
        return HtpyResponse(
            render_search(
                query=result.query,
                hits=converted_hits,
                corrected_query=result.corrected_query,
                original_query=result.original_query,
                lenses=lenses,
                selected_lens_id=result.lens_id,
                limit=body.limit,
                exact=body.exact,
            )
        )

    return response_data


@router.post("/search/fragment")
async def search_fragment(
    body: SearchRequest,
    read_session: AsyncSession = Depends(get_readonly_db_session),
    write_session: AsyncSession = Depends(get_db_session),
    spell_checker: SymSpell | None = Depends(get_spell_checker),
) -> HtpyResponse:
    result = await run_search(
        read_session=read_session,
        q=body.q,
        limit=body.limit,
        exact=body.exact,
        lens_id=body.lens_id,
        spell_checker=spell_checker,
    )

    _record_search_event(result, write_session)

    lenses = await list_lens_picker_options(read_session)
    converted_hits = _convert_hits(result.hits)

    return HtpyResponse(
        render_search_root(
            query=result.query,
            hits=converted_hits,
            corrected_query=result.corrected_query,
            original_query=result.original_query,
            lenses=lenses,
            selected_lens_id=result.lens_id,
            hits_count=len(converted_hits),
        )
    )
