import base64
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell

from rose_server.dependencies import (
    get_db_session,
    get_readonly_db_session,
    get_spell_checker,
)
from rose_server.models.search_events import SearchEvent
from rose_server.routers.factsheets import list_factsheet_picker_options
from rose_server.routers.lenses import list_lens_picker_options
from rose_server.schemas.search import SearchHit, SearchRequest, SearchResponse
from rose_server.services.search import SearchResult, run_search
from rose_server.views.pages.search import render_search, render_search_root

router = APIRouter(prefix="/v1", tags=["search"])


class StructuredSearchQuery(BaseModel):
    lens_ids: list[str] = []
    factsheet_ids: list[str] = []
    limit: int | None = None
    exact: bool | None = None


def _first_non_empty(values: list[str]) -> str | None:
    for candidate in values:
        if candidate and candidate.strip():
            return candidate.strip()
    return None


def _decode_sq(value: str) -> dict[str, Any]:
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        decoded = raw.decode("utf-8")
        parsed = json.loads(decoded)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=400, detail="Invalid sq") from e
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Invalid sq")
    return parsed


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
    sq: str | None = Query(None, description="Structured query JSON"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    exact: bool = Query(False, description="Skip spell correction"),
    read_session: AsyncSession = Depends(get_readonly_db_session),
    write_session: AsyncSession = Depends(get_db_session),
    spell_checker: SymSpell | None = Depends(get_spell_checker),
) -> Any:
    lens_id: str | None = None
    factsheet_ids: list[str] = []
    if sq:
        try:
            parsed = _decode_sq(sq)
            structured = StructuredSearchQuery.model_validate(parsed)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail="Invalid sq") from e
        lens_id = _first_non_empty(structured.lens_ids)
        factsheet_ids = [factsheet_id for factsheet_id in structured.factsheet_ids if factsheet_id]
        if structured.limit is not None:
            limit = structured.limit
        if structured.exact is not None:
            exact = structured.exact

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
        factsheets = await list_factsheet_picker_options(read_session)
        return HtpyResponse(
            render_search(
                query=result.query,
                hits=converted_hits,
                corrected_query=result.corrected_query,
                original_query=result.original_query,
                lenses=lenses,
                factsheets=factsheets,
                selected_lens_id=result.lens_id,
                selected_factsheet_ids=factsheet_ids,
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
    lens_id = _first_non_empty(body.lens_ids)

    result = await run_search(
        read_session=read_session,
        q=body.content,
        limit=body.limit,
        exact=body.exact,
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
        factsheets = await list_factsheet_picker_options(read_session)
        return HtpyResponse(
            render_search(
                query=result.query,
                hits=converted_hits,
                corrected_query=result.corrected_query,
                original_query=result.original_query,
                lenses=lenses,
                factsheets=factsheets,
                selected_lens_id=result.lens_id,
                selected_factsheet_ids=body.factsheet_ids,
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
    lens_id = _first_non_empty(body.lens_ids)

    result = await run_search(
        read_session=read_session,
        q=body.content,
        limit=body.limit,
        exact=body.exact,
        lens_id=lens_id,
        spell_checker=spell_checker,
    )

    _record_search_event(result, write_session)

    lenses = await list_lens_picker_options(read_session)
    converted_hits = _convert_hits(result.hits)
    factsheets = await list_factsheet_picker_options(read_session)

    return HtpyResponse(
        render_search_root(
            query=result.query,
            hits=converted_hits,
            corrected_query=result.corrected_query,
            original_query=result.original_query,
            lenses=lenses,
            factsheets=factsheets,
            selected_lens_id=result.lens_id,
            hits_count=len(converted_hits),
        )
    )
