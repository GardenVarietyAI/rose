import logging
import re
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel
from rake_nltk import Rake
from rose_server.dependencies import (
    get_db_session,
    get_readonly_db_session,
    get_spell_checker,
)
from rose_server.models.search_events import SearchEvent
from rose_server.routers.lenses import list_lens_options
from rose_server.views.pages.search import render_search
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["search"])

KEYWORD_AUGMENTATION_THRESHOLD = 3
KEYWORD_AUGMENTATION_MAX_KEYWORDS = 3


def normalize_query(query: str) -> str:
    """Remove punctuation for spell checking."""
    chars_to_remove = "?!.,;:-+()[]{}*^"
    translator = str.maketrans(chars_to_remove, " " * len(chars_to_remove))
    normalized = query.translate(translator)
    return " ".join(normalized.split())


def sanitize_fts5_query(query: str) -> str:
    """Escape and sanitize query string for FTS5."""
    sanitized = re.sub(r"[^\w\s]", " ", query)
    return " ".join(sanitized.split())


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    excerpt: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    index: str
    query: str
    hits: List[SearchHit]


_FTS_QUERY_SQL = """
        SELECT
            m.uuid,
            -bm25(messages_fts) as score,
            snippet(messages_fts, -1, '', '', '...', 64) as excerpt,
            m.content,
            m.thread_id,
            m.role,
            m.model,
            m.created_at
        FROM messages_fts
        JOIN messages m ON messages_fts.rowid = m.id
        WHERE {where}
        ORDER BY bm25(messages_fts)
        LIMIT :limit
    """


async def _fetch_hits(read_session: AsyncSession, fts_query: str, limit: int, lens_id: str | None) -> list[SearchHit]:
    where_parts = ["messages_fts MATCH :query"]
    params: dict[str, Any] = {"query": fts_query, "limit": limit}
    if lens_id:
        where_parts.append("m.lens_id = :lens_id")
        params["lens_id"] = lens_id

    sql = _FTS_QUERY_SQL.format(where=" AND ".join(where_parts))
    query = text(sql)
    result = await read_session.execute(query, params)
    rows = result.fetchall()

    hits: list[SearchHit] = []
    for row in rows:
        metadata: Dict[str, Any] = {
            "thread_id": row.thread_id,
            "role": row.role,
            "model": row.model,
            "created_at": row.created_at,
        }
        hits.append(
            SearchHit(
                id=row.uuid,
                score=row.score,
                text=row.content,
                excerpt=row.excerpt,
                metadata=metadata,
            )
        )

    return hits


def _extract_keywords(query: str) -> list[str]:
    rake = Rake()
    rake.extract_keywords_from_text(query)
    keywords: list[str] = rake.get_ranked_phrases()
    if not keywords:
        return []
    return keywords[:5]


async def _augment_with_keywords(
    read_session: AsyncSession,
    query: str,
    *,
    remaining: int,
    seen_ids: set[str],
    lens_id: str | None,
) -> tuple[list[SearchHit], bool]:
    if remaining <= 0:
        return ([], False)

    keywords = _extract_keywords(query)
    if not keywords:
        return ([], False)

    augmented_hits: list[SearchHit] = []
    used_keywords = False
    keywords_used = 0

    for keyword in keywords:
        if remaining <= 0:
            break

        sanitized_keyword = sanitize_fts5_query(keyword)
        if not sanitized_keyword:
            continue

        added_from_keyword = 0
        keyword_hits = await _fetch_hits(read_session, sanitized_keyword, remaining, lens_id)
        if keyword_hits:
            used_keywords = True

        for hit in keyword_hits:
            if hit.id in seen_ids:
                continue
            seen_ids.add(hit.id)
            augmented_hits.append(hit)
            remaining -= 1
            added_from_keyword += 1
            if remaining <= 0:
                break

        if added_from_keyword > 0:
            keywords_used += 1
            if keywords_used >= KEYWORD_AUGMENTATION_MAX_KEYWORDS:
                break

    return (augmented_hits, used_keywords)


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
    hits = []
    corrected_query = None
    original_query = q
    used_keywords = False

    if q:
        normalized_q = normalize_query(q)

        if not exact and spell_checker:
            suggestions = spell_checker.lookup_compound(normalized_q, max_edit_distance=2)
            if suggestions and suggestions[0].term.lower() != normalized_q.lower():
                corrected_query = suggestions[0].term
                q = corrected_query

        fts_query = sanitize_fts5_query(q)
        hits = await _fetch_hits(read_session, fts_query, limit, lens_id)

        if len(hits) < KEYWORD_AUGMENTATION_THRESHOLD:
            remaining = max(0, limit - len(hits))
            seen_ids = {hit.id for hit in hits}
            augmented_hits, used_keywords = await _augment_with_keywords(
                read_session,
                q,
                remaining=remaining,
                seen_ids=seen_ids,
                lens_id=lens_id,
            )
            hits.extend(augmented_hits)

        if corrected_query and used_keywords:
            search_mode = "fts_corrected_augmented"
        elif corrected_query:
            search_mode = "fts_corrected"
        elif used_keywords:
            search_mode = "fts_augmented"
        else:
            search_mode = "fts"

        search_event = SearchEvent(
            event_type="search",
            search_mode=search_mode,
            query=fts_query,
            original_query=original_query if original_query != fts_query else None,
            result_count=len(hits),
        )
        write_session.add(search_event)

    response_data = SearchResponse(
        index="messages",
        query=q,
        hits=hits,
    )

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        lenses = await list_lens_options(read_session)
        return HtpyResponse(
            render_search(
                query=q,
                hits=hits,
                corrected_query=corrected_query,
                original_query=original_query,
                lenses=lenses,
                selected_lens_id=lens_id,
            )
        )

    return response_data
