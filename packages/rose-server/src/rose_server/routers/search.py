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
from rose_server.views.pages.search import render_search
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["search"])


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


@router.get("/search")
async def search_messages(
    request: Request,
    q: str = Query("", description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    exact: bool = Query(False, description="Skip spell correction"),
    read_session: AsyncSession = Depends(get_readonly_db_session),
    write_session: AsyncSession = Depends(get_db_session),
    spell_checker: SymSpell | None = Depends(get_spell_checker),
) -> Any:
    hits = []
    corrected_query = None
    original_query = q

    if q:
        normalized_q = normalize_query(q)

        if not exact and spell_checker:
            suggestions = spell_checker.lookup_compound(normalized_q, max_edit_distance=2)
            if suggestions and suggestions[0].term.lower() != normalized_q.lower():
                corrected_query = suggestions[0].term
                q = corrected_query

        fts_query = sanitize_fts5_query(q)

        query = text("""
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
            WHERE messages_fts MATCH :query
            ORDER BY bm25(messages_fts)
            LIMIT :limit
        """)

        result = await read_session.execute(query, {"query": fts_query, "limit": limit})
        rows = result.fetchall()

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

    fallback_keywords: List[str] | None = None
    if q and len(hits) == 0:
        rake = Rake()
        rake.extract_keywords_from_text(q)
        keywords = rake.get_ranked_phrases()
        if keywords:
            verified_keywords = []
            for keyword in keywords[:5]:
                sanitized_keyword = sanitize_fts5_query(keyword)
                check_query = text("SELECT COUNT(*) as count FROM messages_fts WHERE messages_fts MATCH :query")
                result = await read_session.execute(check_query, {"query": sanitized_keyword})
                count = result.scalar()

                if count and count > 0:
                    verified_keywords.append(keyword)
                    if len(verified_keywords) >= 3:
                        break

            fallback_keywords = verified_keywords if verified_keywords else None

    if q:
        if corrected_query:
            search_mode = "fts_corrected"
        elif fallback_keywords:
            search_mode = "fts_rake"
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
        return HtpyResponse(
            render_search(
                query=q,
                hits=hits,
                corrected_query=corrected_query,
                original_query=original_query,
                fallback_keywords=fallback_keywords,
            )
        )

    return response_data
