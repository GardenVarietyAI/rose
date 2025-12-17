import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from sqlalchemy import text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["search"])


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
) -> Any:
    hits = []
    corrected_query = None
    original_query = q

    if q:
        # Normalize query for spell checking (remove punctuation that doesn't affect meaning)
        normalized_q = q
        for char in ["?", "!", ".", ",", ";", ":", "-", "+", "(", ")", "[", "]", "{", "}", "*", "^"]:
            normalized_q = normalized_q.replace(char, " ")
        normalized_q = " ".join(normalized_q.split())

        if not exact and request.app.state.spell_checker:
            suggestions = request.app.state.spell_checker.lookup_compound(normalized_q, max_edit_distance=2)
            if suggestions and suggestions[0].term.lower() != normalized_q.lower():
                corrected_query = suggestions[0].term
                q = corrected_query

        # Escape FTS5 special characters
        fts_query = q.replace('"', '""')

        # Remove FTS5 operators that could cause syntax errors
        for char in ["?", "*", "(", ")", "{", "}", "[", "]", "^", ":", "-", "+"]:
            fts_query = fts_query.replace(char, " ")

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

        async with request.app.state.get_db_session(read_only=True) as session:
            result = await session.execute(query, {"query": fts_query, "limit": limit})
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

    response_data = SearchResponse(
        index="messages",
        query=q,
        hits=hits,
    )

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return request.app.state.templates.TemplateResponse(
            "search.html",
            {
                "request": request,
                "query": q,
                "hits": hits,
                "corrected_query": corrected_query,
                "original_query": original_query,
            },
        )

    return response_data
