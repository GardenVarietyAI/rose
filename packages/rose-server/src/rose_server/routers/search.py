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
) -> Any:
    hits = []

    if q:
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
            result = await session.execute(query, {"query": q, "limit": limit})
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
            {"request": request, "query": q, "hits": hits},
        )

    return response_data
