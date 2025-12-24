import logging
import re
from collections import Counter
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, Request
from htpy.starlette import HtpyResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell

from rose_server.dependencies import (
    get_db_session,
    get_readonly_db_session,
    get_spell_checker,
)
from rose_server.models.search_events import SearchEvent
from rose_server.routers.lenses import list_lens_options
from rose_server.services.stopwords import EN_STOPWORDS
from rose_server.views.pages.search import render_search

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["search"])

KEYWORDS_THRESHOLD = 3
MAX_KEYWORDS = 3
MAX_KEYWORD_CANDIDATES = 5


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
    where_parts = ["messages_fts MATCH :query", "m.deleted_at IS NULL"]
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


def _iter_keyword_phrases(query: str, stopwords_set: set[str]) -> list[str]:
    tokens = re.findall(r"[a-z0-9_\.-]+", query.lower())
    if not tokens:
        return []

    phrases: list[str] = []
    phrase_tokens: list[str] = []
    for token in tokens:
        if ("." in token or "_" in token or "-" in token) and len(token) > 2:
            phrase_tokens.append(token)
            continue

        if len(token) < 3 or token in stopwords_set:
            if phrase_tokens:
                phrases.append(" ".join(phrase_tokens))
                phrase_tokens = []
            continue
        phrase_tokens.append(token)

    if phrase_tokens:
        phrases.append(" ".join(phrase_tokens))

    return phrases


def _score_phrases(phrases: list[str]) -> list[tuple[float, str]]:
    """Score phrases by word frequency and length."""
    word_counts = Counter(word for phrase in phrases for word in phrase.split())

    scored: list[tuple[float, str]] = []
    for phrase in phrases:
        words = phrase.split()
        score = float(len(words) * sum(word_counts[word] for word in words))
        scored.append((score, phrase))

    return scored


def _extract_keywords(
    query: str,
    max_keywords: int = MAX_KEYWORD_CANDIDATES,
    extra_stopwords: set[str] | None = None,
) -> list[str]:
    stopwords = set(EN_STOPWORDS)
    if extra_stopwords:
        stopwords.update(extra_stopwords)

    phrases = _iter_keyword_phrases(query, stopwords)
    if not phrases:
        return []

    scored: list[tuple[float, str]] = _score_phrases(phrases)
    scored.sort(key=lambda item: (-item[0], item[1]))
    keywords: list[str] = []
    seen: set[str] = set()
    for _, phrase in scored:
        if phrase in seen:
            continue
        seen.add(phrase)
        keywords.append(phrase)
        if len(keywords) >= max_keywords:
            break

    return keywords


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
            if keywords_used >= MAX_KEYWORDS:
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

        if len(hits) < KEYWORDS_THRESHOLD:
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
