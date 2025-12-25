import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select
from symspellpy import SymSpell

from rose_server.models.messages import Message
from rose_server.services.keyword_extractor import extract_keywords
from rose_server.services.query_parser import parse_query

KEYWORDS_THRESHOLD = 3
MAX_KEYWORDS = 3
MAX_KEYWORD_CANDIDATES = 5

DEFAULT_LENS_OBJECT = "lens"

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


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    text: str
    excerpt: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SearchResult:
    original_query: str
    clean_query: str
    query: str
    fts_query: str
    corrected_query: str | None
    used_keywords: bool
    search_mode: str | None
    lens_id: str | None
    requested_lens_at_name: str | None
    hits: list[SearchHit]


def normalize_query(query: str) -> str:
    chars_to_remove = "?!.,;:-+()[]{}*^"
    translator = str.maketrans(chars_to_remove, " " * len(chars_to_remove))
    normalized = query.translate(translator)
    return " ".join(normalized.split())


def sanitize_fts5_query(query: str) -> str:
    sanitized = re.sub(r"[^\w\s]", " ", query)
    return " ".join(sanitized.split())


async def _resolve_lens_id_from_mentions(
    session: AsyncSession,
    *,
    lens_object: str,
    at_names: list[str],
) -> str | None:
    if not at_names:
        return None

    last_at_name = at_names[-1]
    result = await session.execute(
        select(Message)
        .where(
            col(Message.object) == lens_object,
            col(Message.at_name) == last_at_name,
            col(Message.deleted_at).is_(None),
        )
        .order_by(col(Message.created_at).desc(), col(Message.id).desc())
        .limit(1)
    )
    lens_msg = result.scalar_one_or_none()
    if lens_msg is None:
        return None
    return lens_msg.uuid


async def _fetch_hits(
    session: AsyncSession,
    *,
    fts_query: str,
    limit: int,
    lens_id: str | None,
) -> list[SearchHit]:
    where_parts = ["messages_fts MATCH :query", "m.deleted_at IS NULL"]
    params: dict[str, Any] = {"query": fts_query, "limit": limit}
    if lens_id:
        where_parts.append("m.lens_id = :lens_id")
        params["lens_id"] = lens_id

    sql = _FTS_QUERY_SQL.format(where=" AND ".join(where_parts))
    result = await session.execute(text(sql), params)
    rows = result.fetchall()

    hits: list[SearchHit] = []
    for row in rows:
        hits.append(
            SearchHit(
                id=row.uuid,
                score=row.score,
                text=row.content,
                excerpt=row.excerpt,
                metadata={
                    "thread_id": row.thread_id,
                    "role": row.role,
                    "model": row.model,
                    "created_at": row.created_at,
                },
            )
        )
    return hits


async def _augment_with_keywords(
    session: AsyncSession,
    *,
    query: str,
    remaining: int,
    seen_ids: set[str],
    lens_id: str | None,
) -> tuple[list[SearchHit], bool]:
    if remaining <= 0:
        return ([], False)

    result = extract_keywords(query, max_keywords=MAX_KEYWORD_CANDIDATES)
    if not result.phrases:
        return ([], False)

    augmented_hits: list[SearchHit] = []
    used_keywords = False
    keywords_used = 0

    for keyword in result.phrases:
        if remaining <= 0:
            break

        sanitized_keyword = sanitize_fts5_query(keyword)
        if not sanitized_keyword:
            continue

        added_from_keyword = 0
        keyword_hits = await _fetch_hits(session, fts_query=sanitized_keyword, limit=remaining, lens_id=lens_id)
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


def _compute_search_mode(*, corrected_query: str | None, used_keywords: bool) -> str:
    if corrected_query and used_keywords:
        return "fts_corrected_augmented"
    if corrected_query:
        return "fts_corrected"
    if used_keywords:
        return "fts_augmented"
    return "fts"


async def run_search(
    *,
    read_session: AsyncSession,
    q: str,
    limit: int,
    exact: bool,
    lens_id: str | None,
    spell_checker: SymSpell | None,
    lens_object: str = DEFAULT_LENS_OBJECT,
) -> SearchResult:
    original_query = q
    corrected_query: str | None = None
    used_keywords = False
    search_mode: str | None = None
    requested_lens_at_name: str | None = None

    if not original_query:
        return SearchResult(
            original_query=original_query,
            clean_query="",
            query="",
            fts_query="",
            corrected_query=None,
            used_keywords=False,
            search_mode=None,
            lens_id=lens_id,
            requested_lens_at_name=None,
            hits=[],
        )

    _, clean_query, at_names = parse_query(original_query)
    requested_lens_at_name = at_names[-1] if at_names else None
    if at_names and not lens_id:
        resolved = await _resolve_lens_id_from_mentions(read_session, lens_object=lens_object, at_names=at_names)
        if resolved:
            lens_id = resolved

    if not clean_query:
        return SearchResult(
            original_query=original_query,
            clean_query="",
            query="",
            fts_query="",
            corrected_query=None,
            used_keywords=False,
            search_mode=None,
            lens_id=lens_id,
            requested_lens_at_name=requested_lens_at_name,
            hits=[],
        )

    effective_query = clean_query
    normalized_query = normalize_query(clean_query)
    if not exact and spell_checker:
        suggestions = spell_checker.lookup_compound(normalized_query, max_edit_distance=2)
        if suggestions and suggestions[0].term.lower() != normalized_query.lower():
            corrected_query = suggestions[0].term
            if corrected_query:
                effective_query = corrected_query

    fts_query = sanitize_fts5_query(effective_query)
    if not fts_query:
        return SearchResult(
            original_query=original_query,
            clean_query=clean_query,
            query=effective_query,
            fts_query="",
            corrected_query=corrected_query,
            used_keywords=False,
            search_mode=None,
            lens_id=lens_id,
            requested_lens_at_name=requested_lens_at_name,
            hits=[],
        )

    hits = await _fetch_hits(read_session, fts_query=fts_query, limit=limit, lens_id=lens_id)

    if len(hits) < KEYWORDS_THRESHOLD:
        remaining = max(0, limit - len(hits))
        seen_ids = {hit.id for hit in hits}
        augmented_hits, used_keywords = await _augment_with_keywords(
            read_session,
            query=effective_query,
            remaining=remaining,
            seen_ids=seen_ids,
            lens_id=lens_id,
        )
        hits.extend(augmented_hits)

    search_mode = _compute_search_mode(corrected_query=corrected_query, used_keywords=used_keywords)

    return SearchResult(
        original_query=original_query,
        clean_query=clean_query,
        query=effective_query,
        fts_query=fts_query,
        corrected_query=corrected_query,
        used_keywords=used_keywords,
        search_mode=search_mode,
        lens_id=lens_id,
        requested_lens_at_name=requested_lens_at_name,
        hits=hits,
    )
