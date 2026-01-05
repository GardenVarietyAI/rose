import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from symspellpy import SymSpell

from rose_server.services.keyword_extractor import extract_keywords
from rose_server.services.search_query_builder import build_fts_search_query

KEYWORDS_THRESHOLD = 3
MAX_KEYWORDS = 3
MAX_KEYWORD_CANDIDATES = 5


@dataclass(frozen=True, slots=True)
class SearchHit:
    thread_id: str
    score: float
    user_message_id: str
    user_message_text: str
    user_message_excerpt: str | None
    user_message_created_at: int
    assistant_message_id: str
    assistant_message_text: str
    assistant_message_excerpt: str
    assistant_message_created_at: int
    assistant_message_model: str | None
    accepted: bool
    matched_role: str
    matched_message_id: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    original_query: str
    query: str
    fts_query: str
    corrected_query: str | None
    used_keywords: bool
    search_mode: str | None
    lens_id: str | None
    hits: list[SearchHit]


def normalize_query(query: str) -> str:
    chars_to_remove = "?!.,;:-+()[]{}*^"
    translator = str.maketrans(chars_to_remove, " " * len(chars_to_remove))
    normalized = query.translate(translator)
    return " ".join(normalized.split())


def sanitize_fts5_query(query: str) -> str:
    sanitized = re.sub(r"[^\w\s]", " ", query)
    return " ".join(sanitized.split())


def _build_search_hit(row: Any) -> SearchHit:
    return SearchHit(
        thread_id=row.thread_id,
        score=row.score,
        user_message_id=row.user_uuid,
        user_message_text=row.user_content or "",
        user_message_excerpt=row.user_excerpt,
        user_message_created_at=row.user_created_at,
        assistant_message_id=row.assistant_uuid,
        assistant_message_text=row.assistant_content or "",
        assistant_message_excerpt=row.assistant_excerpt or "",
        assistant_message_created_at=row.assistant_created_at,
        assistant_message_model=row.assistant_model,
        accepted=row.accepted_at is not None,
        matched_role=row.matched_role,
        matched_message_id=row.matched_message_id,
    )


async def _fetch_hits(
    session: AsyncSession,
    *,
    fts_query: str,
    limit: int,
    lens_id: str | None,
) -> list[SearchHit]:
    sql, params = build_fts_search_query(fts_query=fts_query, limit=limit, lens_id=lens_id)
    result = await session.execute(text(sql), params)
    rows = result.fetchall()

    return [_build_search_hit(row) for row in rows]


async def _augment_with_keywords(
    session: AsyncSession,
    *,
    query: str,
    remaining: int,
    seen_thread_ids: set[str],
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
            if hit.thread_id in seen_thread_ids:
                continue
            seen_thread_ids.add(hit.thread_id)
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
) -> SearchResult:
    original_query = q
    corrected_query: str | None = None
    used_keywords = False
    clean_query = original_query.strip()
    if not clean_query:
        if lens_id:
            hits = await _fetch_hits(read_session, fts_query="", limit=limit, lens_id=lens_id)
            return SearchResult(
                original_query=original_query,
                query="",
                fts_query="",
                corrected_query=None,
                used_keywords=False,
                search_mode="filter",
                lens_id=lens_id,
                hits=hits,
            )
        return SearchResult(
            original_query=original_query,
            query="",
            fts_query="",
            corrected_query=None,
            used_keywords=False,
            search_mode=None,
            lens_id=lens_id,
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
            query=effective_query,
            fts_query="",
            corrected_query=corrected_query,
            used_keywords=False,
            search_mode=None,
            lens_id=lens_id,
            hits=[],
        )

    hits = await _fetch_hits(read_session, fts_query=fts_query, limit=limit, lens_id=lens_id)

    if len(hits) < KEYWORDS_THRESHOLD:
        remaining = max(0, limit - len(hits))
        seen_thread_ids = {hit.thread_id for hit in hits}
        augmented_hits, used_keywords = await _augment_with_keywords(
            read_session,
            query=effective_query,
            remaining=remaining,
            seen_thread_ids=seen_thread_ids,
            lens_id=lens_id,
        )
        hits.extend(augmented_hits)

    search_mode = _compute_search_mode(corrected_query=corrected_query, used_keywords=used_keywords)

    return SearchResult(
        original_query=original_query,
        query=effective_query,
        fts_query=fts_query,
        corrected_query=corrected_query,
        used_keywords=used_keywords,
        search_mode=search_mode,
        lens_id=lens_id,
        hits=hits,
    )
