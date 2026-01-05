from typing import Any, Literal

_THREADS_LIST_BASE_SQL = """
WITH thread_first_messages AS (
    SELECT
        thread_id,
        content AS first_message_content,
        role AS first_message_role,
        created_at,
        import_source,
        ROW_NUMBER() OVER (PARTITION BY thread_id ORDER BY created_at ASC, id ASC) AS rn
    FROM messages
    WHERE deleted_at IS NULL
      AND thread_id IS NOT NULL
),
thread_stats AS (
    SELECT
        thread_id,
        MAX(created_at) AS last_activity_at,
        MAX(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) AS has_assistant_response
    FROM messages
    WHERE deleted_at IS NULL
      AND thread_id IS NOT NULL
    GROUP BY thread_id
)
SELECT
    tfm.thread_id,
    tfm.first_message_content,
    tfm.first_message_role,
    tfm.created_at,
    ts.last_activity_at,
    ts.has_assistant_response,
    tfm.import_source
FROM thread_first_messages tfm
JOIN thread_stats ts ON tfm.thread_id = ts.thread_id
WHERE tfm.rn = 1
"""


_THREADS_COUNT_BASE_SQL = """
WITH thread_first_messages AS (
    SELECT
        thread_id,
        created_at,
        import_source,
        ROW_NUMBER() OVER (PARTITION BY thread_id ORDER BY created_at ASC, id ASC) AS rn
    FROM messages
    WHERE deleted_at IS NULL
      AND thread_id IS NOT NULL
),
thread_stats AS (
    SELECT
        thread_id,
        MAX(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) AS has_assistant_response
    FROM messages
    WHERE deleted_at IS NULL
      AND thread_id IS NOT NULL
    GROUP BY thread_id
)
SELECT COUNT(*)
FROM thread_first_messages tfm
JOIN thread_stats ts ON tfm.thread_id = ts.thread_id
WHERE tfm.rn = 1
"""


def build_threads_list_queries(
    *,
    sort: Literal["last_activity", "created_at"],
    page: int,
    limit: int,
    date_from: int | None,
    date_to: int | None,
    has_assistant: bool | None,
    import_source: str | None,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    where_clauses: list[str] = []
    params: dict[str, Any] = {}

    if date_from is not None:
        where_clauses.append("tfm.created_at >= :date_from")
        params["date_from"] = date_from

    if date_to is not None:
        where_clauses.append("tfm.created_at <= :date_to")
        params["date_to"] = date_to

    if has_assistant is not None:
        where_clauses.append("ts.has_assistant_response = :has_assistant")
        params["has_assistant"] = 1 if has_assistant else 0

    if import_source is not None:
        where_clauses.append("tfm.import_source = :import_source")
        params["import_source"] = import_source

    where_suffix = ""
    if where_clauses:
        where_suffix = " AND " + " AND ".join(where_clauses)

    offset = (page - 1) * limit

    sort_col = "ts.last_activity_at" if sort == "last_activity" else "tfm.created_at"
    list_sql = _THREADS_LIST_BASE_SQL + where_suffix + f" ORDER BY {sort_col} DESC LIMIT :limit OFFSET :offset"
    list_params = dict(params)
    list_params["limit"] = limit
    list_params["offset"] = offset

    count_sql = _THREADS_COUNT_BASE_SQL + where_suffix
    count_params = params

    return list_sql, list_params, count_sql, count_params

