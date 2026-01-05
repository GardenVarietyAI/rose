from typing import Any

_FTS_QUERY_SQL = """
WITH base_hits AS (
    SELECT
        m.uuid,
        m.thread_id,
        m.role,
        m.content,
        m.created_at,
        m.accepted_at,
        m.model,
        -bm25(messages_fts) AS score,
        snippet(messages_fts, -1, '', '', '...', 64) AS excerpt
    FROM messages_fts
    JOIN messages AS m ON messages_fts.rowid = m.id
    WHERE {where}
),
best_per_thread AS (
    SELECT *
    FROM (
        SELECT
            bh.*,
            ROW_NUMBER() OVER (
                PARTITION BY bh.thread_id
                ORDER BY bh.score DESC, bh.created_at DESC, bh.uuid DESC
            ) AS rn
        FROM base_hits AS bh
    )
    WHERE rn = 1
),
user_selected AS (
    SELECT *
    FROM (
        SELECT
            m.thread_id,
            m.uuid,
            m.content,
            m.created_at,
            uh.excerpt,
            ROW_NUMBER() OVER (
                PARTITION BY m.thread_id
                ORDER BY m.created_at DESC, m.id DESC
            ) AS rn
        FROM messages AS m
        JOIN best_per_thread AS t ON t.thread_id = m.thread_id
        LEFT JOIN base_hits AS uh
          ON uh.uuid = m.uuid
         AND uh.role = 'user'
        WHERE m.role = 'user'
          AND m.deleted_at IS NULL
    )
    WHERE rn = 1
),
assistant_selected AS (
    SELECT *
    FROM (
        SELECT
            m.thread_id,
            m.uuid,
            m.content,
            m.created_at,
            m.accepted_at,
            m.model,
            ah.score,
            ah.excerpt,
            ROW_NUMBER() OVER (
                PARTITION BY m.thread_id
                ORDER BY
                    (m.accepted_at IS NULL) ASC,
                    (ah.score IS NULL) ASC,
                    ah.score DESC,
                    m.created_at DESC,
                    m.id DESC
            ) AS rn
        FROM messages AS m
        JOIN best_per_thread AS t ON t.thread_id = m.thread_id
        LEFT JOIN base_hits AS ah
          ON ah.uuid = m.uuid
         AND ah.role = 'assistant'
        WHERE m.role = 'assistant'
          AND m.deleted_at IS NULL
    )
    WHERE rn = 1
)
SELECT
    u.thread_id,
    u.uuid AS user_uuid,
    u.content AS user_content,
    u.excerpt AS user_excerpt,
    u.created_at AS user_created_at,
    a.uuid AS assistant_uuid,
    a.content AS assistant_content,
    COALESCE(a.excerpt, a.content) AS assistant_excerpt,
    a.created_at AS assistant_created_at,
    a.model AS assistant_model,
    a.accepted_at,
    t.score AS score,
    t.role AS matched_role,
    t.uuid AS matched_message_id
FROM best_per_thread AS t
JOIN user_selected AS u ON u.thread_id = t.thread_id
JOIN assistant_selected AS a ON a.thread_id = t.thread_id
ORDER BY t.score DESC
LIMIT :limit
"""


def build_fts_search_query(
    *,
    fts_query: str,
    limit: int,
    lens_id: str | None,
) -> tuple[str, dict[str, Any]]:
    where_parts: list[str] = []
    params: dict[str, Any] = {"limit": limit}

    if fts_query:
        where_parts.append("messages_fts MATCH :query")
        params["query"] = fts_query

    where_parts.append("m.deleted_at IS NULL")
    where_parts.append("m.object IS NULL")

    if lens_id:
        where_parts.append(
            """
            EXISTS (
                SELECT 1
                FROM messages AS lm
                WHERE lm.thread_id = m.thread_id
                  AND lm.role = 'assistant'
                  AND lm.deleted_at IS NULL
                  AND lm.lens_id = :lens_id
            )
            """.strip()
        )
        params["lens_id"] = lens_id

    where_clause = " AND ".join(where_parts)
    sql = _FTS_QUERY_SQL.format(where=where_clause)

    return sql, params
