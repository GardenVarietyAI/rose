from yoyo import step

__depends__: set[str] = {"20250117_01_aB3Cd-initial-schema"}

steps = [
    step(
        """
        CREATE TABLE search_events (
            id INTEGER NOT NULL,
            uuid VARCHAR NOT NULL,
            event_type VARCHAR NOT NULL,
            search_mode VARCHAR NOT NULL,
            query VARCHAR NOT NULL,
            original_query VARCHAR,
            result_count INTEGER NOT NULL,
            thread_id VARCHAR,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (id)
        )
        """,
        "DROP TABLE search_events",
    ),
    step(
        "CREATE UNIQUE INDEX ix_search_events_uuid ON search_events (uuid)",
        "DROP INDEX ix_search_events_uuid",
    ),
    step(
        "CREATE INDEX ix_search_events_event_type ON search_events (event_type)",
        "DROP INDEX ix_search_events_event_type",
    ),
    step(
        "CREATE INDEX ix_search_events_search_mode ON search_events (search_mode)",
        "DROP INDEX ix_search_events_search_mode",
    ),
    step(
        "CREATE INDEX ix_search_events_result_count ON search_events (result_count)",
        "DROP INDEX ix_search_events_result_count",
    ),
    step(
        "CREATE INDEX ix_search_events_thread_id ON search_events (thread_id)",
        "DROP INDEX ix_search_events_thread_id",
    ),
    step(
        "CREATE INDEX ix_search_events_created_at ON search_events (created_at)",
        "DROP INDEX ix_search_events_created_at",
    ),
    step(
        "CREATE INDEX ix_search_events_type_created ON search_events (event_type, created_at)",
        "DROP INDEX ix_search_events_type_created",
    ),
]
