from yoyo import step

__depends__: set[str] = set()

steps = [
    step(
        """
        CREATE TABLE messages (
            id INTEGER NOT NULL,
            uuid VARCHAR NOT NULL,
            thread_id VARCHAR NOT NULL,
            role VARCHAR NOT NULL,
            content VARCHAR,
            reasoning VARCHAR,
            model VARCHAR NOT NULL,
            meta JSON,
            created_at INTEGER NOT NULL,
            completion_id VARCHAR GENERATED ALWAYS AS (json_extract(meta, '$.completion_id')),
            PRIMARY KEY (id)
        )
        """,
        "DROP TABLE messages",
    ),
    step(
        "CREATE UNIQUE INDEX ix_messages_uuid ON messages (uuid)",
        "DROP INDEX ix_messages_uuid",
    ),
    step(
        "CREATE INDEX ix_messages_thread_id ON messages (thread_id)",
        "DROP INDEX ix_messages_thread_id",
    ),
    step(
        "CREATE INDEX ix_messages_completion_id ON messages (completion_id)",
        "DROP INDEX ix_messages_completion_id",
    ),
    step(
        """
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content,
            role UNINDEXED,
            model UNINDEXED,
            thread_id UNINDEXED,
            created_at UNINDEXED,
            content='messages',
            content_rowid='id',
            tokenize = 'porter unicode61 remove_diacritics 2'
        )
        """,
        "DROP TABLE messages_fts",
    ),
    step(
        """
        CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content, role, model, thread_id, created_at)
            VALUES (new.id, COALESCE(new.content, ''), new.role, new.model, new.thread_id, new.created_at);
        END
        """,
        "DROP TRIGGER messages_ai",
    ),
    step(
        """
        CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content, role, model, thread_id, created_at)
            VALUES('delete', old.id, COALESCE(old.content, ''), old.role, old.model, old.thread_id, old.created_at);
        END
        """,
        "DROP TRIGGER messages_ad",
    ),
    step(
        """
        CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content, role, model, thread_id, created_at)
            VALUES('delete', old.id, COALESCE(old.content, ''), old.role, old.model, old.thread_id, old.created_at);
            INSERT INTO messages_fts(rowid, content, role, model, thread_id, created_at)
            VALUES (new.id, COALESCE(new.content, ''), new.role, new.model, new.thread_id, new.created_at);
        END
        """,
        "DROP TRIGGER messages_au",
    ),
]
