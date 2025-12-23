from yoyo import step

__depends__ = {"20251223_01_h1sT9-add-message-history-virtual-columns"}

steps = [
    step("DROP TRIGGER IF EXISTS messages_ai", "SELECT 1"),
    step("DROP TRIGGER IF EXISTS messages_au", "SELECT 1"),
    step("DROP TRIGGER IF EXISTS messages_ad", "SELECT 1"),
    step(
        "DROP TABLE IF EXISTS messages_fts",
        """
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content,
            role UNINDEXED,
            model UNINDEXED,
            thread_id UNINDEXED,
            created_at UNINDEXED,
            tokenize = 'porter unicode61 remove_diacritics 2'
        )
        """,
    ),
    step(
        """
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content,
            root_message_id UNINDEXED,
            role UNINDEXED,
            model UNINDEXED,
            thread_id UNINDEXED,
            created_at UNINDEXED,
            tokenize = 'porter unicode61 remove_diacritics 2'
        )
        """,
        "DROP TABLE messages_fts",
    ),
    step(
        """
        INSERT INTO messages_fts(rowid, content, root_message_id, role, model, thread_id, created_at)
        WITH ranked AS (
            SELECT
                id,
                COALESCE(content, '') AS content,
                COALESCE(root_message_id, uuid) AS root_message_id,
                role,
                COALESCE(model, '') AS model,
                COALESCE(thread_id, '') AS thread_id,
                created_at,
                row_number() OVER (
                    PARTITION BY COALESCE(root_message_id, uuid)
                    ORDER BY created_at DESC, id DESC
                ) AS rn
            FROM messages
            WHERE deleted_at IS NULL
        )
        SELECT id, content, root_message_id, role, model, thread_id, created_at
        FROM ranked
        WHERE rn = 1
        """,
        "DELETE FROM messages_fts",
    ),
    step(
        """
        CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
            DELETE FROM messages_fts
            WHERE root_message_id = COALESCE(new.root_message_id, new.uuid);

            INSERT INTO messages_fts(rowid, content, root_message_id, role, model, thread_id, created_at)
            SELECT
                m.id,
                COALESCE(m.content, ''),
                COALESCE(m.root_message_id, m.uuid),
                m.role,
                COALESCE(m.model, ''),
                COALESCE(m.thread_id, ''),
                m.created_at
            FROM messages m
            WHERE m.deleted_at IS NULL
              AND COALESCE(m.root_message_id, m.uuid) = COALESCE(new.root_message_id, new.uuid)
            ORDER BY m.created_at DESC, m.id DESC
            LIMIT 1;
        END
        """,
        "DROP TRIGGER messages_ai",
    ),
    step(
        """
        CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts
            WHERE root_message_id = COALESCE(old.root_message_id, old.uuid);

            INSERT INTO messages_fts(rowid, content, root_message_id, role, model, thread_id, created_at)
            SELECT
                m.id,
                COALESCE(m.content, ''),
                COALESCE(m.root_message_id, m.uuid),
                m.role,
                COALESCE(m.model, ''),
                COALESCE(m.thread_id, ''),
                m.created_at
            FROM messages m
            WHERE m.deleted_at IS NULL
              AND COALESCE(m.root_message_id, m.uuid) = COALESCE(old.root_message_id, old.uuid)
            ORDER BY m.created_at DESC, m.id DESC
            LIMIT 1;
        END
        """,
        "DROP TRIGGER messages_au",
    ),
    step(
        """
        CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts
            WHERE root_message_id = COALESCE(old.root_message_id, old.uuid);

            INSERT INTO messages_fts(rowid, content, root_message_id, role, model, thread_id, created_at)
            SELECT
                m.id,
                COALESCE(m.content, ''),
                COALESCE(m.root_message_id, m.uuid),
                m.role,
                COALESCE(m.model, ''),
                COALESCE(m.thread_id, ''),
                m.created_at
            FROM messages m
            WHERE m.deleted_at IS NULL
              AND COALESCE(m.root_message_id, m.uuid) = COALESCE(old.root_message_id, old.uuid)
            ORDER BY m.created_at DESC, m.id DESC
            LIMIT 1;
        END
        """,
        "DROP TRIGGER messages_ad",
    ),
]
