from yoyo import step

__depends__ = {"20251218_02_45FBe-add-accepted-at-to-messages"}

steps = [
    step(
        "ALTER TABLE messages ADD COLUMN lens_id VARCHAR GENERATED ALWAYS AS (json_extract(meta, '$.lens_id'))",
        "ALTER TABLE messages DROP COLUMN lens_id",
    ),
    step(
        "ALTER TABLE messages ADD COLUMN at_name VARCHAR GENERATED ALWAYS AS (json_extract(meta, '$.at_name'))",
        "ALTER TABLE messages DROP COLUMN at_name",
    ),
    step(
        "ALTER TABLE messages ADD COLUMN object VARCHAR GENERATED ALWAYS AS (json_extract(meta, '$.object'))",
        "ALTER TABLE messages DROP COLUMN object",
    ),
    step(
        "CREATE INDEX ix_messages_lens_id ON messages (lens_id)",
        "DROP INDEX ix_messages_lens_id",
    ),
    step(
        "CREATE INDEX ix_messages_at_name ON messages (at_name)",
        "DROP INDEX ix_messages_at_name",
    ),
    step(
        "CREATE INDEX ix_messages_object ON messages (object)",
        "DROP INDEX ix_messages_object",
    ),
]
