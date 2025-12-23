from yoyo import step

__depends__ = {"20251222_02_R5k1n-add-deleted-at-to-messages"}

steps = [
    step(
        (
            "ALTER TABLE messages ADD COLUMN root_message_id VARCHAR GENERATED ALWAYS AS "
            "(json_extract(meta, '$.root_message_id'))"
        ),
        "ALTER TABLE messages DROP COLUMN root_message_id",
    ),
    step(
        (
            "ALTER TABLE messages ADD COLUMN parent_message_id VARCHAR GENERATED ALWAYS AS "
            "(json_extract(meta, '$.parent_message_id'))"
        ),
        "ALTER TABLE messages DROP COLUMN parent_message_id",
    ),
    step(
        "CREATE INDEX ix_messages_root_message_id ON messages (root_message_id)",
        "DROP INDEX ix_messages_root_message_id",
    ),
    step(
        "CREATE INDEX ix_messages_parent_message_id ON messages (parent_message_id)",
        "DROP INDEX ix_messages_parent_message_id",
    ),
]
