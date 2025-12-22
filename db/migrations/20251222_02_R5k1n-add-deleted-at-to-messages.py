from yoyo import step

__depends__ = {"20251222_01_LP7yU-add-lens-meta-virtual-columns"}

steps = [
    step(
        "ALTER TABLE messages ADD COLUMN deleted_at INTEGER",
        "ALTER TABLE messages DROP COLUMN deleted_at",
    ),
    step(
        "CREATE INDEX ix_messages_deleted_at ON messages (deleted_at)",
        "DROP INDEX ix_messages_deleted_at",
    ),
]
