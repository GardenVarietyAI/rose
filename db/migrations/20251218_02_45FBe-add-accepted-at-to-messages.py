"""
add accepted_at to messages
"""

from yoyo import step

__depends__ = {"20251218_01_X4ii6-add-search-events-table"}

steps = [
    step(
        "ALTER TABLE messages ADD COLUMN accepted_at INTEGER",
        "ALTER TABLE messages DROP COLUMN accepted_at",
    ),
    step(
        "CREATE INDEX ix_messages_accepted_at ON messages (accepted_at)",
        "DROP INDEX ix_messages_accepted_at",
    ),
]
