"""backfill lens root_message_id for revision tracking

Revision ID: aaab3a1ce85f
Revises: 5c3c54831d07
Create Date: 2026-01-02 15:43:39.143740

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aaab3a1ce85f'
down_revision: Union[str, Sequence[str], None] = '5c3c54831d07'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Backfill root_message_id for existing lenses
    # For legacy lenses without root_message_id in meta, set root_message_id = uuid
    op.execute("""
        UPDATE messages
        SET meta = json_set(
            COALESCE(meta, '{}'),
            '$.root_message_id',
            uuid
        )
        WHERE json_extract(meta, '$.object') = 'lens'
          AND json_extract(meta, '$.root_message_id') IS NULL
    """)

    # Normalize stored lens_id references to the lens root_message_id
    op.execute("""
        UPDATE messages
        SET meta = json_set(
            COALESCE(meta, '{}'),
            '$.lens_id',
            (
                SELECT json_extract(m2.meta, '$.root_message_id')
                FROM messages m2
                WHERE m2.uuid = json_extract(messages.meta, '$.lens_id')
                  AND json_extract(m2.meta, '$.object') = 'lens'
                LIMIT 1
            )
        )
        WHERE json_extract(meta, '$.lens_id') IS NOT NULL
          AND json_extract(meta, '$.lens_id') != ''
          AND EXISTS (
              SELECT 1 FROM messages m2
              WHERE m2.uuid = json_extract(messages.meta, '$.lens_id')
                AND json_extract(m2.meta, '$.object') = 'lens'
          )
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Remove root_message_id and parent_message_id from lens meta
    op.execute("""
        UPDATE messages
        SET meta = json_remove(
            json_remove(meta, '$.root_message_id'),
            '$.parent_message_id'
        )
        WHERE json_extract(meta, '$.object') = 'lens'
    """)
