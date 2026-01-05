"""add messages tag computed column

Revision ID: 9c7f1a2b3c4d
Revises: aaab3a1ce85f
Create Date: 2026-01-04 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c7f1a2b3c4d"
down_revision: Union[str, Sequence[str], None] = "aaab3a1ce85f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        ALTER TABLE messages
        ADD COLUMN tag STRING
        GENERATED ALWAYS AS (json_extract(meta, '$.tag')) VIRTUAL
        """
    )
    op.execute("CREATE INDEX ix_messages_tag ON messages (tag)")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_messages_tag")
