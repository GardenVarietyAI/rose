"""add_imported_source_column

Revision ID: 8f4a2b19c3c7
Revises: 3dcbfbea1a0d
Create Date: 2025-12-28 14:12:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8f4a2b19c3c7"
down_revision: Union[str, Sequence[str], None] = "3dcbfbea1a0d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "ALTER TABLE messages "
        "ADD COLUMN imported_source TEXT "
        "GENERATED ALWAYS AS (json_extract(meta, '$.imported_source')) VIRTUAL"
    )
    op.create_index("ix_messages_imported_source", "messages", ["imported_source"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_messages_imported_source", table_name="messages")
    # SQLite cannot drop columns without table rebuild.
