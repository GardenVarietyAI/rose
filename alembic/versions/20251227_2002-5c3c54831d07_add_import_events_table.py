"""add_import_events_table

Revision ID: 5c3c54831d07
Revises: 3dcbfbea1a0d
Create Date: 2025-12-27 20:02:22.095836

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5c3c54831d07"
down_revision: Union[str, Sequence[str], None] = "3dcbfbea1a0d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "import_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uuid", sa.String(), nullable=False),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("import_source", sa.String(), nullable=False),
        sa.Column("imported_count", sa.Integer(), nullable=False),
        sa.Column("skipped_duplicates", sa.Integer(), nullable=False),
        sa.Column("total_records", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_import_events_uuid", "import_events", ["uuid"], unique=True)
    op.create_index("ix_import_events_batch_id", "import_events", ["batch_id"])
    op.create_index("ix_import_events_import_source", "import_events", ["import_source"])
    op.create_index("ix_import_events_created_at", "import_events", ["created_at"])

    op.add_column("messages", sa.Column("import_batch_id", sa.String(), nullable=True))
    op.add_column("messages", sa.Column("import_external_id", sa.String(), nullable=True))

    # Add computed column for import_source
    op.execute("""
        ALTER TABLE messages ADD COLUMN import_source TEXT
        GENERATED ALWAYS AS (json_extract(meta, '$.import_source')) STORED
    """)

    op.create_index("ix_messages_import_batch_id", "messages", ["import_batch_id"])
    op.create_index("ix_messages_import_source", "messages", ["import_source"])

    # Create partial unique index
    op.execute("""
        CREATE UNIQUE INDEX ix_messages_import_dedup
        ON messages (import_source, import_external_id)
        WHERE import_source IS NOT NULL AND import_external_id IS NOT NULL AND deleted_at IS NULL
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_messages_import_dedup", table_name="messages")
    op.drop_index("ix_messages_import_source", table_name="messages")
    op.drop_index("ix_messages_import_batch_id", table_name="messages")
    op.drop_index("ix_import_events_created_at", table_name="import_events")
    op.drop_index("ix_import_events_import_source", table_name="import_events")
    op.drop_index("ix_import_events_batch_id", table_name="import_events")
    op.drop_index("ix_import_events_uuid", table_name="import_events")
    op.drop_table("import_events")
