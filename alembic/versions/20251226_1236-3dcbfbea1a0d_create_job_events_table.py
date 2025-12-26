"""create_job_events_table

Revision ID: 3dcbfbea1a0d
Revises: eb8ca7eaf12d
Create Date: 2025-12-26 12:36:52.035238

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3dcbfbea1a0d"
down_revision: Union[str, Sequence[str], None] = "eb8ca7eaf12d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "job_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("event_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("job_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("thread_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("error", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_job_events_uuid"), "job_events", ["uuid"], unique=True)
    op.create_index(op.f("ix_job_events_event_type"), "job_events", ["event_type"], unique=False)
    op.create_index(op.f("ix_job_events_job_id"), "job_events", ["job_id"], unique=False)
    op.create_index(op.f("ix_job_events_thread_id"), "job_events", ["thread_id"], unique=False)
    op.create_index(op.f("ix_job_events_created_at"), "job_events", ["created_at"], unique=False)
    op.create_index(op.f("ix_job_events_status"), "job_events", ["status"], unique=False)
    op.create_index("ix_job_events_job_id_created_at", "job_events", ["job_id", "created_at"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_job_events_job_id_created_at", table_name="job_events")
    op.drop_index(op.f("ix_job_events_status"), table_name="job_events")
    op.drop_index(op.f("ix_job_events_created_at"), table_name="job_events")
    op.drop_index(op.f("ix_job_events_thread_id"), table_name="job_events")
    op.drop_index(op.f("ix_job_events_job_id"), table_name="job_events")
    op.drop_index(op.f("ix_job_events_event_type"), table_name="job_events")
    op.drop_index(op.f("ix_job_events_uuid"), table_name="job_events")
    op.drop_table("job_events")
