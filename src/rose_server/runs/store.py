"""Database storage for runs."""

import logging
import time
from typing import Any, Dict, List, Optional

from sqlmodel import select

from rose_server.database import Run, get_session

logger = logging.getLogger(__name__)


async def create_run(run: Run) -> Run:
    """Create a new run in the database."""
    async with get_session() as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)

        logger.info(f"Created run {run.id} for thread {run.thread_id}")
        return run


async def get_run(run_id: str) -> Optional[Run]:
    """Get a run by ID."""
    async with get_session(read_only=True) as session:
        return await session.get(Run, run_id)


async def update_run(
    run_id: str,
    *,
    status: Optional[str] = None,
    last_error: Optional[Dict[str, Any]] = None,
    required_action: Optional[Dict[str, Any]] = None,
    usage: Optional[Dict[str, Any]] = None,
    incomplete_details: Optional[Dict[str, Any]] = None,
) -> Optional[Run]:
    """Update a run with the provided fields."""
    async with get_session() as session:
        run = await session.get(Run, run_id)
        if not run:
            return None

        # Update provided fields
        if status is not None:
            run.status = status
        if last_error is not None:
            run.last_error = last_error
        if required_action is not None:
            run.required_action = required_action
        if usage is not None:
            run.usage = usage
        if incomplete_details is not None:
            run.incomplete_details = incomplete_details

        # Update timestamp fields based on status
        current_time = int(time.time())
        if status == "in_progress" and not run.started_at:
            run.started_at = current_time
        elif status == "completed":
            run.completed_at = current_time
        elif status == "failed":
            run.failed_at = current_time
        elif status == "cancelled":
            run.cancelled_at = current_time

        session.add(run)
        await session.commit()
        await session.refresh(run)

        return run


async def list_runs(thread_id: str, limit: int = 20, order: str = "desc") -> List[Run]:
    """List runs for a thread."""
    async with get_session(read_only=True) as session:
        statement = select(Run).where(Run.thread_id == thread_id)

        if order == "desc":
            statement = statement.order_by(Run.created_at.desc())
        else:
            statement = statement.order_by(Run.created_at.asc())

        statement = statement.limit(limit)
        runs = (await session.execute(statement)).scalars().all()

        return list(runs)


async def cancel_run(run_id: str) -> Optional[Run]:
    """Cancel a run."""
    return await update_run(run_id, status="cancelled")
