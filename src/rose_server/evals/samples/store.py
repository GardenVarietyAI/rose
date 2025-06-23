"""Store for EvalSample entity operations."""

import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import false, func, select, true

from rose_server.database import get_session
from rose_server.entities.evals import EvalSample


async def create_eval_sample(
    eval_run_id: str,
    sample_index: int,
    input: str,
    ideal: str,
    completion: str,
    score: float,
    passed: bool,
    response_time: Optional[float] = None,
    tokens_used: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvalSample:
    """Create a new evaluation sample result."""
    sample_id = f"sample-{uuid.uuid4().hex[:8]}"
    eval_sample = EvalSample(
        id=sample_id,
        eval_run_id=eval_run_id,
        sample_index=sample_index,
        input=input,
        ideal=ideal,
        completion=completion,
        score=score,
        passed=passed,
        response_time=response_time,
        tokens_used=tokens_used,
        meta=metadata,
        created_at=int(time.time()),
    )

    async with get_session() as session:
        session.add(eval_sample)
        await session.commit()
        await session.refresh(eval_sample)
        return eval_sample


async def get_eval_sample(sample_id: str) -> Optional[EvalSample]:
    """Get a specific evaluation sample by ID."""
    async with get_session(read_only=True) as session:
        result = await session.execute(select(EvalSample).where(EvalSample.id == sample_id))
        return result.scalar_one_or_none()


async def list_eval_samples(
    eval_run_id: str, limit: int = 100, offset: int = 0, only_failed: bool = False
) -> List[EvalSample]:
    """Get evaluation samples for a specific run."""

    async with get_session(read_only=True) as session:
        query = select(EvalSample).where(EvalSample.eval_run_id == eval_run_id)
        if only_failed:
            query = query.where(EvalSample.passed == false())
        query = query.order_by(EvalSample.sample_index).offset(offset).limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())


async def count_eval_samples(eval_run_id: str) -> Dict[str, int]:
    """Get sample counts for an evaluation run."""
    async with get_session(read_only=True) as session:
        total_result = await session.execute(
            select(func.count(EvalSample.id)).where(EvalSample.eval_run_id == eval_run_id)
        )
        total = total_result.scalar() or 0
        passed_result = await session.execute(
            select(func.count(EvalSample.id))
            .where(EvalSample.eval_run_id == eval_run_id)
            .where(EvalSample.passed == true())
        )
        passed = passed_result.scalar() or 0
        return {"total": total, "passed": passed, "failed": total - passed}
