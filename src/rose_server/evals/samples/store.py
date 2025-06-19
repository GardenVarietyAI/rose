"""Store for EvalSample entity operations."""

import time
import uuid
from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import run_in_session
from ...entities.evals import EvalSample


class EvalSampleStore:
    """Store for evaluation sample operations."""

    async def create(
        self,
        eval_run_id: str,
        sample_index: int,
        input: str,
        ideal: str,
        completion: str,
        score: float,
        passed: bool,
        response_time: Optional[float] = None,
        tokens_used: Optional[int] = None,
        metadata: Optional[Dict] = None,
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

        async def _create(session: AsyncSession) -> EvalSample:
            session.add(eval_sample)
            await session.commit()
            await session.refresh(eval_sample)
            return eval_sample

        return await run_in_session(_create)

    async def get(self, sample_id: str) -> Optional[EvalSample]:
        """Get a specific evaluation sample by ID."""

        async def _get(session: AsyncSession) -> Optional[EvalSample]:
            result = await session.execute(select(EvalSample).where(EvalSample.id == sample_id))  # type: ignore[arg-type]
            return result.scalar_one_or_none()

        return await run_in_session(_get)

    async def list(
        self, eval_run_id: str, limit: int = 100, offset: int = 0, only_failed: bool = False
    ) -> List[EvalSample]:
        """Get evaluation samples for a specific run."""

        async def _list(session: AsyncSession) -> List[EvalSample]:
            query = select(EvalSample).where(EvalSample.eval_run_id == eval_run_id)  # type: ignore[arg-type]
            if only_failed:
                query = query.where(EvalSample.passed is False)  # type: ignore[arg-type]
            query = query.order_by(EvalSample.sample_index).offset(offset).limit(limit)  # type: ignore[arg-type]
            result = await session.execute(query)
            return list(result.scalars().all())

        return await run_in_session(_list)

    async def count(self, eval_run_id: str) -> Dict[str, int]:
        """Get sample counts for an evaluation run."""

        async def _count(session: AsyncSession) -> Dict[str, int]:
            total_result = await session.execute(
                select(func.count(EvalSample.id)).where(EvalSample.eval_run_id == eval_run_id)  # type: ignore[arg-type]
            )
            total = total_result.scalar() or 0
            passed_result = await session.execute(
                select(func.count(EvalSample.id))  # type: ignore[arg-type]
                .where(EvalSample.eval_run_id == eval_run_id)  # type: ignore[arg-type]
                .where(EvalSample.passed is True)  # type: ignore[arg-type]
            )
            passed = passed_result.scalar() or 0
            return {"total": total, "passed": passed, "failed": total - passed}

        return await run_in_session(_count)
