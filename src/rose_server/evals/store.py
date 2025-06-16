"""Async database operations for evaluations."""
import time
import uuid
from typing import Dict, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import run_in_session
from ..entities.evals import Eval, EvalRun, EvalSample


class EvalStore:
    """Store for evaluation-related database operations."""

    _instance = None

    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super(EvalStore, cls).__new__(cls)
        return cls._instance

    async def create_eval(
        self,
        id: str,
        name: str,
        description: Optional[str] = None,
        data_source_config: Optional[Dict] = None,
        testing_criteria: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> Eval:
        """Create a new evaluation definition."""
        eval = Eval(
            id=id,
            name=name,
            description=description,
            data_source_config=data_source_config or {},
            testing_criteria=testing_criteria or [],
            created_at=int(time.time()),
            meta=metadata,
        )

        async def _create(session: AsyncSession) -> Eval:
            session.add(eval)
            await session.commit()
            await session.refresh(eval)
            return eval
        return await run_in_session(_create)

    async def get_eval(self, eval_id: str) -> Optional[Eval]:
        """Get evaluation definition by ID."""

        async def _get(session: AsyncSession) -> Optional[Eval]:
            result = await session.execute(select(Eval).where(Eval.id == eval_id))
            return result.scalar_one_or_none()
        return await run_in_session(_get)

    async def list_evals(self, limit: int = 20) -> List[Eval]:
        """List evaluation definitions."""

        async def _list(session: AsyncSession) -> List[Eval]:
            result = await session.execute(select(Eval).order_by(Eval.created_at.desc()).limit(limit))
            return list(result.scalars().all())
        return await run_in_session(_list)

    async def delete_eval(self, eval_id: str) -> bool:
        """Delete an evaluation definition."""

        async def _delete(session: AsyncSession) -> bool:
            eval = await session.get(Eval, eval_id)
            if not eval:
                return False
            await session.execute(delete(Eval).where(Eval.id == eval_id))
            await session.commit()
            return True
        return await run_in_session(_delete)

    async def create_eval_run(
        self,
        id: str,
        eval_id: str,
        name: str,
        model: str,
        data_source: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> EvalRun:
        """Create a new evaluation run (execution of an eval)."""
        eval_run = EvalRun(
            id=id,
            eval_id=eval_id,
            name=name,
            model=model,
            status="queued",
            created_at=int(time.time()),
            data_source=data_source,
            meta=metadata,
        )

        async def _create(session: AsyncSession) -> EvalRun:
            session.add(eval_run)
            await session.commit()
            await session.refresh(eval_run)
            return eval_run
        return await run_in_session(_create)

    async def get_eval_run(self, run_id: str) -> Optional[EvalRun]:
        """Get evaluation run by ID."""

        async def _get(session: AsyncSession) -> Optional[EvalRun]:
            result = await session.execute(select(EvalRun).where(EvalRun.id == run_id))
            return result.scalar_one_or_none()
        return await run_in_session(_get)

    async def list_eval_runs(self, eval_id: Optional[str] = None, limit: int = 20) -> List[EvalRun]:
        """List evaluation runs, optionally filtered by eval_id."""

        async def _list(session: AsyncSession) -> List[EvalRun]:
            query = select(EvalRun)
            if eval_id:
                query = query.where(EvalRun.eval_id == eval_id)
            query = query.order_by(EvalRun.created_at.desc()).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())
        return await run_in_session(_list)

    async def update_eval_run_results(self, run_id: str, results: Dict) -> None:
        """Update evaluation run with results."""

        async def _update(session: AsyncSession) -> None:
            run = await session.get(EvalRun, run_id)
            if run:
                run.status = "completed"
                run.results = results
                run.completed_at = int(time.time())
                await session.commit()
        await run_in_session(_update)

    async def update_eval_run_error(self, run_id: str, error: str) -> None:
        """Update evaluation run with error."""

        async def _update(session: AsyncSession) -> None:
            run = await session.get(EvalRun, run_id)
            if run:
                run.status = "failed"
                run.error_message = error
                run.completed_at = int(time.time())
                await session.commit()
        await run_in_session(_update)

    async def update_eval_run_status(self, run_id: str, status: str) -> None:
        """Update evaluation run status."""

        async def _update(session: AsyncSession) -> None:
            run = await session.get(EvalRun, run_id)
            if run:
                run.status = status
                if status == "running":
                    run.started_at = int(time.time())
                await session.commit()
        await run_in_session(_update)

    async def create_eval_sample(
        self,
        eval_run_id: str,
        sample_index: int,
        input: str,
        expected_output: str,
        actual_output: str,
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
            expected_output=expected_output,
            actual_output=actual_output,
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

    async def get_eval_samples(
        self, 
        eval_run_id: str, 
        limit: int = 100,
        offset: int = 0,
        only_failed: bool = False
    ) -> List[EvalSample]:
        """Get evaluation samples for a specific run."""

        async def _get_samples(session: AsyncSession) -> List[EvalSample]:
            query = select(EvalSample).where(EvalSample.eval_run_id == eval_run_id)
            if only_failed:
                query = query.where(EvalSample.passed == False)
            query = query.order_by(EvalSample.sample_index).offset(offset).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())
        return await run_in_session(_get_samples)

    async def get_eval_sample(self, sample_id: str) -> Optional[EvalSample]:
        """Get a specific evaluation sample by ID."""

        async def _get(session: AsyncSession) -> Optional[EvalSample]:
            result = await session.execute(
                select(EvalSample).where(EvalSample.id == sample_id)
            )
            return result.scalar_one_or_none()
        return await run_in_session(_get)

    async def count_eval_samples(self, eval_run_id: str) -> Dict[str, int]:
        """Get sample counts for an evaluation run."""

        async def _count(session: AsyncSession) -> Dict[str, int]:
            total_result = await session.execute(
                select(func.count(EvalSample.id))
                .where(EvalSample.eval_run_id == eval_run_id)
            )
            total = total_result.scalar() or 0
            passed_result = await session.execute(
                select(func.count(EvalSample.id))
                .where(EvalSample.eval_run_id == eval_run_id)
                .where(EvalSample.passed == True)
            )
            passed = passed_result.scalar() or 0
            return {
                "total": total,
                "passed": passed,
                "failed": total - passed
            }
        return await run_in_session(_count)