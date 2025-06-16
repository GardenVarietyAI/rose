"""Simple in-memory storage for runs with singleton pattern."""
import time
import uuid
from typing import Any, Dict, List, Optional

from rose_server.schemas.assistants import CreateRunRequest, Run
from rose_server.schemas.runs import RunStep, RunStepType


class RunsStore:
    """Simple in-memory storage for runs using singleton pattern."""

    _instance = None

    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super(RunsStore, cls).__new__(cls)
            cls._instance.runs = {}
            cls._instance.steps = {}
        return cls._instance

    async def create_run(self, thread_id: str, request: CreateRunRequest) -> Run:
        """Create a new run."""
        run_id = f"run_{uuid.uuid4().hex}"
        created_at = int(time.time())
        run = Run(
            id=run_id,
            created_at=created_at,
            thread_id=thread_id,
            assistant_id=request.assistant_id,
            status="queued",
            model=request.model or "zephyr",
            instructions=request.instructions,
            tools=request.tools or [],
            metadata=request.metadata,
            temperature=request.temperature,
            top_p=request.top_p,
            max_prompt_tokens=request.max_prompt_tokens,
            max_completion_tokens=request.max_completion_tokens,
            truncation_strategy=request.truncation_strategy,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls if request.parallel_tool_calls is not None else True,
            response_format=request.response_format,
        )
        self.runs[run_id] = run
        return run

    async def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        return self.runs.get(run_id)

    async def list_runs(self, thread_id: str, limit: int = 20, order: str = "desc") -> List[Run]:
        """List runs for a thread."""
        thread_runs = [run for run in self.runs.values() if run.thread_id == thread_id]
        if order == "desc":
            thread_runs = sorted(thread_runs, key=lambda r: r.created_at, reverse=True)
        else:
            thread_runs = sorted(thread_runs, key=lambda r: r.created_at)
        return thread_runs[:limit]

    async def update_run_status(self, run_id: str, status: str, **kwargs) -> Optional[Run]:
        """Update run status and other fields."""
        run = self.runs.get(run_id)
        if not run:
            return None
        run.status = status
        current_time = int(time.time())
        if status == "in_progress" and not run.started_at:
            run.started_at = current_time
        elif status == "completed" and not run.completed_at:
            run.completed_at = current_time
        elif status == "failed" and not run.failed_at:
            run.failed_at = current_time
        elif status == "cancelled" and not run.cancelled_at:
            run.cancelled_at = current_time
        for key, value in kwargs.items():
            if hasattr(run, key):
                setattr(run, key, value)
        return run

    async def cancel_run(self, run_id: str) -> Optional[Run]:
        """Cancel a run."""
        return await self.update_run_status(run_id, "cancelled")

    async def create_run_step(
        self,
        run_id: str,
        assistant_id: str,
        thread_id: str,
        step_type: RunStepType,
        step_details: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunStep:
        """Create a new run step."""
        step_id = f"step_{uuid.uuid4().hex}"
        created_at = int(time.time())
        step = RunStep(
            id=step_id,
            created_at=created_at,
            run_id=run_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            type=step_type,
            status="in_progress",
            step_details=step_details,
            metadata=metadata or {},
        )
        self.steps[step_id] = step
        return step

    async def get_run_step(self, step_id: str) -> Optional[RunStep]:
        """Get a step by ID."""
        return self.steps.get(step_id)

    async def list_run_steps(self, run_id: str, limit: int = 20, order: str = "desc") -> List[RunStep]:
        """List steps for a run."""
        run_steps = [step for step in self.steps.values() if step.run_id == run_id]
        if order == "desc":
            run_steps = sorted(run_steps, key=lambda s: s.created_at, reverse=True)
        else:
            run_steps = sorted(run_steps, key=lambda s: s.created_at)
        return run_steps[:limit]

    async def update_run_step(
        self,
        step_id: str,
        status: Optional[str] = None,
        step_details: Optional[Dict[str, Any]] = None,
        last_error: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[RunStep]:
        """Update a run step."""
        step = self.steps.get(step_id)
        if not step:
            return None
        current_time = int(time.time())
        if status:
            step.status = status
            if status == "completed" and not step.completed_at:
                step.completed_at = current_time
            elif status == "failed" and not step.failed_at:
                step.failed_at = current_time
            elif status == "cancelled" and not step.cancelled_at:
                step.cancelled_at = current_time
        if step_details:
            step.step_details = step_details
        if last_error:
            step.last_error = last_error
        if usage:
            step.usage = usage
        for key, value in kwargs.items():
            if hasattr(step, key):
                setattr(step, key, value)
        return step