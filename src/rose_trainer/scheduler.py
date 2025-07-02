"""Job scheduler for the training worker using APScheduler."""

import logging
import signal
import sys
from typing import Any

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from rose_core.config.settings import settings
from rose_trainer.worker import process_next_training_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Scheduler for training jobs."""

    def __init__(self, interval_seconds: int = 30):
        """Initialize scheduler.

        Args:
            interval_seconds: How often to check for jobs (default: 30 seconds)
        """
        self.scheduler = BlockingScheduler()
        self.interval_seconds = interval_seconds
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on SIGINT/SIGTERM."""

        def shutdown(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, shutting down scheduler...")
            self.scheduler.shutdown(wait=True)
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

    def start(self) -> None:
        """Start the scheduler."""
        # Add job to check for training jobs
        self.scheduler.add_job(
            func=process_next_training_job,
            trigger=IntervalTrigger(seconds=self.interval_seconds),
            id="process_training_jobs",
            name="Process training jobs",
            max_instances=1,  # Prevent overlapping runs
            coalesce=True,  # If multiple runs are pending, only run once
            misfire_grace_time=30,  # Allow 30 seconds late before skipping
        )

        logger.info(f"Training scheduler started - checking every {self.interval_seconds} seconds")
        logger.info("Press Ctrl+C to stop")

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")


def main() -> None:
    """Entry point for the scheduler."""
    # Use interval from settings
    scheduler = TrainingScheduler(interval_seconds=settings.training_interval)
    scheduler.start()


if __name__ == "__main__":
    main()
