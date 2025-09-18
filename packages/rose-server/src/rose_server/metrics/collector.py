import logging
from typing import Optional

from rose_server.events.event_types import LLMEvent, ResponseCompleted, ResponseStarted, TokenGenerated
from rose_server.metrics.timing import (
    PerformanceMetrics,
    TimingData,
    TokenCounts,
    calculate_performance_metrics,
    current_time,
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self, model: str):
        self.model = model
        self.timing = TimingData(request_start=current_time())
        self.token_counts: Optional[TokenCounts] = None
        self.first_token_recorded = False
        self.quality_score: Optional[float] = None

    def process_event(self, event: LLMEvent) -> None:
        if isinstance(event, ResponseStarted):
            self.token_counts = TokenCounts(
                input_tokens=event.input_tokens,
                output_tokens=0,  # Will be updated on completion
            )

        elif isinstance(event, TokenGenerated) and not self.first_token_recorded:
            self.timing.first_token_time = current_time()
            self.first_token_recorded = True

        elif isinstance(event, ResponseCompleted):
            self.timing.completion_time = current_time()
            if self.token_counts:
                self.token_counts.output_tokens = event.output_tokens or 0

    def set_quality_score(self, score: float) -> None:
        self.quality_score = score

    def get_metrics(self) -> Optional[PerformanceMetrics]:
        if not self.token_counts:
            return None

        metrics = calculate_performance_metrics(self.timing, self.token_counts)

        if metrics.ttfb_ms and metrics.total_time_ms:
            tokens_per_sec = f"{metrics.tokens_per_second:.1f}" if metrics.tokens_per_second else "N/A"
            quality_str = f", Quality: {self.quality_score:.2f}" if self.quality_score else ""
            logger.info(
                f"[METRICS] {self.model}, "
                f"TTFB: {metrics.ttfb_ms:.1f}ms, "
                f"Total: {metrics.total_time_ms:.1f}ms, "
                f"Tokens/sec: {tokens_per_sec}"
                f"{quality_str}"
            )

        return metrics
