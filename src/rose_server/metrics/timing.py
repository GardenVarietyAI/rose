import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimingData:
    request_start: float
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None


@dataclass
class TokenCounts:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class PerformanceMetrics:
    ttfb_ms: Optional[float]
    total_time_ms: Optional[float]
    tokens_per_second: Optional[float]

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "ttfb_ms": round(self.ttfb_ms, 2) if self.ttfb_ms else None,
            "total_time_ms": round(self.total_time_ms, 2) if self.total_time_ms else None,
            "tokens_per_second": round(self.tokens_per_second, 2) if self.tokens_per_second else None,
        }


def calculate_performance_metrics(timing: TimingData, tokens: TokenCounts) -> PerformanceMetrics:
    ttfb_ms = None
    total_time_ms = None
    tokens_per_second = None

    if timing.first_token_time is not None:
        ttfb_ms = (timing.first_token_time - timing.request_start) * 1000

    if timing.completion_time is not None:
        total_time_ms = (timing.completion_time - timing.request_start) * 1000

    # Calculate tokens/sec based on generation time (excluding TTFB)
    if ttfb_ms is not None and total_time_ms is not None and total_time_ms > ttfb_ms and tokens.output_tokens > 0:
        generation_time_ms = total_time_ms - ttfb_ms
        tokens_per_second = (tokens.output_tokens * 1000) / generation_time_ms

    return PerformanceMetrics(ttfb_ms=ttfb_ms, total_time_ms=total_time_ms, tokens_per_second=tokens_per_second)


def current_time() -> float:
    return time.perf_counter()
