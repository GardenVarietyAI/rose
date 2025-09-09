# mypy: ignore-errors

from rose_server.metrics import TimingData, TokenCounts, calculate_performance_metrics


def test_pure_timing_calculation():
    """Test that pure timing calculations work correctly."""
    # Simulate timing data
    timing = TimingData(
        request_start=0.0,
        first_token_time=0.1,  # 100ms TTFB
        completion_time=0.5,  # 500ms total
    )

    tokens = TokenCounts(input_tokens=10, output_tokens=20)

    metrics = calculate_performance_metrics(timing, tokens)

    assert metrics.ttfb_ms == 100.0
    assert metrics.total_time_ms == 500.0
    # Generation time = 500 - 100 = 400ms
    # Tokens/sec = 20 tokens * 1000 / 400ms = 50 tokens/sec
    assert abs(metrics.tokens_per_second - 50.0) < 0.1


def test_edge_cases():
    """Test edge cases in timing calculations."""
    # No first token time
    timing = TimingData(request_start=0.0, completion_time=0.5)
    tokens = TokenCounts(input_tokens=5, output_tokens=10)
    metrics = calculate_performance_metrics(timing, tokens)

    assert metrics.ttfb_ms is None
    assert metrics.tokens_per_second is None


def test_partial_metrics_without_completion():
    """Requesting metrics before completion returns partial data."""
    timing = TimingData(request_start=0.0, first_token_time=0.1)
    tokens = TokenCounts(input_tokens=5, output_tokens=10)
    metrics = calculate_performance_metrics(timing, tokens)

    assert metrics.ttfb_ms == 100.0
    assert metrics.total_time_ms is None
    assert metrics.tokens_per_second is None


def test_api_response_format():
    """Test that metrics convert to API response format correctly."""
    timing = TimingData(request_start=0.0, first_token_time=0.123, completion_time=0.456)
    tokens = TokenCounts(input_tokens=10, output_tokens=15)
    metrics = calculate_performance_metrics(timing, tokens)

    api_dict = metrics.to_dict()

    assert api_dict["ttfb_ms"] == 123.0
    assert api_dict["total_time_ms"] == 456.0
    assert "tokens_per_second" in api_dict
