# mypy: ignore-errors

import pytest

from rose_server.config.settings import settings
from rose_server.inference.client import InferenceClient


@pytest.mark.asyncio
async def test_connection_error_message():
    """Test that connection errors return a user-friendly message."""
    # Use a non-existent port to ensure connection failure
    client = InferenceClient(uri="ws://localhost:99999")

    with pytest.raises(RuntimeError) as exc_info:
        async for _ in client.stream_inference(
            model_name="test-model",
            model_config={},
            prompt="test prompt",
            generation_kwargs={},
            response_id="test-id",
        ):
            pass

    assert (
        str(exc_info.value) == "Unable to connect to inference worker. Please ensure the inference service is running."
    )


@pytest.mark.asyncio
async def test_configurable_uri():
    """Test that the client uses the configured URI."""
    # Test with custom URI
    custom_uri = "ws://custom-host:8888"
    client = InferenceClient(uri=custom_uri)
    assert client.uri == custom_uri

    # Test with default URI from config
    client_default = InferenceClient()
    assert client_default.uri == settings.inference_uri
