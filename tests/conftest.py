"""Pytest configuration and fixtures."""

import os
from typing import Generator

import pytest

# Disable authentication for tests
os.environ["ROSE_SERVER_AUTH_ENABLED"] = "false"

# Alternatively, we could set a test token:
# os.environ["ROSE_API_KEY"] = "test-key"


@pytest.fixture(autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Set up test environment variables."""
    # Store original values
    original_auth = os.environ.get("ROSE_SERVER_AUTH_ENABLED")

    # Disable auth for tests
    os.environ["ROSE_SERVER_AUTH_ENABLED"] = "false"

    yield

    # Restore original values
    if original_auth is not None:
        os.environ["ROSE_SERVER_AUTH_ENABLED"] = original_auth
    else:
        os.environ.pop("ROSE_SERVER_AUTH_ENABLED", None)
