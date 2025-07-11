"""Tests for chat completions endpoint error states."""

from fastapi.testclient import TestClient


class TestChatCompletionsErrors:
    """Test error states for chat completions endpoint."""

    def test_invalid_model(self, client: TestClient):
        """Test that invalid model returns proper error."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "non-existent-model", "messages": [{"role": "user", "content": "Hello"}]},
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "model"
        assert error["code"] == "model_not_found"
        assert "non-existent-model" in error["message"]

    def test_missing_messages(self, client: TestClient):
        """Test that missing messages returns validation error."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-0.5b"
                # messages field missing
            },
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_invalid_message_role(self, client: TestClient):
        """Test that invalid message role is rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen2.5-0.5b", "messages": [{"role": "invalid_role", "content": "Hello"}]},
        )

        assert response.status_code == 422  # Pydantic validation

    def test_logprobs_with_streaming(self, client: TestClient):
        """Test that logprobs + streaming returns error."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-0.5b",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "logprobs": True,
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "logprobs"
        assert error["code"] == "invalid_parameter_combination"
        assert "streaming" in error["message"].lower()

    def test_invalid_top_logprobs(self, client: TestClient):
        """Test top_logprobs validation."""
        # Test too high (max is 5)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-0.5b",
                "messages": [{"role": "user", "content": "Hello"}],
                "logprobs": True,
                "top_logprobs": 10,
            },
        )

        assert response.status_code == 422  # Pydantic validation

        # Test negative
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-0.5b",
                "messages": [{"role": "user", "content": "Hello"}],
                "logprobs": True,
                "top_logprobs": -1,
            },
        )

        assert response.status_code == 422  # Pydantic validation

    def test_code_interpreter_tool(self, client: TestClient):
        """Test that code_interpreter tool type is rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-0.5b",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "code_interpreter"}],
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "tools"
        assert error["code"] == "unsupported_tool_type"
        assert "code_interpreter" in error["message"]

    def test_web_search_tool(self, client: TestClient):
        """Test that web_search tool type is rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-0.5b",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "web_search"}],
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "tools"
        assert error["code"] == "unsupported_tool_type"
        assert "web_search" in error["message"]

    def test_malformed_json_body(self, client: TestClient):
        """Test that malformed JSON returns proper error."""
        response = client.post(
            "/v1/chat/completions",
            content='{"model": "qwen2.5-0.5b", invalid json}',
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422  # FastAPI JSON parse error
