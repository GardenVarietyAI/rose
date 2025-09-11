# mypy: ignore-errors

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

        assert response.status_code == 422

    def test_invalid_message_role(self, client: TestClient):
        """Test that invalid message role is rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen2.5-0.5b", "messages": [{"role": "invalid_role", "content": "Hello"}]},
        )

        assert response.status_code == 422

    def test_malformed_json_body(self, client: TestClient):
        """Test that malformed JSON returns proper error."""
        response = client.post(
            "/v1/chat/completions",
            content='{"model": "qwen2.5-0.5b", invalid json}',
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422
