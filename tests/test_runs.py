# mypy: ignore-errors

from fastapi.testclient import TestClient


class TestRunsErrors:
    """Test error paths for runs endpoints."""

    def _create_assistant(self, client: TestClient, model: str = "qwen2.5-0.5b") -> str:
        response = client.post(
            "/v1/assistants",
            json={
                "model": model,
                "name": "Test Assistant",
                "instructions": "You are a helpful assistant.",
            },
        )
        assert response.status_code == 200
        return response.json()["id"]

    def _create_thread(self, client: TestClient) -> str:
        response = client.post("/v1/threads", json={})
        assert response.status_code == 200
        return response.json()["id"]

    def test_create_run_invalid_thread(self, client: TestClient):
        assistant_id = self._create_assistant(client)
        response = client.post(
            "/v1/threads/thread_nonexistent/runs",
            json={"assistant_id": assistant_id},
        )
        assert response.status_code == 404
        assert "Thread not found" in response.json()["detail"]

    def test_create_run_invalid_assistant(self, client: TestClient):
        thread_id = self._create_thread(client)
        response = client.post(
            f"/v1/threads/{thread_id}/runs",
            json={"assistant_id": "asst_nonexistent"},
        )
        assert response.status_code == 404
        assert "Assistant not found" in response.json()["detail"]

    def test_create_run_missing_assistant_id(self, client: TestClient):
        thread_id = self._create_thread(client)
        response = client.post(
            f"/v1/threads/{thread_id}/runs",
            json={},  # Missing assistant_id
        )
        assert response.status_code == 422

    def test_list_runs_invalid_thread(self, client: TestClient):
        response = client.get("/v1/threads/thread_nonexistent/runs")
        assert response.status_code == 404
        assert "Thread not found" in response.json()["detail"]

    def test_get_run_invalid_thread_or_run(self, client: TestClient):
        thread_id = self._create_thread(client)
        response = client.get(f"/v1/threads/{thread_id}/runs/run_nonexistent")
        assert response.status_code == 404
        assert "Run not found" in response.json()["detail"]

    def test_cancel_run_invalid_thread_or_run(self, client: TestClient):
        thread_id = self._create_thread(client)
        response = client.post(f"/v1/threads/{thread_id}/runs/run_nonexistent/cancel")
        assert response.status_code == 404
        assert "Run not found" in response.json()["detail"]

    def test_submit_tool_outputs_missing_or_empty(self, client: TestClient):
        thread_id = self._create_thread(client)

        response = client.post(
            f"/v1/threads/{thread_id}/runs/run_nonexistent/submit_tool_outputs",
            json={},  # Missing
        )
        assert response.status_code == 400
        assert "tool_outputs required" in response.json()["detail"]

        response = client.post(
            f"/v1/threads/{thread_id}/runs/run_nonexistent/submit_tool_outputs",
            json={"tool_outputs": []},  # Empty
        )
        assert response.status_code == 400
        assert "tool_outputs required" in response.json()["detail"]

    def test_malformed_json_body(self, client: TestClient):
        thread_id = self._create_thread(client)
        response = client.post(
            f"/v1/threads/{thread_id}/runs",
            content='{"assistant_id": "test", invalid json}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_run_steps_invalid_thread_or_run(self, client: TestClient):
        response = client.get("/v1/threads/thread_nonexistent/runs/run_nonexistent/steps")
        assert response.status_code == 200
        assert response.json()["data"] == []

    def test_get_run_step_invalid(self, client: TestClient):
        response = client.get("/v1/threads/thread_nonexistent/runs/run_nonexistent/steps/step_nonexistent")
        assert response.status_code == 404
        assert "Step not found" in response.json()["detail"]
