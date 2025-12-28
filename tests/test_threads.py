from fastapi.testclient import TestClient


def test_threads_create_saves_user_message_without_generation(client: TestClient) -> None:
    created = client.post("/v1/threads", json={"messages": [{"role": "user", "content": "hello"}]})
    thread_id = created.json()["thread_id"]

    thread = client.get(f"/v1/threads/{thread_id}")
    payload = thread.json()
    assert payload["prompt"]["role"] == "user"
    assert payload["prompt"]["content"] == "hello"
    assert len(payload["responses"]) == 1
    assert payload["responses"][0]["role"] == "assistant"
    assert payload["responses"][0]["content"] == "hello"
