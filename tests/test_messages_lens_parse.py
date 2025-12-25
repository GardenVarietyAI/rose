from fastapi.testclient import TestClient


def test_messages_generate_assistant_parses_at_name(client: TestClient) -> None:
    lens_response = client.post(
        "/v1/lenses",
        data={"at_name": "socrates", "label": "Socrates", "system_prompt": "Use the socratic method."},
        headers={"Accept": "application/json"},
    )
    assert lens_response.status_code == 200
    lens_id = lens_response.json()["uuid"]

    created = client.post("/v1/threads", json={"messages": [{"role": "user", "content": "hi @socrates"}]})
    assert created.status_code == 200
    thread_id = created.json()["thread_id"]

    generated = client.post("/v1/messages", json={"thread_id": thread_id, "generate_assistant": True})
    assert generated.status_code == 200

    thread = client.get(f"/v1/threads/{thread_id}")
    assert thread.status_code == 200
    responses = thread.json()["responses"]
    assert responses
    response_meta = responses[0]["meta"] or {}
    assert response_meta.get("lens_id") == lens_id
    assert response_meta.get("lens_at_name") == "socrates"
