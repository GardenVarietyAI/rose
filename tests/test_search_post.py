from fastapi.testclient import TestClient


def test_search_post_returns_json(client: TestClient) -> None:
    created = client.post("/v1/threads", json={"messages": [{"role": "user", "content": "alpha beta"}]})
    assert created.status_code == 200

    response = client.post(
        "/v1/search",
        json={"content": "alpha", "exact": True, "limit": 5, "lens_ids": [], "factsheet_ids": []},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["index"] == "messages"
    assert data["query"] == "alpha"
    assert len(data["hits"]) >= 1


def test_search_post_returns_html(client: TestClient) -> None:
    created = client.post("/v1/threads", json={"messages": [{"role": "user", "content": "gamma delta"}]})
    assert created.status_code == 200

    response = client.post(
        "/v1/search",
        json={"content": "gamma", "exact": True, "limit": 5, "lens_ids": [], "factsheet_ids": []},
        headers={"Accept": "text/html"},
    )
    assert response.status_code == 200
    assert "Results:" in response.text


def test_search_with_lens_matches_user_query_within_lensed_threads(client: TestClient) -> None:
    lens_response = client.post(
        "/v1/lenses",
        data={"at_name": "socrates", "label": "Socrates", "system_prompt": "Use the socratic method."},
        headers={"Accept": "application/json"},
    )
    assert lens_response.status_code == 200
    lens_id = lens_response.json()["uuid"]

    created = client.post("/v1/threads", json={"messages": [{"role": "user", "content": "needle"}]})
    assert created.status_code == 200
    thread_id = created.json()["thread_id"]

    manual_assistant = client.post(
        "/v1/messages",
        json={
            "thread_id": thread_id,
            "content": "haystack",
            "meta": {"lens_id": lens_id, "lens_at_name": "socrates"},
        },
        headers={"Accept": "application/json"},
    )
    assert manual_assistant.status_code == 200

    response = client.post(
        "/v1/search",
        json={"content": "needle", "exact": True, "limit": 5, "lens_ids": [lens_id], "factsheet_ids": []},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["hits"]
