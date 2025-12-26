from fastapi.testclient import TestClient


def test_search_post_returns_json(client: TestClient) -> None:
    created = client.post("/v1/threads", json={"messages": [{"role": "user", "content": "alpha beta"}]})
    assert created.status_code == 200

    response = client.post("/v1/search", json={"q": "alpha", "exact": True, "limit": 5})
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
        json={"q": "gamma", "exact": True, "limit": 5},
        headers={"Accept": "text/html"},
    )
    assert response.status_code == 200
    assert "Results:" in response.text
