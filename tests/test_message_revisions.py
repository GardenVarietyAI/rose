from fastapi.testclient import TestClient


def test_message_revisions_update_prompt(client: TestClient) -> None:
    created = client.post("/v1/threads", json={"messages": [{"role": "user", "content": "hello"}]})
    assert created.status_code == 200
    thread_id = created.json()["thread_id"]

    thread = client.get(f"/v1/threads/{thread_id}")
    assert thread.status_code == 200
    prompt_uuid = thread.json()["prompt"]["uuid"]

    revised = client.post(f"/v1/messages/{prompt_uuid}/revisions", json={"content": "hello revised"})
    assert revised.status_code == 200

    updated_thread = client.get(f"/v1/threads/{thread_id}")
    assert updated_thread.status_code == 200
    assert updated_thread.json()["prompt"]["content"] == "hello revised"


def test_message_revisions_keep_latest_only_in_search(client: TestClient) -> None:
    created = client.post("/v1/threads", json={"messages": [{"role": "user", "content": "hello"}]})
    assert created.status_code == 200
    thread_id = created.json()["thread_id"]

    thread = client.get(f"/v1/threads/{thread_id}")
    assert thread.status_code == 200
    prompt_uuid = thread.json()["prompt"]["uuid"]

    revised = client.post(f"/v1/messages/{prompt_uuid}/revisions", json={"content": "hello revised"})
    assert revised.status_code == 200

    # Old prompt revision should not be searchable if FTS indexes latest-only revisions
    old_search = client.get("/v1/search", params={"q": "hello", "exact": True, "limit": 10})
    assert old_search.status_code == 200
    for hit in old_search.json()["hits"]:
        if hit["metadata"]["role"] == "user":
            assert hit["id"] != prompt_uuid

    new_search = client.get("/v1/search", params={"q": "revised", "exact": True, "limit": 10})
    assert new_search.status_code == 200
    assert len(new_search.json()["hits"]) == 1
