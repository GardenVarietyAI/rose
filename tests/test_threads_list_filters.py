from fastapi.testclient import TestClient


def test_threads_list_filters_has_assistant(client: TestClient) -> None:
    created = client.post("/v1/threads", json={"messages": [{"role": "user", "content": "local thread"}]})
    local_thread_id = created.json()["thread_id"]

    external_thread_id = "import-thread-2"
    client.post(
        "/v1/import/messages",
        json={
            "import_source": "claude_code_jsonl",
            "messages": [
                {
                    "thread_id": external_thread_id,
                    "role": "user",
                    "content": "imported thread prompt",
                    "model": None,
                    "created_at": 200,
                    "import_external_id": "external-message-id-2",
                    "meta": {"imported_external_id": "external-uuid-3", "imported_external_session_id": "session-2"},
                }
            ],
        },
    )

    threads_without_assistant_response = client.get("/v1/threads", params={"has_assistant": "false", "limit": 100})
    threads_without_assistant = threads_without_assistant_response.json()["threads"]
    imported_thread = next(
        (t for t in threads_without_assistant if t.get("import_source") == "claude_code_jsonl"), None
    )
    assert imported_thread is not None
    imported_thread_id = imported_thread["thread_id"]

    thread_ids_without_assistant = {t["thread_id"] for t in threads_without_assistant}
    assert imported_thread_id in thread_ids_without_assistant
    assert local_thread_id not in thread_ids_without_assistant

    threads_with_assistant_response = client.get("/v1/threads", params={"has_assistant": "true", "limit": 100})
    thread_ids_with_assistant = {t["thread_id"] for t in threads_with_assistant_response.json()["threads"]}
    assert local_thread_id in thread_ids_with_assistant
    assert imported_thread_id not in thread_ids_with_assistant


def test_threads_list_filters_import_source(client: TestClient) -> None:
    external_thread_id = "import-thread-3"
    client.post(
        "/v1/import/messages",
        json={
            "import_source": "claude_code_jsonl",
            "messages": [
                {
                    "thread_id": external_thread_id,
                    "role": "user",
                    "content": "imported thread prompt",
                    "model": None,
                    "created_at": 300,
                    "import_external_id": "external-message-id-3",
                    "meta": {"imported_external_id": "external-uuid-4", "imported_external_session_id": "session-3"},
                }
            ],
        },
    )

    response = client.get("/v1/threads", params={"import_source": "claude_code_jsonl", "limit": 100})
    threads = response.json()["threads"]
    assert len(threads) > 0
    assert all(t["import_source"] == "claude_code_jsonl" for t in threads)
