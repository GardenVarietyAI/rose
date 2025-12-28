from fastapi.testclient import TestClient


def test_import_messages_inserts_and_skips_duplicates(client: TestClient) -> None:
    external_thread_id = "import-thread-1"

    payload = {
        "import_source": "claude_code_jsonl",
        "messages": [
            {
                "thread_id": external_thread_id,
                "role": "user",
                "content": "hello from import",
                "model": None,
                "created_at": 100,
                "import_external_id": "external-message-id-1",
                "meta": {"imported_external_id": "external-uuid-1", "imported_external_session_id": "session-1"},
            },
            {
                "thread_id": external_thread_id,
                "role": "user",
                "content": "duplicate record, should be ignored",
                "model": None,
                "created_at": 101,
                "import_external_id": "external-message-id-1",
                "meta": {"imported_external_id": "external-uuid-2", "imported_external_session_id": "session-1"},
            },
        ],
    }
    response = client.post("/v1/import/messages", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["imported"] == 1
    assert data["skipped_duplicates"] == 1

    threads = client.get("/v1/threads", params={"import_source": "claude_code_jsonl", "limit": 100})
    assert threads.status_code == 200
    thread_list = threads.json()["threads"]
    assert len(thread_list) == 1
    thread_id = thread_list[0]["thread_id"]

    messages = client.get("/v1/messages", params={"thread_id": thread_id})
    assert messages.status_code == 200
    message_list = messages.json()["messages"]
    assert len(message_list) == 1

    inserted = message_list[0]
    assert inserted["thread_id"] == thread_id
    assert inserted["role"] == "user"
    assert inserted["content"] == "hello from import"
    assert inserted["import_external_id"] == "external-message-id-1"
    assert isinstance(inserted["import_batch_id"], str) and inserted["import_batch_id"]
    assert inserted["meta"]["import_source"] == "claude_code_jsonl"
    assert inserted["meta"]["imported_external_id"] == "external-uuid-1"
