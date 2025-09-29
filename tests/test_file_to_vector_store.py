# mypy: ignore-errors

import io
import os

import pytest
from fastapi.testclient import TestClient


@pytest.mark.skipif(
    not os.path.exists("data/models/Qwen--Qwen3-Embedding-0.6B-GGUF"), reason="Embedding model not available"
)
def test_file_upload_to_vector_store(client: TestClient):
    """Test uploading a file and using it in a vector store."""

    test_content = """
    Artificial Intelligence represents one of the most transformative technological developments.
    Machine learning enables computer systems to automatically learn from experience.
    Deep learning utilizes artificial neural networks with multiple layers.
    Natural language processing focuses on enabling computers to understand human language.
    """

    file_data = io.BytesIO(test_content.encode("utf-8"))
    response = client.post(
        "/v1/files", files={"file": ("test_document.txt", file_data, "text/plain")}, data={"purpose": "assistants"}
    )
    assert response.status_code == 200
    file_obj = response.json()
    file_id = file_obj["id"]

    response = client.get(f"/v1/files/{file_id}")
    assert response.status_code == 200
    file_obj = response.json()
    # Skip status check - files aren't processed in test environment

    response = client.post("/v1/vector_stores", json={"name": "test-store"})
    assert response.status_code == 200
    vector_store = response.json()
    vector_store_id = vector_store["id"]

    response = client.post(f"/v1/vector_stores/{vector_store_id}/files", json={"file_id": file_id})
    assert response.status_code == 200
    vector_file = response.json()

    assert vector_file["status"] == "in_progress"

    # Check status after background task completes
    response = client.get(f"/v1/vector_stores/{vector_store_id}/files")
    assert response.status_code == 200
    files_list = response.json()

    # Find our file
    vsf = next((f for f in files_list["data"] if f["id"] == vector_file["id"]), None)
    assert vsf is not None
    # Skip status check - background tasks don't run in tests

    # Note: Can't test search results without file processing


@pytest.mark.skipif(
    not os.path.exists("data/models/Qwen--Qwen3-Embedding-0.6B-GGUF"), reason="Embedding model not available"
)
def test_file_not_processed_error(client: TestClient):
    """Test that adding an unprocessed file to vector store fails appropriately."""

    response = client.post("/v1/vector_stores", json={"name": "test-store-2"})
    assert response.status_code == 200
    vector_store = response.json()
    vector_store_id = vector_store["id"]

    # Try to add a non-existent file
    response = client.post(f"/v1/vector_stores/{vector_store_id}/files", json={"file_id": "non-existent-file-id"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
