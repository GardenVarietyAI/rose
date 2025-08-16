# mypy: ignore-errors

"""Tests for vector store chunking functionality."""

import io

from fastapi.testclient import TestClient

CONTENT_REPETITION_COUNT = 5


def test_document_chunking_via_api(client: TestClient):
    """Test document chunking through the API endpoints."""
    # Create vector store
    response = client.post("/v1/vector_stores", json={"name": "chunking-test"})
    assert response.status_code == 200
    vector_store = response.json()
    vector_store_id = vector_store["id"]

    # Create a large test document that should be chunked (repeated content)
    large_content = (
        "Artificial Intelligence represents one of the most transformative "
        "technological developments of the 21st century. "
        "Machine learning enables computer systems to automatically learn "
        "and improve from experience without being explicitly programmed. "
        "Deep learning utilizes artificial neural networks with multiple "
        "layers to process complex, high-dimensional data. "
        "Natural language processing focuses on enabling computers to "
        "understand, interpret, and generate human language. "
        "Computer vision algorithms allow machines to interpret and analyze "
        "visual information from the world around them. "
        "Reinforcement learning represents a paradigm where AI agents learn "
        "optimal strategies through trial and error. "
        "The healthcare industry has witnessed remarkable transformations "
        "through AI applications in medical imaging and drug discovery. "
        "Financial services have embraced AI for fraud detection, "
        "algorithmic trading, and risk assessment. "
        "Autonomous vehicles combine computer vision, sensor fusion, "
        "real-time decision making, and advanced control systems. "
        "Manufacturing has been revolutionized by AI-powered predictive "
        "maintenance and quality control mechanisms. " * CONTENT_REPETITION_COUNT
    )  # Repeat to ensure chunking

    # Upload file
    file_data = io.BytesIO(large_content.encode("utf-8"))
    response = client.post(
        "/v1/files", files={"file": ("test_chunking.txt", file_data, "text/plain")}, data={"purpose": "assistants"}
    )
    assert response.status_code == 200
    file_obj = response.json()
    file_id = file_obj["id"]

    # Add file to vector store (should trigger chunking)
    response = client.post(f"/v1/vector_stores/{vector_store_id}/files", json={"file_id": file_id})
    assert response.status_code == 200
    vector_file = response.json()
    assert vector_file["status"] == "completed"

    # Search the vector store
    response = client.post(
        f"/v1/vector_stores/{vector_store_id}/search", json={"query": "machine learning", "max_num_results": 10}
    )
    assert response.status_code == 200
    search_results = response.json()

    # max_num_results should be respected
    assert len(search_results["data"]) <= 10

    # All scores should be numeric
    scores = [r["score"] for r in search_results["data"]]
    assert all(isinstance(s, (int, float)) for s in scores)

    # Verify chunking occurred
    assert len(search_results["data"]) > 1  # Should have multiple chunks

    # Check first result has proper structure
    first_result = search_results["data"][0]

    assert "file_id" in first_result
    assert "filename" in first_result
    assert "score" in first_result
    assert "content" in first_result
    assert first_result["filename"] == "test_chunking.txt"
    assert first_result["file_id"] == file_id

    # Verify similarity scores are reasonable
    assert isinstance(first_result["score"], (int, float))
    assert first_result["score"] is not None

    # Verify content structure
    assert isinstance(first_result["content"], list)
    assert len(first_result["content"]) > 0
    assert first_result["content"][0]["type"] == "text"
    assert len(first_result["content"][0]["text"]) > 0

    # Verify all chunks have consistent structure
    for result in search_results["data"]:
        assert result["file_id"] == file_id
        assert result["filename"] == "test_chunking.txt"
        assert isinstance(result["score"], (int, float))
        assert isinstance(result["content"], list)

    # Verify that smaller max_num_results limits results appropriately
    response = client.post(
        f"/v1/vector_stores/{vector_store_id}/search", json={"query": "machine learning", "max_num_results": 2}
    )
    assert response.status_code == 200
    limited_results = response.json()["data"]
    assert len(limited_results) == 2
