# mypy: ignore-errors

"""Tests for vector store chunking functionality."""

import io

from fastapi.testclient import TestClient


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
        "maintenance and quality control mechanisms. " * 5
    )  # Repeat 5 times to ensure chunking

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
        f"/v1/vector_stores/{vector_store_id}/search", json={"query": "machine learning", "k": 10}
    )
    assert response.status_code == 200
    search_results = response.json()

    # Verify chunking occurred
    assert len(search_results["data"]) > 1  # Should have multiple chunks

    # Check first result has proper metadata
    first_result = search_results["data"][0]
    metadata = first_result["metadata"]

    assert "file_id" in metadata
    assert "filename" in metadata
    assert "total_chunks" in metadata
    assert "start_index" in metadata
    assert "end_index" in metadata
    assert metadata["total_chunks"] > 1  # Should be chunked
    assert metadata["start_index"] >= 0
    assert metadata["end_index"] > metadata["start_index"]

    # Verify similarity scores are reasonable
    assert 0.0 <= first_result["score"] <= 1.0

    # Verify all chunks have consistent metadata
    total_chunks = metadata["total_chunks"]
    for result in search_results["data"]:
        assert result["metadata"]["total_chunks"] == total_chunks
        assert result["metadata"]["file_id"] == file_id
