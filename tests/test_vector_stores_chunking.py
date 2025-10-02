# mypy: ignore-errors

import io
import os

import pytest
from fastapi.testclient import TestClient

# Configuration
CONTENT_REPETITION_COUNT = 10


@pytest.mark.skipif(
    not os.path.exists("data/models/Qwen--Qwen3-Embedding-0.6B-GGUF"), reason="Embedding model not available"
)
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

    response = client.get(f"/v1/files/{file_id}")
    assert response.status_code == 200
    file_obj = response.json()

    # Add file to vector store (should trigger processing)
    response = client.post(f"/v1/vector_stores/{vector_store_id}/files", json={"file_id": file_id})
    assert response.status_code == 200
    vector_file = response.json()

    # Initial status should be "in_progress"
    assert vector_file["status"] == "in_progress"
