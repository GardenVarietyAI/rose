import io
import pytest
from fastapi.testclient import TestClient


def test_file_upload_with_size_consistency(client: TestClient):
    """Test that file upload uses file.size for memory efficiency and handles size correctly."""
    # Create test file content
    test_content = b"This is test file content for BLOB storage"
    test_file = io.BytesIO(test_content)
    
    # Upload file
    response = client.post(
        "/v1/files",
        files={"file": ("test.txt", test_file, "text/plain")},
        data={"purpose": "assistants"}
    )
    
    assert response.status_code == 200
    file_data = response.json()
    
    # Verify the file size matches the actual content length
    assert file_data["bytes"] == len(test_content)
    assert file_data["filename"] == "test.txt"
    assert file_data["purpose"] == "assistants"
    assert file_data["object"] == "file"
    assert "id" in file_data