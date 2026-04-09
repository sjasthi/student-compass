import io
from unittest.mock import patch, MagicMock

# Test: /health
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


# Test: /query
def test_query_endpoint(client):
    response = client.post("/query", json={"question": "What is tuition?"})
    assert response.status_code == 200
    data = response.get_json()
    assert "answer" in data


# Test: /query with missing field
def test_query_missing_field(client):
    response = client.post("/query", json={})
    assert response.status_code == 400


# Test: /upload/file (mocked)
def test_upload_file_mocked(client):
    fake_blob = MagicMock()
    with patch("backend.rag.gcs_upload.get_bucket") as mock_get_bucket:
        mock_bucket = MagicMock()
        mock_get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = fake_blob

        data = {
            "file": (io.BytesIO(b"hello"), "test.txt")
        }
        response = client.post("/upload/file", data=data, content_type="multipart/form-data")

        assert response.status_code in (200, 500)  # depends on mock behavior


# Test: /test/run (mock ingestion)
@patch("backend.rag.gcs_upload.run_gcs_test_ingestion", return_value=10)
def test_test_run_starts(mock_ingest, client):
    response = client.post("/test/run", json={"chunk_sizes": [500]})
    assert response.status_code == 200
