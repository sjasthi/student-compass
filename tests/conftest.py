import pytest
from unittest.mock import patch, MagicMock
from backend.rag.gcs_upload import app

# Automatically mock GCS for all tests
@pytest.fixture(autouse=True)
def mock_gcs():
    with patch("backend.rag.gcs_upload.storage.Client") as mock_client:
        mock_client.return_value = MagicMock()
        yield

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
