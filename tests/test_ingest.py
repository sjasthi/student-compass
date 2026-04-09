from unittest.mock import patch
from backend.rag.ingest import remove_blob_from_chroma

# Test: no nodes found
def test_remove_blob_no_nodes():
    with patch("backend.rag.ingest._get_chroma_collection") as mock_col:
        # Simulate Chroma returning no matching IDs
        mock_col.return_value.get.return_value = {"ids": []}

        removed = remove_blob_from_chroma("fake_blob")
        assert removed == 0

# Test: nodes found and deleted
def test_remove_blob_deletes_nodes():
    with patch("backend.rag.ingest._get_chroma_collection") as mock_col:
        # Simulate Chroma returning two IDs
        mock_col.return_value.get.return_value = {"ids": ["1", "2"]}

        removed = remove_blob_from_chroma("fake_blob")
        assert removed == 2
