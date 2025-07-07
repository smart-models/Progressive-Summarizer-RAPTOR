import pytest
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raptor_api import app

# Directory for test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Create test data directory and ensure test file exists."""
    # Create test data directory if it doesn't exist
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Path to the test file
    alice_path = os.path.join(TEST_DATA_DIR, "alice_in_wonderland.json")

    # Check if test file exists, if not raise an error
    if not os.path.exists(alice_path):
        raise FileNotFoundError(
            f"Required test file not found: {alice_path}. Please ensure the Alice in Wonderland JSON file exists in the {TEST_DATA_DIR} directory."
        )

    yield  # Run the tests


@pytest.fixture
def client():
    """Create a test client with actual embedder."""
    with TestClient(app) as test_client:
        yield test_client


def test_alice_file_processing(client):
    """Test processing alice_in_wonderland.json and validate response structure."""
    # Path to the test file
    alice_path = os.path.join(TEST_DATA_DIR, "alice_in_wonderland.json")

    # Open the file for sending to the API
    with open(alice_path, "rb") as f:
        # Send request to the API with minimal required parameters to avoid None values
        response = client.post(
            "/raptor/",
            files={"file": ("alice_in_wonderland.json", f, "application/json")},
            params={
                "llm_model": "gemma3:4b",
                "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
                "temperature": 0.3,
                "context_window": 18432,
            },
        )

    # Check response status code
    assert response.status_code == 200, f"API returned error: {response.text}"

    # Parse the response
    data = response.json()

    # Basic structure validation
    assert "chunks" in data, "Response missing 'chunks' key"
    assert "metadata" in data, "Response missing 'metadata' key"

    # Validate chunks structure
    chunks = data["chunks"]
    assert isinstance(chunks, list), "'chunks' should be a list"
    assert len(chunks) > 0, "No chunks were generated"

    # Validate metadata structure
    metadata = data["metadata"]

    # Check basic metadata fields
    required_metadata_fields = [
        "input_chunks",
        "level_1_clusters",
        "level_2_clusters",
        "level_3_clusters",
        "total_clusters",
        "reduction_ratio",
        "llm_model",
        "embedder_model",
        "temperature",
        "context_window",
        "custom_prompt_used",
        "processing_time",
    ]

    for field in required_metadata_fields:
        assert field in metadata, f"Missing required metadata field: {field}"

    # Validate processing_time structure
    processing_time = metadata["processing_time"]
    required_processing_time_fields = ["total", "level_1", "level_2", "level_3"]
    for field in required_processing_time_fields:
        assert field in processing_time, (
            f"Missing required processing_time field: {field}"
        )

    # Validate chunks structure
    for chunk in chunks:
        assert "text" in chunk, "Chunk missing 'text' field"
        assert "token_count" in chunk, "Chunk missing 'token_count' field"
        assert "cluster_level" in chunk, "Chunk missing 'cluster_level' field"
        assert "id" in chunk, "Chunk missing 'id' field"

        # Validate basic values
        assert len(chunk["text"]) > 0, "Chunk text should not be empty"
        assert chunk["token_count"] > 0, "Chunk token_count should be positive"
        assert chunk["cluster_level"] in [1, 2, 3], (
            "Chunk cluster_level should be 1, 2, or 3"
        )
        assert chunk["id"] > 0, "Chunk id should be positive"

    # Validate basic metadata values
    assert metadata["input_chunks"] > 0, "input_chunks should be positive"
    assert metadata["total_clusters"] >= 0, "total_clusters should be non-negative"
    assert metadata["level_1_clusters"] >= 0, "level_1_clusters should be non-negative"
    assert metadata["level_2_clusters"] >= 0, "level_2_clusters should be non-negative"
    assert metadata["level_3_clusters"] >= 0, "level_3_clusters should be non-negative"

    # Check that the total_clusters equals the sum of level clusters
    expected_total = (
        metadata["level_1_clusters"]
        + metadata["level_2_clusters"]
        + metadata["level_3_clusters"]
    )
    assert metadata["total_clusters"] == expected_total, (
        f"total_clusters ({metadata['total_clusters']}) should equal sum of level clusters ({expected_total})"
    )

    # Check that the number of chunks matches expected distribution across levels
    level_1_chunks = len([c for c in chunks if c["cluster_level"] == 1])
    level_2_chunks = len([c for c in chunks if c["cluster_level"] == 2])
    level_3_chunks = len([c for c in chunks if c["cluster_level"] == 3])

    assert level_1_chunks == metadata["level_1_clusters"], (
        f"Number of level 1 chunks ({level_1_chunks}) doesn't match metadata ({metadata['level_1_clusters']})"
    )
    assert level_2_chunks == metadata["level_2_clusters"], (
        f"Number of level 2 chunks ({level_2_chunks}) doesn't match metadata ({metadata['level_2_clusters']})"
    )
    assert level_3_chunks == metadata["level_3_clusters"], (
        f"Number of level 3 chunks ({level_3_chunks}) doesn't match metadata ({metadata['level_3_clusters']})"
    )

    # Check chunk IDs are unique and sequential
    chunk_ids = [chunk["id"] for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"
    assert chunk_ids == list(range(1, len(chunks) + 1)), (
        "Chunk IDs should be sequential starting from 1"
    )

    # Log some information for debugging
    print(f"Successfully processed {metadata['input_chunks']} input chunks")
    print(
        f"Generated {metadata['total_clusters']} total clusters: L1={metadata['level_1_clusters']}, L2={metadata['level_2_clusters']}, L3={metadata['level_3_clusters']}"
    )
    print(f"Reduction ratio: {metadata['reduction_ratio']:.2f}")
    print(
        f"Model used: {metadata['llm_model']} (LLM), {metadata['embedder_model']} (Embedder)"
    )
    print(f"Processing time: {metadata['processing_time']['total']:.2f}s total")
