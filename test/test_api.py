import pytest
import sys
import os
import json
from fastapi.testclient import TestClient

# Add the parent directory to sys.path to import the raptor_api module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raptor_api import app

# Directory for test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

# Test environment variables
os.environ["OLLAMA_API_URL"] = "http://localhost:11434/api/generate"  # Use local test Ollama instance


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Create test data directory and ensure test file exists."""
    # Create test data directory if it doesn't exist
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Path to the test file - a sample article for summarization
    sample_path = os.path.join(TEST_DATA_DIR, "sample_article.txt")

    # Check if test file exists, if not create a simple one for testing
    if not os.path.exists(sample_path):
        sample_text = """
        # Artificial Intelligence: A Modern Approach

        Artificial Intelligence (AI) is revolutionizing how we interact with technology, from voice assistants to autonomous vehicles. 
        Machine learning, a subset of AI, enables systems to learn and improve from experience without explicit programming.
        Deep learning, a further subset of machine learning, utilizes neural networks with many layers to analyze various factors of data.
        
        Natural Language Processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language.
        Computer vision systems can identify objects, people, and activities in images and videos with increasing accuracy.
        
        AI ethics is concerned with ensuring AI systems are developed and deployed responsibly, with considerations for privacy, bias, transparency, and accountability.
        The future of AI may include general artificial intelligence that can perform any intellectual task a human can do.
        
        However, challenges exist in areas like ensuring privacy, eliminating bias, maintaining transparency, and establishing accountability in AI systems.
        Responsible AI development requires collaboration between technologists, policymakers, ethicists, and the public to maximize benefits while minimizing risks.
        """
        with open(sample_path, 'w') as f:
            f.write(sample_text)
        print(f"Created sample test article at {sample_path}")

    yield  # Run the tests


@pytest.fixture
def client():
    """Create a test client with the app."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_summarize_text(client):
    """Test the text summarization endpoint."""
    # Prepare test data
    test_text = """
    Artificial Intelligence (AI) is transforming our world in numerous ways. 
    Machine learning models can now recognize patterns in data more effectively than humans in many domains.
    Deep learning has revolutionized computer vision and natural language processing.
    Ethical considerations are increasingly important as AI systems become more powerful and widespread.
    """
    
    payload = {
        "text": test_text,
        "levels": 2  # Request 2 levels of summarization
    }
    
    # Send request to the API
    response = client.post("/summarize/text", json=payload)
    
    # Check response status code
    assert response.status_code == 200, f"API returned error: {response.text}"
    
    # Parse the response
    data = response.json()
    
    # Validate response structure
    assert "summaries" in data, "Response missing 'summaries' key"
    assert "metadata" in data, "Response missing 'metadata' key"
    
    summaries = data["summaries"]
    metadata = data["metadata"]
    
    # Validate metadata fields
    assert "model" in metadata, "Metadata missing 'model' field"
    assert "processing_time" in metadata, "Metadata missing 'processing_time' field"
    assert "levels_requested" in metadata, "Metadata missing 'levels_requested' field"
    assert "levels_generated" in metadata, "Metadata missing 'levels_generated' field"
    
    # Validate summaries
    assert len(summaries) > 0, "Should have at least one summary level"
    assert len(summaries) <= metadata["levels_requested"], "Should not have more levels than requested"
    
    # Validate types and content
    assert isinstance(metadata["levels_requested"], int)
    assert isinstance(metadata["levels_generated"], int)
    assert isinstance(metadata["processing_time"], (int, float))
    
    # Each level should be progressively shorter
    if len(summaries) > 1:
        for i in range(1, len(summaries)):
            assert len(summaries[i]["text"]) < len(summaries[i-1]["text"]), f"Level {i} should be shorter than level {i-1}"
    
    print(f"Successfully validated response with {len(summaries)} summary levels.")
    print(f"Processing time: {metadata['processing_time']:.2f} seconds")


def test_summarize_file(client):
    """Test processing a file for summarization."""
    # Path to the test file
    sample_path = os.path.join(TEST_DATA_DIR, "sample_article.txt")
    
    # Open the file for sending to the API
    with open(sample_path, "rb") as f:
        # Send request to the API
        response = client.post(
            "/summarize/file",
            files={"file": ("sample_article.txt", f, "text/plain")},
            data={"levels": 2}  # Request 2 levels of summarization
        )
    
    # Check response status code
    assert response.status_code == 200, f"API returned error: {response.text}"
    
    # Parse the response
    data = response.json()
    
    # Validate the top-level structure 
    assert "summaries" in data, "Response missing 'summaries' key"
    assert "metadata" in data, "Response missing 'metadata' key"
    
    summaries = data["summaries"]
    metadata = data["metadata"]
    
    # Validate metadata
    assert "filename" in metadata, "Metadata missing 'filename' field"
    assert metadata["filename"] == "sample_article.txt"
    assert "model" in metadata, "Metadata missing 'model' field"
    assert "levels_generated" in metadata, "Metadata missing 'levels_generated' field"
    
    # Validate summaries
    for i, summary in enumerate(summaries):
        assert "level" in summary, f"Summary {i} missing 'level' field"
        assert "text" in summary, f"Summary {i} missing 'text' field"
        assert isinstance(summary["level"], int), f"Summary {i} level should be an integer"
        assert isinstance(summary["text"], str), f"Summary {i} text should be a string"
        assert len(summary["text"]) > 0, f"Summary {i} text should not be empty"
    
    print(f"Successfully validated file summarization with {len(summaries)} levels.")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])