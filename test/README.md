# Test Suite for Progressive-Summarizer-RAPTOR

This directory contains tests for the Progressive-Summarizer-RAPTOR API.

## Structure

- `test_api.py`: Main test file that tests API endpoints functionality
- `pytest.ini`: Configuration file for pytest
- `test_data/`: Directory for sample text files used in tests
  - `alice_in_wonderland.json`: Required test data file
- `models/`: Directory where test embedding models can be cached
- `logs/`: Directory for test-specific logs

## Running Tests

To run the tests, execute the following command from the project root directory:

```bash
python -m pytest test/test_api.py -v -s
```

## Current Test Coverage

The test suite currently includes:

1. **File Summarization Test (`test_alice_file_processing`)**: 
   - Tests the `/raptor/` endpoint with a JSON file upload
   - Validates the response structure including chunks and metadata
   - Verifies correct clustering across three levels
   - Checks token counts, reduction ratios, and processing times
   

## Test Dependencies

Tests require:
- pytest
- FastAPI TestClient
- An active Ollama instance (defaults to http://localhost:11434)
- The `alice_in_wonderland.json` test file in the `test_data/` directory
- Qwen2.5:7b-instruct model available in Ollama

## Adding New Tests

When adding new tests:
1. Follow the existing pattern in `test_api.py`
2. Add test data files to the `test_data/` directory if needed
3. Run tests to verify they pass before committing changes

## CI/CD Integration

These tests can be incorporated into a CI/CD pipeline by running:

```bash
python -m pytest test/ --junitxml=test-results.xml
```

This will generate an XML report that can be consumed by CI/CD platforms.