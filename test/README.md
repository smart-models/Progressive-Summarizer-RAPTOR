# Test Suite for Progressive-Summarizer-RAPTOR

This directory contains tests for the Progressive-Summarizer-RAPTOR API.

## Structure

- `test_api.py`: Main test file that tests API endpoints functionality
- `pytest.ini`: Configuration file for pytest
- `test_data/`: Directory for sample text files used in tests
- `models/`: Directory where test embedding models can be cached
- `logs/`: Directory for test-specific logs

## Running Tests

To run the tests, execute the following command from the project root:

```bash
python -m pytest test/test_api.py -v
```

## Test Coverage

The test suite covers:

1. **Health Endpoint Test**: Verifies that the API is running and healthy
2. **Text Summarization Test**: Tests the direct text summarization endpoint
3. **File Summarization Test**: Tests the ability to upload and summarize a file

## Test Dependencies

Tests require:
- pytest
- FastAPI TestClient
- An active Ollama instance (defaults to http://localhost:11434/api/generate)

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