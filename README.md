![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green)
![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-blue)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

# Progressive-Summarizer-RAPTOR

Progressive-Summarizer-RAPTOR (Recursive API for Progressive Text Organization and Refinement) is an advanced text summarization system that recursively condenses documents while preserving essential information through progressive levels of abstraction.

## Key Features

- **Progressive Summarization**: Creates multiple levels of summarization for the same text, with each level more condensed than the previous
- **Semantic Understanding**: Leverages embedding models to maintain semantic coherence during summarization
- **Flexible Deployment**: Supports both CPU and GPU acceleration via Docker
- **Configurable**: Easily adjust API endpoints, model parameters, and summarization depth
- **Integration-Ready**: Clean REST API interface for seamless integration with other systems

## Architecture

RAPTOR uses a recursive approach to summarization:

1. Text is processed to generate embeddings using sentence transformers
2. Initial summaries are created using embedded semantic understanding
3. Summaries are recursively refined through multiple passes
4. Each pass produces a more condensed version that preserves key information

## Use Cases

- **Document Analysis**: Quickly extract the essence of long documents
- **Content Triage**: Identify key points in large text collections
- **Information Hierarchy**: Navigate between different levels of detail as needed
- **Research Assistance**: Condense academic papers while preserving core findings

## Requirements

- Python 3.10+
- FastAPI
- Sentence Transformers
- CUDA-compatible GPU (optional, for acceleration)
- Ollama (for local LLM deployment)

## Quick Start

### Using Docker (Recommended)

```bash
# CPU Mode
docker-compose -f docker/docker-compose.yml --profile cpu up

# GPU Mode (requires NVIDIA Docker support)
docker-compose -f docker/docker-compose.yml --profile gpu up
```

### Manual Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn raptor_api:app --host 0.0.0.0 --port 8000
```

## Environment Variables

RAPTOR supports several environment variables for configuration:

- `OLLAMA_API_URL`: URL for Ollama API (default: "http://localhost:11434/api/generate")
- `LOG_LEVEL`: Logging verbosity (default: "INFO")
- `APP_PORT`: Port for the FastAPI application (default: 8000)

## API Documentation

Once running, navigate to `http://localhost:8000/docs` for comprehensive Swagger documentation of all available endpoints.

## Performance Considerations

- GPU acceleration is recommended for processing large documents
- Model loading occurs at startup using FastAPI's lifespan context management
- Embedding models are cached in memory for improved performance

## License

[MIT License](LICENSE)

## Further Reading

See [WHAT_IS_IT.md](WHAT_IS_IT.md) for a deeper technical dive into how RAPTOR works.