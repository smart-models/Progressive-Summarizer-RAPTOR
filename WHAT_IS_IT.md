
![What is it?](what-is-it.jpg)

# Technical Deep Dive: How RAPTOR Works

The Progressive Summarizer RAPTOR (Recursive API for Progressive Text Organization and Refinement) is an innovative text summarization system that creates hierarchical summaries through recursive refinement. It condenses documents while preserving essential information across multiple levels of abstraction, enabling users to navigate between different levels of detail seamlessly.

## Core Concepts

### Progressive Summarization

Unlike traditional summarization approaches that generate a single condensed version, RAPTOR creates a hierarchy of summaries at different levels of detail. Each level builds upon the previous, producing increasingly concise representations of the original text while maintaining semantic coherence.

The progressive approach has several advantages:
- Allows users to "zoom" between different levels of abstraction
- Preserves more context than single-pass summarization
- Creates natural information hierarchy for better document understanding

### Recursive Refinement

RAPTOR's name reflects its core technique: applying summarization recursively to its own outputs. This process is similar to how humans might first create a rough summary and then further refine it through multiple revision passes.

## Technical Architecture

### 1. API Layer (FastAPI)

The system is built around a FastAPI application (`raptor_api.py`) that provides:
- RESTful endpoints for text submission and processing
- Swagger documentation via OpenAPI
- Efficient resource management using FastAPI's lifespan context managers

The API uses FastAPI's lifespan context manager for embedding model management, preloading models at startup and cleaning up resources on shutdown. This improves performance by keeping models in memory during the application lifecycle.

### 2. Embedding Engine

RAPTOR leverages sentence transformers to convert text segments into high-dimensional vector representations that capture semantic meaning. These embeddings are then used to:
- Identify semantically important parts of the text
- Measure similarity between different segments
- Guide the summarization process to maintain coherence

The embedding model management system:
- Handles GPU placement when available
- Implements caching for improved performance
- Provides graceful fallbacks for error scenarios

### 3. Summarization Pipeline

The core summarization flow consists of several stages:

1. **Input Processing**: Accepting JSON documents with text chunks through the `/raptor/` endpoint
2. **Embedding Generation**: Converting each segment to vector representation using sentence transformers
3. **Semantic Clustering (Level 1)**: Grouping semantically related segments through dimensionality reduction and clustering
4. **Initial Summarization**: Creating first-level summaries for each cluster using the LLM
5. **Recursive Clustering (Level 2)**: Clustering Level 1 summaries to identify higher-level relationships
6. **Intermediate Summarization**: Generating second-level summaries from Level 2 clusters
7. **Final Consolidation (Level 3)**: Combining Level 2 summaries to create a comprehensive final summary
8. **Token Optimization**: Ensuring summaries stay within configurable token limits
9. **Hierarchical Output**: Returning all three levels with detailed metadata

### 4. LLM Integration

RAPTOR connects with Ollama for LLM capabilities, making use of template-based prompting to guide the summarization process. The system uses environment variables like `OLLAMA_BASE_URL` to configure the LLM endpoint, making deployment flexible across different environments.

The summarization prompts are designed to produce consistent, high-quality outputs, with careful attention to template string formatting to ensure proper content insertion at runtime. The system supports custom prompt templates through the API, allowing users to tailor the summarization process to specific domains or requirements.

## Implementation Details

### Key Components

1. **Model Management**
   - Models are loaded at application startup and properly released on shutdown
   - The `get_model()` function dynamically handles GPU placement and error scenarios
   - Resource management is handled through FastAPI's lifecycle events

2. **Embedding Generation**
   - The `generate_embeddings()` function converts text to vector representations
   - Optimized for performance and resource utilization
   - Implements batching for efficient processing of large documents

3. **Summary Generation**
   - Uses carefully crafted template strings with placeholders for proper runtime evaluation
   - Contextual prompts guide the LLM to maintain document focus
   - Three-level hierarchical summarization process with progressive refinement
   - Token optimization to stay within configurable limits

4. **API Endpoints**
   - Primary POST `/raptor/` endpoint for document processing
   - Health check GET `/` endpoint for service status
   - RESTful design with comprehensive parameter validation
   - Stateless architecture for scalability

### Configuration and Environment

RAPTOR is designed for flexible deployment with configuration via environment variables:

- `OLLAMA_BASE_URL`: Configures the endpoint for LLM services (default: http://localhost:11434)
- `LOG_LEVEL`: Controls logging verbosity **(Docker only – not consumed by the Python code)**
- `LLM_MODEL`: Override the default LLM model (`qwen2.5:7b-instruct`) used for summarization
- `EMBEDDER_MODEL`: Override the default embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
- `TEMPERATURE`: Override the default sampling temperature (0.1)
- `CONTEXT_WINDOW`: Override the default LLM context window (25600)
- `RANDOM_SEED`: Set the random seed for reproducibility (default: 224)
- `MAX_WORKERS`: Number of parallel threads used for processing (default: 75 % of available CPU cores)

**Note:** The following variables are for Docker configuration only and do not affect the Python code.

The system supports both local deployment with Uvicorn and containerized deployment with Docker and docker-compose, with separate profiles for CPU and GPU environments.

## Performance Considerations

### Processing Efficiency

- Embedding models are cached to avoid repeated loading
- Batched processing where appropriate
- Parallel execution of independent tasks

- The stateless API design allows for horizontal scaling
- Docker configurations for both CPU and GPU environments
- Resource-aware processing adapts to available hardware

## API Usage

### Key Parameters

- `llm_model`: LLM model to use for summarization (default: qwen2.5:7b-instruct)
- `embedder_model`: Model for generating embeddings (default: sentence-transformers/all-MiniLM-L6-v2)
- `threshold_tokens`: Maximum token limit for summaries
- `temperature`: Controls randomness in LLM output (default: 0.1)
- `context_window`: Maximum context window size for LLM (default: 25600)
- `custom_prompt`: Optional custom prompt template for summarization

### Response Structure

The API returns a JSON structure containing:

- `chunks`: Array of summary objects with text, token count, cluster level, and ID
- `metadata`: Detailed processing information including input counts, cluster counts per level, reduction ratio, model names, and processing times

## Future Directions

- Enhanced semantic preservation metrics
- Multi-modal summarization capabilities
- Comparison-based summarization for document sets
- Custom fine-tuning options for domain-specific summarization

By understanding how RAPTOR works at a technical level, developers can better integrate, extend, and optimize its capabilities for specific use cases and deployment scenarios.