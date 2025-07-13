import json
import random
import time
import multiprocessing
import logging
import re
import numpy as np
import pandas as pd
import umap
import warnings
import torch
import tiktoken
import requests
import os
import threading
import gc
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from numba.core.errors import NumbaWarning
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from tqdm import tqdm
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from functools import lru_cache


# Define Settings class for configuration
class Settings(BaseSettings):
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"

    # Model settings
    llm_model: str = "gemma3:4b"
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Generation settings
    temperature: float = 0.3
    context_window: int = 18432

    # Performance settings
    random_seed: int = 224
    max_workers: int = max(
        1, min(multiprocessing.cpu_count() - 1, int(multiprocessing.cpu_count() * 0.75))
    )

    # File settings
    allowed_extensions: set = {"json"}

    class Config:
        case_sensitive = False
        extra = "ignore"

    @property
    def ollama_api_generate_url(self) -> str:
        """URL for the Ollama generate API endpoint"""
        # Ensure no space in URL by using urljoin or careful string manipulation
        base = self.ollama_base_url.rstrip("/")
        return f"{base}/api/generate"

    @property
    def ollama_api_tags_url(self) -> str:
        """URL for the Ollama tags API endpoint"""
        # Ensure no space in URL by using urljoin or careful string manipulation
        base = self.ollama_base_url.rstrip("/")
        return f"{base}/api/tags"

    @property
    def ollama_api_pull_url(self) -> str:
        """URL for the Ollama pull API endpoint"""
        # Ensure no space in URL by using urljoin or careful string manipulation
        base = self.ollama_base_url.rstrip("/")
        return f"{base}/api/pull"


# Create settings instance
@lru_cache()
def get_settings():
    """
    Returns a cached settings instance.
    Using lru_cache ensures settings are loaded only once during the application lifecycle.
    """
    return Settings()


# Define prompt template for summarization
PROMPT_TEMPLATE = """
    Act as an expert technical writer specializing in creating concise, accurate, and objective summaries.
    Summarize the following text (delimited by lines containing only dashes) according to these guidelines:

    1. CORE REQUIREMENTS:
    - Extract all key facts, arguments, and essential details.
    - Preserve technical terms, numbers, and data points exactly as in the original.
    - Maintain the chronological flow and causal relationships present in the text.
    - Use only information explicitly stated in the text.

    2. FORMATTING:
    - Write as a cohesive narrative using neutral, objective language.
    - Start directly with key information—do not include introductions or meta-references.
    - Use original terminology and maintain technical accuracy.
    - Ensure clarity and readability throughout.

    3. AVOID:
    - Meta-references (e.g., "this text discusses").
    - Personal interpretations or external knowledge.
    - Bullet points or lists.
    - Redundant or repetitive information.
    - Introductory or concluding phrases (e.g., "Here’s a concise, objective summary of the provided text").

    If the text is ambiguous or incomplete, summarize only what is clear and explicitly stated.   

    Text:
    ------------------------------------------------------------------------------------------
    <text_to_summarize>
    {chunk}
    </text_to_summarize>
    ------------------------------------------------------------------------------------------

    IMPORTANT:
    - Begin your response immediately with the summary content.
    - Use the same language as the original text.
    """

# Use the settings property for Ollama URL
OLLAMA_URL = get_settings().ollama_api_generate_url


def check_ollama_server_reachable(
    ollama_base_url: str = None, timeout: int = 5, verbose: bool = False
):
    """
    Check if the Ollama server is reachable before attempting any model operations.

    Args:
        ollama_base_url: Optional override of the base Ollama URL. If None, uses the value from settings.
        timeout: Timeout in seconds for the connection attempt
        verbose: Whether to log detailed messages about server connectivity

    Returns:
        bool: True if the server is reachable, False otherwise
    """
    settings = get_settings()
    base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")

    # Try to connect to the root endpoint - Ollama shows "Ollama is running" on root
    try:
        # First try the root endpoint
        response = requests.get(f"{base_url}", timeout=timeout)
        if response.status_code == 200:
            if verbose:
                logger.info(f"Ollama server at {base_url} is reachable")
            return True

        # If that fails, try the /api/tags endpoint which should exist
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if response.status_code == 200:
            if verbose:
                logger.info(f"Ollama server at {base_url} is reachable via /api/tags")
            return True

        # If both fail but server responds, log the error
        logger.error(
            f"Ollama server at {base_url} returned unexpected status code {response.status_code}"
        )
        return False
    except requests.exceptions.ConnectionError:
        logger.error(
            f"Could not connect to Ollama server at {base_url}. Is Ollama running?"
        )
        return False
    except requests.exceptions.Timeout:
        logger.error(f"Timeout connecting to Ollama server at {base_url}")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama server availability: {e}")
        return False


def check_ollama_model(model_name: str, ollama_base_url: str = None):
    """
    Checks if a specific Ollama model is available locally via the API.

    Args:
        model_name: The name of the model to check (e.g., "llama3:latest").
        ollama_base_url: The base URL of the Ollama API.

    Returns:
        True if the model is available locally, False otherwise.
    """
    settings = get_settings()

    # Use provided base URL or get from settings
    base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")
    api_url = f"{base_url}/api/tags"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        if not isinstance(models, list):
            logger.error(
                f"Unexpected format from Ollama API /api/tags. Expected a list under 'models'. Response: {data}"
            )
            return False

        # Check if any model name exactly matches our model_name
        for model in models:
            if isinstance(model, dict) and model.get("name") == model_name:
                return True

        return False

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Could not connect to Ollama API at {ollama_base_url or settings.ollama_base_url}. Is Ollama running?"
        )
        return False
    except requests.exceptions.Timeout:
        logger.error(
            f"Timeout connecting to Ollama API at {ollama_base_url or settings.ollama_base_url}"
        )
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking for Ollama model: {e}")
        return False


def ensure_ollama_model(model_name: str, fallback_model: str = None) -> str:
    """
    Ensures an Ollama model is available locally, attempting to pull it if not.
    First checks if the Ollama server is reachable before attempting any operations.
    If pulling fails and a fallback model is provided, it will verify the fallback is available.

    Args:
        model_name: The name of the model to ensure is available
        fallback_model: Optional fallback model to use if the requested model can't be pulled

    Returns:
        The name of the model that is available to use (either the requested or fallback model)
    """
    # First check if the Ollama server is reachable at all
    if not check_ollama_server_reachable(verbose=False):
        logger.warning(
            f"Ollama server is not reachable. Cannot ensure model '{model_name}' is available."
        )
        # Return the requested model name even though we can't verify it
        # This allows the application to start even if Ollama is not available
        return model_name
    # Check if model exists
    logger.info(f"Checking if Ollama model '{model_name}' is available locally...")
    if not check_ollama_model(model_name):
        logger.warning(f"Model '{model_name}' not found locally. Attempting to pull...")

        # Try to pull the model
        if pull_ollama_model(model_name, stream=False):
            logger.info(f"Successfully pulled model '{model_name}'")
            return model_name

        # If pull fails and we have a fallback model
        if fallback_model and fallback_model != model_name:
            logger.warning(
                f"Failed to pull model '{model_name}'. Trying fallback model '{fallback_model}'"
            )

            # Check if fallback model exists
            if not check_ollama_model(fallback_model):
                logger.warning(
                    f"Fallback model '{fallback_model}' not found locally. Attempting to pull..."
                )

                # Try to pull the fallback model
                if pull_ollama_model(fallback_model, stream=False):
                    logger.info(
                        f"Successfully pulled fallback model '{fallback_model}'"
                    )
                    return fallback_model
                else:
                    logger.error(
                        f"Failed to pull fallback model '{fallback_model}'. Processing may fail."
                    )
                    # Return the fallback model name anyway, as that's our best option
                    return fallback_model
            else:
                logger.info(f"Fallback model '{fallback_model}' is available locally")
                return fallback_model
        else:
            # No fallback provided or fallback is the same as requested model
            logger.error(
                f"Failed to pull model '{model_name}' and no valid fallback available. Processing may fail."
            )
            return model_name
    else:
        logger.info(f"Model '{model_name}' is available locally")
        return model_name


def pull_ollama_model(
    model_name: str,
    ollama_base_url: str = None,
    stream: bool = False,
):
    """
    Triggers Ollama to pull a model using the API.

    Args:
        model_name: The name of the model to pull (e.g., "llama3:latest").
        ollama_base_url: The base URL of the Ollama API.
        stream: Whether to process the response as a stream (True) or wait for completion (False).

    Returns:
        True if the pull request was successful, False otherwise.
    """
    settings = get_settings()

    # Use provided base URL or get from settings
    base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")
    api_url = f"{base_url}/api/pull"

    # Use the stream parameter as specified by the documentation
    # If stream=False in the API request, Ollama will wait until download completes and return single response
    payload = {"model": model_name, "stream": stream}
    logger.info(f"Pulling model '{model_name}' from Ollama...")

    try:
        # Always use stream=True for requests to allow processing response in chunks
        response = requests.post(api_url, json=payload, stream=True)
        response.raise_for_status()

        # If API stream=True, we'll get multiple status updates
        if stream:
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    status = json.loads(line.decode("utf-8"))
                    logger.info(f"Pull status: {status.get('status', 'unknown')}")

                    # Show progress percentage for downloads
                    if (
                        status.get("status", "").startswith("downloading")
                        and "total" in status
                        and "completed" in status
                        and status["total"] > 0
                    ):
                        progress = (status["completed"] / status["total"]) * 100
                        logger.info(f"Download progress: {progress:.1f}%")

                    if status.get("status") == "success":
                        return True
                except json.JSONDecodeError:
                    continue

            # If we got here without returning True, the stream ended without success
            logger.error("Model pull stream ended without success status")
            return False

        # If API stream=False, we'll get a single response at the end
        else:
            # Even with API stream=False, we still need to process the response
            last_status = None
            for line in response.iter_lines():
                if line:
                    try:
                        status = json.loads(line.decode("utf-8"))
                        last_status = status
                    except json.JSONDecodeError:
                        continue

            # Check final status
            if last_status and last_status.get("status") == "success":
                logger.info(f"Model '{model_name}' pulled successfully")
                return True
            else:
                logger.error(f"Failed to pull model. Final status: {last_status}")
                return False

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Could not connect to Ollama API at {ollama_base_url or settings.ollama_base_url}. Is Ollama running?"
        )
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error pulling Ollama model: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager that initializes models before startup
    and cleans up resources on shutdown.
    """
    # First check if the Ollama server is reachable at all
    # Check server with verbose logging (this is only done once at startup)
    server_reachable = check_ollama_server_reachable(verbose=False)

    if server_reachable:
        # Only try to ensure the model is available if the server is reachable
        logger.info(
            f"Ollama server at {get_settings().ollama_base_url} is reachable. Checking for required models..."
        )
        ensure_ollama_model(get_settings().llm_model)
    else:
        # Use critical level for more visibility in logs
        logger.critical("⚠️ WARNING: OLLAMA SERVER NOT AVAILABLE ⚠️")
        logger.critical(
            "The application is starting with LIMITED FUNCTIONALITY. "
            "LLM-dependent features (summarization and RAG) will NOT WORK "
            "until the Ollama server becomes available."
        )
        logger.warning(
            "Please ensure Ollama is running and accessible at: "
            + get_settings().ollama_base_url
        )

    # Continue with embedding model loading
    logger.info("Loading embedding model during application startup...")
    _get_model(get_settings().embedder_model)
    logger.info("Embedding model loaded.")

    # Store server status for post-startup message
    app.state.ollama_available = server_reachable

    yield

    # Comprehensive cleanup on shutdown
    logger.info("Application shutting down, cleaning up resources...")
    try:
        with _model_lock:
            # Clean up all models in the cache
            for model_name in list(_model_cache.keys()):
                if model_name in _model_cache:
                    logger.info(
                        f"Removing model {model_name} from cache during shutdown"
                    )
                    del _model_cache[model_name]

            # Clear the last used tracking dictionary
            _model_last_used.clear()
            logger.info("Model cache and tracking dictionaries cleared")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(
                f"GPU memory cleaned up. Current usage: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB"
            )
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {str(e)}")
        # Don't re-raise the exception to avoid blocking shutdown


# Initialize FastAPI app.
app = FastAPI(
    title="RAPTOR API",
    description="API for Recursive Abstraction and Processing for Text Organization and Reduction",
    version="0.5.0",
    lifespan=lifespan,
)

# Suppress warnings.
warnings.filterwarnings("ignore", category=NumbaWarning)
warnings.filterwarnings("ignore", message=".*force_all_finite.*")


# Create logs directory if it doesn't exist.
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the logger.
logger = logging.getLogger(__name__)

# Create a file handler for error logs.
error_log_path = logs_dir / "errors.log"
file_handler = RotatingFileHandler(
    error_log_path,
    maxBytes=10485760,  # 10 MB.
    backupCount=5,  # Keep 5 backup logs.
    encoding="utf-8",
)

# Set the file handler to only log errors and critical messages.
file_handler.setLevel(logging.ERROR)

# Create a formatter.
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d"
)
file_handler.setFormatter(formatter)

# Add the handler to the logger.
logger.addHandler(file_handler)

# Create a singleton for model caching.
_model_cache = {}
_model_last_used = {}  # Track when each model was last used
_model_lock = threading.RLock()  # Thread-safe lock for model cache

# Cache timeout in seconds (1 hour)
MODEL_CACHE_TIMEOUT = int(os.environ.get("MODEL_CACHE_TIMEOUT", 3600))

# Set random seed.
random.seed(get_settings().random_seed)


def _get_model(model_name: str) -> SentenceTransformer:
    """Get model from cache or load it into RAM.

    Args:
        model_name (str): Name or path of the model to use.

    Returns:
        SentenceTransformer: The loaded model instance.

    Raises:
        ValueError: If model_name is None or empty.
        RuntimeError: If there's an error loading the model from disk or downloading it.
        Exception: For any other unexpected errors during model loading.
    """
    if not model_name:
        error_msg = "Model name cannot be None or empty"
        logger.error(error_msg)
        raise ValueError(error_msg)

    current_time = time.time()
    logger.debug(f"Requesting model: {model_name}")

    # Track memory usage before any operations
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.debug(f"GPU memory before model operations: {before_mem:.2f} MB")

    with _model_lock:
        logger.debug(f"Acquired lock for model operations: {model_name}")
        # Check for expired models and remove them
        expired_count = 0
        for name, last_used in list(_model_last_used.items()):
            if current_time - last_used > MODEL_CACHE_TIMEOUT:
                if name in _model_cache:
                    unused_minutes = int((current_time - last_used) / 60)
                    logger.info(
                        f"Removing expired model {name} from cache (unused for {unused_minutes} minutes)"
                    )
                    try:
                        del _model_cache[name]
                        del _model_last_used[name]
                        expired_count += 1
                        # Clean up GPU memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            after_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                            logger.info(f"GPU memory after cleanup: {after_mem:.2f} MB")
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Error during expired model cleanup for {name}: {str(cleanup_error)}"
                        )

        if expired_count > 0:
            logger.info(f"Removed {expired_count} expired models from cache")

        # Load model if not in cache
        if model_name not in _model_cache:
            logger.info(f"Model {model_name} not found in cache, loading...")
            try:
                # Create models directory if it doesn't exist.
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)

                # Local path for the model.
                local_model_path = models_dir / model_name.replace("/", "_")

                if local_model_path.exists():
                    # Load from local storage.
                    logger.info(f"Loading model from local storage: {local_model_path}")
                    start_time = time.time()
                    _model_cache[model_name] = SentenceTransformer(
                        str(local_model_path)
                    )
                    load_time = time.time() - start_time
                    logger.info(
                        f"Model {model_name} loaded from disk in {load_time:.2f} seconds"
                    )
                else:
                    # Download and save model.
                    logger.info(
                        f"Downloading model {model_name} and saving to {local_model_path}"
                    )
                    start_time = time.time()
                    _model_cache[model_name] = SentenceTransformer(model_name)
                    download_time = time.time() - start_time

                    logger.info(
                        f"Model {model_name} downloaded in {download_time:.2f} seconds, saving to disk..."
                    )
                    save_start = time.time()
                    _model_cache[model_name].save(str(local_model_path))
                    save_time = time.time() - save_start
                    logger.info(
                        f"Model {model_name} saved to disk in {save_time:.2f} seconds"
                    )

                # Log model size information
                model_size = sum(
                    p.numel() * p.element_size()
                    for p in _model_cache[model_name].parameters()
                ) / (1024 * 1024)
                logger.info(
                    f"Model {model_name} loaded, approximate size: {model_size:.2f} MB"
                )

            except FileNotFoundError as e:
                error_msg = f"Model directory not accessible: {str(e)}"
                logger.error(error_msg)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(error_msg) from e
            except (OSError, IOError) as e:
                error_msg = f"I/O error loading model {model_name}: {str(e)}"
                logger.error(error_msg)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = f"Unexpected error loading model {model_name}: {str(e)}"
                logger.error(error_msg)
                # Try to clean up memory in case of error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("GPU memory cleaned up after model loading error")
                raise
        else:
            logger.debug(f"Model {model_name} found in cache")

        # Update last used timestamp
        _model_last_used[model_name] = current_time
        logger.debug(f"Updated last used timestamp for {model_name}")

        # Log current cache status
        logger.debug(f"Current model cache size: {len(_model_cache)} models")

        return _model_cache[model_name]


def generate_summary(
    chunk: str,
    model: str = None,
    prompt: str = PROMPT_TEMPLATE,
    temperature: float = None,
    context_window: int = None,
) -> str:
    """Generate a summary using OLLAMA.

    Args:
        chunk (str): The text chunk to summarize.
        model (str, optional): The LLM model identifier to use. Defaults to value from settings.
        prompt (str, optional): The prompt template for summarization. Defaults to PROMPT_TEMPLATE.
        temperature (float, optional): Controls randomness in output (0.0 to 1.0). Defaults to TEMPERATURE.
        context_window (int, optional): Maximum context window size. Defaults to CONTEXT_WINDOW.

    Returns:
        The generated summary content.
    """

    # Get settings
    settings = get_settings()

    # Use provided values or defaults from settings
    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.temperature
    context_window = (
        context_window if context_window is not None else settings.context_window
    )

    # Get the Ollama URL from environment settings
    api_url = settings.ollama_api_generate_url

    # Format the prompt with the actual chunk content.
    formatted_prompt = prompt.format(chunk=chunk)

    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": formatted_prompt,
        "temperature": temperature,
        "options": {"num_ctx": context_window},
    }

    # Retry configuration.
    max_retries = 3
    base_delay = 2  # Seconds.
    timeout = 360  # Seconds.
    last_exception = None

    # Implement retry with exponential backoff.
    for attempt in range(max_retries):
        try:
            # Add timeout to prevent hanging indefinitely.
            response = requests.post(
                api_url, headers=headers, data=json.dumps(data), timeout=timeout
            )

            if response.status_code == 200:
                # Process streaming response.
                full_response = ""
                for line in response.text.strip().split("\n"):
                    try:
                        resp_json = json.loads(line)
                        if "response" in resp_json:
                            full_response += resp_json["response"]
                    except json.JSONDecodeError:
                        # Continue processing other lines.
                        continue

                if not full_response.strip():
                    # If we got an empty response, log and retry.
                    logger.warning(
                        f"Empty response received from OLLAMA (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt == max_retries - 1:
                        return "Unable to generate summary due to empty response from OLLAMA."
                else:
                    return full_response
            else:
                # Log the error and prepare for retry.
                logger.warning(
                    f"Error response from OLLAMA (attempt {attempt + 1}/{max_retries}): "
                    f"Status {response.status_code}, Response: {response.text[:200]}..."
                )
                if attempt == max_retries - 1:
                    # On last attempt, raise the exception.
                    raise Exception(f"Error generating summary: {response.status_code}")

        except requests.exceptions.Timeout as e:
            logger.warning(
                f"Timeout error connecting to OLLAMA (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            last_exception = e

        except requests.exceptions.ConnectionError as e:
            logger.warning(
                f"Connection error to OLLAMA (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            last_exception = e

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Request error to OLLAMA (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            last_exception = e

        except Exception as e:
            logger.warning(
                f"Unexpected error during OLLAMA request (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            last_exception = e

        # Only sleep if we're going to retry.
        if attempt < max_retries - 1:
            # Exponential backoff with jitter.
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            logger.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

    # If we've exhausted all retries, log the error and raise an exception.
    logger.error(f"Failed to connect to OLLAMA after {max_retries} attempts")
    if last_exception:
        raise Exception(
            f"Failed to connect to OLLAMA after {max_retries} attempts: {str(last_exception)}"
        )
    else:
        raise Exception(f"Failed to connect to OLLAMA after {max_retries} attempts")


def get_embeddings(
    chunks: List[str],
    model: str = None,
    batch_size: int = 4,
    show_progress_bar: bool = True,
    convert_to_numpy: bool = True,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Generate embeddings for text chunks using a Sentence Transformer model.

    Args:
        chunks: List of text chunks to generate embeddings for.
        model (str, optional): Name or path of the model to use.
            Defaults to "BAAI/bge-m3".
        batch_size (int, optional): Batch size for embedding generation.
            Defaults to 4.
        show_progress_bar (bool, optional): Whether to show progress bar.
            Defaults to True.
        convert_to_numpy (bool, optional): Whether to convert output to numpy array.
            Defaults to True.
        normalize_embeddings (bool, optional): Whether to normalize embeddings.
            Defaults to True.

    Returns:
        A numpy array of embeddings.

    Raises:
        HTTPException: If there's an error during the embedding process.
    """
    # Use settings model if none provided
    if model is None:
        model = get_settings().embedder_model

    # Adjust batch size dynamically based on document size
    adjusted_batch_size = batch_size
    if len(chunks) > 1000:
        adjusted_batch_size = max(1, batch_size // 2)
        logger.info(
            f"Large document detected ({len(chunks)} chunks), reducing batch size to {adjusted_batch_size}"
        )

    # Log memory usage before processing if GPU is available
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.info(f"GPU memory usage before embedding: {before_mem:.2f} MB")

    model_instance = None
    try:
        # Get model from cache (loads from disk if not in RAM).
        model_instance = _get_model(model)

        # Move to GPU if available.
        if torch.cuda.is_available():
            model_instance = model_instance.to("cuda")

        # Get embeddings with adjusted batch size.
        embeddings = model_instance.encode(
            chunks,
            batch_size=adjusted_batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )

        return embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}",
        )
    finally:
        # Always clean up GPU memory in finally block to ensure it runs
        if torch.cuda.is_available() and model_instance is not None:
            try:
                model_instance.cpu()
                torch.cuda.empty_cache()
                gc.collect()

                # Log memory usage after cleanup
                after_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                logger.info(f"GPU memory usage after cleanup: {after_mem:.2f} MB")
                logger.info("GPU memory cleared after embeddings processing")
            except Exception as cleanup_error:
                logger.warning(f"Error during GPU memory cleanup: {str(cleanup_error)}")


def _global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
):
    """Perform global dimensionality reduction using UMAP.

    Args:
        embeddings: Input embeddings to reduce.
        dim: Target dimension.
        n_neighbors: Number of neighbors to consider.
        metric: Distance metric to use.

    Returns:
        Reduced dimension embeddings.
    """
    # For small datasets, adjust parameters
    n_samples = len(embeddings)
    if n_neighbors is None:
        n_neighbors = min(15, max(2, n_samples - 1))

    # Adjust target dimension for small datasets
    target_dim = min(dim, max(2, n_samples - 2))

    reducer = umap.UMAP(
        n_components=target_dim,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=None,  # Allow parallelization
        n_jobs=get_settings().max_workers,
        min_dist=0.1,  # Increase min_dist for better separation
        spread=1.0,  # Default spread
    )
    return reducer.fit_transform(embeddings)


def _split_into_sentences(doc: str) -> List[str]:
    """Split a document into sentences using regex pattern matching.

    Args:
        doc (str): The input document text to be split into sentences.

    Returns:
        List[str]: A list of sentences, with each sentence stripped of leading/trailing whitespace.

    Note:
        The function handles common edge cases like:
        - Titles (Mr., Mrs., Dr., etc.)
        - Common abbreviations (i.e., e.g., etc.)
        - Decimal numbers
        - Ellipsis
        - Quotes and brackets
    """
    # Define a pattern that looks for sentence boundaries but doesn't include them in the split
    # Instead of splitting directly at punctuation, we'll look for patterns that indicate sentence endings
    pattern = r"""
        # Match sentence ending punctuation followed by space and capital letter
        # Negative lookbehind for common titles and abbreviations
        (?<![A-Z][a-z]\.)                                                  # Not an abbreviation like U.S.
        (?<!Mr\.)(?<!Mrs\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ms\.) # Not a title
        (?<!i\.e\.)(?<!e\.g\.)(?<!vs\.)(?<!etc\.)                          # Not a common abbreviation
        (?<!\d\.)(?<!\.\d)                                                 # Not a decimal or numbered list
        (?<!\.\.\.)                                                        # Not an ellipsis
        [\.!\?]                                                            # Sentence ending punctuation
        \s+                                                                # One or more whitespace
        (?=[A-Z])                                                          # Followed by capital letter
    """

    # Find all positions where we should split
    split_positions = []
    for match in re.finditer(pattern, doc, re.VERBOSE):
        # Split after the punctuation and space
        split_positions.append(match.end())

    # Use the positions to extract sentences
    sentences = []
    start = 0
    for pos in split_positions:
        if pos > start:
            sentences.append(doc[start:pos].strip())
            start = pos

    # Add the last sentence if there's remaining text
    if start < len(doc):
        sentences.append(doc[start:].strip())

    # Filter out empty sentences
    return [s for s in sentences if s]


def concatenate_strings(list_of_lists: List[List[str]]) -> List[str]:
    """Concatenate each inner list of strings into a single string.

    Args:
        list_of_lists: A list containing lists of strings to concatenate.

    Returns:
        A list of concatenated strings.
    """
    return ["".join(inner_list) for inner_list in list_of_lists]


def _count_tokens_for_text(data: tuple[str, str]) -> int:
    """Count tokens in a text string using the specified encoding.

    Args:
        data (tuple): Tuple containing:
            - text (str): Text to count tokens for
            - encoding_name (str): Name of the tiktoken encoding to use

    Returns:
        int: Number of tokens in the text

    Note:
        Provides fallback estimation if the specified encoding cannot be loaded.
    """
    text, encoding_name = data
    try:
        # Attempt to get encoding and count tokens
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except tiktoken.registry.RegistryError as e:
        # Handle case where encoding is not found
        logger.error(f"Encoding '{encoding_name}' not found: {e}. Falling back.")
    except Exception as e:
        # Handle other potential errors during encoding/counting
        logger.error(
            f"Error counting tokens with tiktoken for text '{text[:50]}...': {e}. "
            f"Falling back to estimated count."
        )

    # Fallback: Estimate tokens per character (crude fallback)
    # Average ~3-4 chars per token is a common heuristic
    estimated_tokens = int(len(text) * 0.3)
    logger.warning(f"Using estimated token count: {estimated_tokens}")
    return estimated_tokens


def parallel_count_tokens(
    chunks: List[str], encoding_name: str = "cl100k_base"
) -> List[int]:
    """Count tokens for multiple text chunks in parallel using ThreadPoolExecutor.

    This function efficiently processes multiple chunks of text by distributing
    token counting tasks across multiple threads. It's optimized for batch processing
    of text chunks where parallel execution provides performance benefits.

    Args:
        chunks: List of text strings to count tokens for
        encoding_name: Name of the tiktoken encoding to use (default: "cl100k_base")

    Returns:
        List of token counts corresponding to each input chunk

    Note:
        Uses ThreadPoolExecutor rather than ProcessPoolExecutor since token counting
        with tiktoken is relatively lightweight and the underlying Rust implementation
        may release the GIL.
    """
    if not chunks:
        return []

    # Use ThreadPoolExecutor for parallel token counting
    max_workers = get_settings().max_workers
    logger.info(f"Counting tokens for {len(chunks)} chunks using {max_workers} threads")

    with ThreadPoolExecutor(max_workers=get_settings().max_workers) as executor:
        tasks = [(chunk, encoding_name) for chunk in chunks]
        results = list(executor.map(_count_tokens_for_text, tasks))

    return results


def _split_large_sentence(
    sentence: str, max_tokens: int
) -> List[Dict[str, Union[str, int]]]:
    """Split an oversized sentence at token boundaries to fit within token limits.

    Args:
        sentence: The sentence text to split
        max_tokens: Maximum number of tokens per chunk

    Returns:
        List of dictionaries containing text and token_count
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.error(f"Could not get tiktoken encoding 'cl100k_base': {e}")
        # If encoding fails, cannot split by token. Return original sentence as one chunk.
        # Or raise error? Let's return a single chunk with estimated tokens as fallback.
        estimated_tokens = int(len(sentence) * 0.3)
        logger.warning(
            f"Returning oversized sentence as single chunk with estimated tokens: {estimated_tokens}"
        )
        return [{"text": sentence, "token_count": estimated_tokens}]

    all_tokens = encoding.encode(sentence)

    chunks = []
    current_pos = 0
    while current_pos < len(all_tokens):
        end_pos = min(current_pos + max_tokens, len(all_tokens))
        chunk_tokens = all_tokens[current_pos:end_pos]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({"text": chunk_text, "token_count": len(chunk_tokens)})
        current_pos = end_pos  # Move to the next position

    return chunks


def _split_oversized_chunk(
    chunk_text: str, max_tokens: int
) -> List[Dict[str, Union[str, int]]]:
    """Split an oversized chunk using a sentence-based approach with semantic prioritization.
       Adapted from normalized_semantic_chunker.py.

    Args:
        chunk_text: The text to be split into chunks
        max_tokens: Maximum number of tokens allowed per chunk

    Returns:
        List of dictionaries containing 'text' and 'token_count'
    """
    # Split text into sentences using the function defined in this file
    sentences = _split_into_sentences(chunk_text)  # Uses the renamed function

    # If sentence splitting fails or returns nothing, fall back to basic splitting
    if not sentences:
        logger.warning(
            "Sentence splitting yielded no results, falling back to basic splitting by tokens."
        )
        # Use _split_large_sentence to break the whole text by max_tokens
        return _split_large_sentence(chunk_text, max_tokens)

    # Calculate token count for each sentence using the helper in this file
    sentence_tokens = [
        (s, _count_tokens_for_text((s, "cl100k_base"))) for s in sentences
    ]

    # Create chunks respecting sentence boundaries
    chunks = []
    current_chunk_sentences = []
    current_tokens = 0

    # Optimal target is 70-80% of max_tokens for better balance
    target_size = int(max_tokens * 0.75)

    for sentence, tokens in sentence_tokens:
        # If the sentence is already too large (rare), split at token level
        if tokens > max_tokens:
            # If a chunk was being built, finalize it first
            if current_chunk_sentences:
                chunks.append(
                    {
                        "text": " ".join(current_chunk_sentences),
                        "token_count": current_tokens,
                    }
                )
                current_chunk_sentences = []
                current_tokens = 0
            # Split the large sentence and add its parts as separate chunks
            chunks.extend(_split_large_sentence(sentence, max_tokens))
            continue  # Move to the next sentence

        # If adding this sentence would exceed max_tokens, finalize current chunk
        # Check current_chunk_sentences to ensure we don't create empty chunks if the first sentence is too long
        if current_chunk_sentences and (current_tokens + tokens > max_tokens):
            chunks.append(
                {
                    "text": " ".join(current_chunk_sentences),
                    "token_count": current_tokens,
                }
            )
            current_chunk_sentences = []
            current_tokens = 0

        # If we've reached optimal size and are at a "natural" sentence boundary
        # Also ensure we don't create empty chunks
        elif (
            current_tokens >= target_size
            and sentence[-1] in ".!?"  # Check if sentence ends with punctuation
            and current_chunk_sentences  # Ensure current chunk is not empty
        ):
            chunks.append(
                {
                    "text": " ".join(current_chunk_sentences),
                    "token_count": current_tokens,
                }
            )
            current_chunk_sentences = []
            current_tokens = 0

        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_tokens += tokens

    # Add final chunk if there's anything remaining
    if current_chunk_sentences:
        chunks.append(
            {"text": " ".join(current_chunk_sentences), "token_count": current_tokens}
        )

    # Post-processing: Ensure no chunk accidentally exceeds max_tokens due to edge cases/joining spaces
    final_chunks = []
    for chunk in chunks:
        if chunk["token_count"] > max_tokens:
            logger.warning(
                f"Chunk exceeded max_tokens ({max_tokens}) after joining sentences. Re-splitting."
            )
            # Re-split the problematic chunk purely by token limit
            final_chunks.extend(_split_large_sentence(chunk["text"], max_tokens))
        else:
            final_chunks.append(chunk)

    return final_chunks


def apply_chunk_optimization(
    summaries: List[str],
    token_counts: List[int],
    threshold_tokens: Optional[int],
    max_chunk_tokens: Optional[int] = None,
    level: str = "",
) -> List["SummaryWithTokens"]:
    """
    Apply chunk optimization to ensure no summary exceeds the token threshold.
    If threshold_tokens is None, no optimization is applied.

    Args:
        summaries: List of summary texts
        token_counts: List of token counts for each summary
        threshold_tokens: Token threshold that triggers splitting, None to skip optimization
        max_chunk_tokens: Maximum tokens allowed per chunk, defaults to 75% of threshold if not specified
        level: The RAPTOR level name for logging (e.g., "Level 1", "Level 2", etc.)

    Returns:
        List of optimized SummaryWithTokens objects
    """
    # If threshold_tokens is None, skip optimization
    if threshold_tokens is None:
        if level:
            logger.info(f"{level} - Chunk optimization skipped (no threshold set)")
        return [
            SummaryWithTokens(summary=summary, token_count=token_count)
            for summary, token_count in zip(summaries, token_counts)
        ]

    # If max_chunk_tokens not set, default to 75% of threshold
    if max_chunk_tokens is None:
        max_chunk_tokens = int(threshold_tokens * 0.75)

    optimized_summaries = []

    # Count chunks over threshold
    chunks_over_threshold = sum(
        1 for token_count in token_counts if token_count > threshold_tokens
    )
    total_original_chunks = len(summaries)

    # Log the initial state
    if level:
        logger.info(
            f"{level} - Found {chunks_over_threshold} of {total_original_chunks} chunks over threshold of {threshold_tokens} tokens"
        )

    # Track how many new chunks were created by splitting
    new_chunks_created = 0

    for summary, token_count in zip(summaries, token_counts):
        # Apply smart text splitting if needed
        if token_count > threshold_tokens:
            # Pass summary text and max_chunk_tokens
            split_chunks = _split_oversized_chunk(summary, max_chunk_tokens)
            # Convert each chunk to SummaryWithTokens and add to result
            for chunk in split_chunks:
                optimized_summaries.append(
                    SummaryWithTokens(
                        summary=chunk["text"], token_count=chunk["token_count"]
                    )
                )
            # Track new chunks (subtract 1 because we're replacing 1 chunk with multiple)
            new_chunks_created += len(split_chunks) - 1
        else:
            # No splitting needed, add as is
            optimized_summaries.append(
                SummaryWithTokens(summary=summary, token_count=token_count)
            )

    # Log the final state
    if level:
        logger.info(
            f"{level} - After optimization - {total_original_chunks + new_chunks_created} chunks total ({new_chunks_created} new chunks created by splitting)"
        )

    return optimized_summaries


def _local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
):
    """Perform local dimensionality reduction using UMAP.

    Args:
        embeddings: Input embeddings to reduce.
        dim: Target dimension.
        num_neighbors: Number of neighbors to consider.
        metric: Distance metric to use.

    Returns:
        Reduced dimension embeddings.
    """
    # For small datasets, adjust parameters
    n_samples = len(embeddings)
    num_neighbors = min(num_neighbors, max(2, n_samples - 1))

    # Adjust target dimension for small datasets
    target_dim = min(dim, max(2, n_samples - 2))

    reducer = umap.UMAP(
        n_components=target_dim,
        n_neighbors=num_neighbors,
        metric=metric,
        random_state=None,  # Allow parallelization
        n_jobs=get_settings().max_workers,
        min_dist=0.1,  # Increase min_dist for better separation
        spread=1.0,  # Default spread
    )
    return reducer.fit_transform(embeddings)


def _get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = None,
    is_level_1: bool = True,
) -> int:
    """Determine the optimal number of clusters using BIC criterion with smart scaling.

    Args:
        embeddings: Input embeddings to cluster.
        max_clusters: Maximum number of clusters to consider.
        random_state: Random seed for reproducibility.
        is_level_1: Whether this is Level 1 (global) clustering.

    Returns:
        The optimal number of clusters.
    """
    n_samples = len(embeddings)

    if n_samples <= 3:
        return 1

    # Smart scaling for different input sizes
    if is_level_1:
        # Level 1 (Global) clustering
        if n_samples < 50:
            max_ratio = 0.15  # Small datasets: up to 15% of input size
            min_clusters = 2
            max_absolute = 7
        elif n_samples < 200:
            max_ratio = 0.10  # Medium datasets: up to 10% of input size
            min_clusters = 3
            max_absolute = 12
        else:
            max_ratio = 0.08  # Large datasets: up to 8% of input size
            min_clusters = 5
            max_absolute = 15
    else:
        # Level 2 (Local) clustering
        if n_samples < 20:
            max_ratio = 0.30  # Small clusters: up to 30% of input size
            min_clusters = 2
            max_absolute = 5
        elif n_samples < 50:
            max_ratio = 0.25  # Medium clusters: up to 25% of input size
            min_clusters = 2
            max_absolute = 8
        else:
            max_ratio = 0.20  # Large clusters: up to 20% of input size
            min_clusters = 3
            max_absolute = 10

    # Calculate suggested number of clusters
    suggested_max = max(min_clusters, min(int(n_samples * max_ratio), max_absolute))

    # Ensure minimum cluster size (average of 4 documents per cluster)
    min_cluster_size = 4
    max_by_size = max(2, n_samples // min_cluster_size)

    # Final max clusters determination
    max_clusters = min(suggested_max, max_by_size, max_clusters, n_samples - 1)

    # If we only have a few possible clusters, try them all
    if max_clusters <= 2:
        return max_clusters

    # Calculate BIC scores
    n_clusters = np.arange(min_clusters, max_clusters + 1)
    bics = []

    for n in n_clusters:
        gm = GaussianMixture(
            n_components=n,
            random_state=random_state,
            max_iter=100,
            n_init=1,
        )
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))

    return n_clusters[np.argmin(bics)]


def _gmm_cluster(
    embeddings: np.ndarray,
    threshold: float,
    n_components: Optional[int] = None,
    random_state: int = 0,
) -> Tuple[np.ndarray, int]:
    """Cluster embeddings using Gaussian Mixture Model.

    Args:
        embeddings: Input embeddings to cluster.
        threshold: Probability threshold for cluster assignment.
        n_components: Number of components (clusters) to use. If None, will be determined automatically.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple containing cluster labels and number of clusters.
    """
    if n_components is None:
        n_components = _get_optimal_clusters(embeddings, is_level_1=True)

    gm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        max_iter=100,
        n_init=1,
    )

    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)

    # Assign points to clusters only if probability exceeds threshold
    labels = np.zeros(len(embeddings), dtype=int)
    for i in range(len(embeddings)):
        max_prob = np.max(probs[i])
        if max_prob >= threshold:
            labels[i] = np.argmax(probs[i]) + 1  # 1-based indexing

    # Count unique non-zero labels to get number of clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels > 0])

    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """Perform hierarchical clustering on embeddings.

    Args:
        embeddings: Input embeddings to cluster.
        dim: Target dimension for reduction.
        threshold: Probability threshold for cluster assignment.

    Returns:
        List of cluster assignments for each embedding.
    """
    n_samples = len(embeddings)

    # If dataset is too small, return single cluster
    if n_samples <= 3:
        return [np.array([1]) for _ in range(n_samples)]

    # Adjust dimension for small datasets
    target_dim = min(dim, max(2, n_samples - 2))

    # Level 1: Global clustering
    reduced_embeddings_global = _global_cluster_embeddings(embeddings, target_dim)
    optimal_global_clusters = _get_optimal_clusters(
        reduced_embeddings_global, max_clusters=50, is_level_1=True
    )
    global_clusters, n_global_clusters = _gmm_cluster(
        reduced_embeddings_global, threshold, n_components=optimal_global_clusters
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Process each global cluster
    for i in range(1, n_global_clusters + 1):
        cluster_mask = global_clusters == i
        global_cluster_embeddings_ = embeddings[cluster_mask]

        # Skip if cluster is too small for meaningful local clustering
        if len(global_cluster_embeddings_) <= 3:
            cluster_indices = np.where(cluster_mask)[0]
            for idx in cluster_indices:
                all_local_clusters[idx] = np.array([total_clusters + 1])
            total_clusters += 1
            continue

        # Level 2: Local clustering
        reduced_embeddings_local = _local_cluster_embeddings(
            global_cluster_embeddings_, target_dim
        )
        optimal_local_clusters = _get_optimal_clusters(
            reduced_embeddings_local, max_clusters=30, is_level_1=False
        )
        local_clusters, n_local_clusters = _gmm_cluster(
            reduced_embeddings_local, threshold, n_components=optimal_local_clusters
        )

        # Update cluster assignments
        cluster_indices = np.where(cluster_mask)[0]
        for j, idx in enumerate(cluster_indices):
            if local_clusters[j] > 0:
                all_local_clusters[idx] = np.array([total_clusters + local_clusters[j]])

        total_clusters += n_local_clusters

    return all_local_clusters


class ProcessingTimeMetadata(BaseModel):
    total: float
    level_1: float
    level_2: float
    level_3: float


class ClusteringMetadata(BaseModel):
    input_chunks: int
    level_1_clusters: int
    level_2_clusters: int
    level_3_clusters: int
    total_clusters: int
    reduction_ratio: float
    llm_model: str
    embedder_model: str
    temperature: float
    context_window: int
    custom_prompt_used: bool = False
    processing_time: ProcessingTimeMetadata


class SummaryWithTokens(BaseModel):
    summary: str
    token_count: int


class ChunkOutput(BaseModel):
    text: str
    token_count: int
    cluster_level: int
    id: int


class ClusteringResult(BaseModel):
    chunks: List[ChunkOutput]
    metadata: ClusteringMetadata


@app.get("/")
async def health_check():
    """Check the health status of the API service.

    Returns:
        dict: A dictionary containing:
            - status: Current health status of the service (healthy or degraded)
            - gpu_available: Boolean indicating if GPU is available
            - version: Current API version
            - ollama_status: Status of the Ollama server connectivity
            - ollama_url: URL of the Ollama server
    """
    # Check current Ollama connectivity
    ollama_available = check_ollama_server_reachable()

    # Determine overall status - we're "degraded" if Ollama isn't available
    status = "healthy" if ollama_available else "degraded"

    return {
        "status": status,
        "gpu_available": torch.cuda.is_available(),
        "version": app.version,
        "ollama_status": "connected" if ollama_available else "unavailable",
        "ollama_url": get_settings().ollama_base_url,
    }


@app.post("/raptor/", response_class=JSONResponse)
async def raptor(
    file: UploadFile = File(...),
    llm_model: str = Query(
        None, description="LLM model to use", example=get_settings().llm_model
    ),
    embedder_model: str = Query(
        None,
        description="Embedding model to use",
        example=get_settings().embedder_model,
    ),
    threshold_tokens: Optional[int] = Query(
        None, description="Token threshold for chunk optimization"
    ),
    temperature: float = Query(
        None,
        description="Temperature for text generation",
        example=get_settings().temperature,
    ),
    context_window: int = Query(
        None, description="Context window size", example=get_settings().context_window
    ),
    custom_prompt: Optional[str] = Query(
        None, description="Custom prompt template for summarization", example=""
    ),
):
    """Process semantic chunks from an uploaded JSON file for hierarchical clustering.

    Args:
        file (UploadFile): JSON file (.json) containing chunks to process with a 'chunks' array
        llm_model (str): LLM model to use for summarization
        embedder_model (str): Model to use for generating embeddings
        threshold_tokens (Optional[int]): Maximum token limit for summaries
        temperature (float): Controls randomness in LLM output (0.0 to 1.0)
        context_window (int): Maximum context window size for LLM
        custom_prompt (Optional[str]): Optional custom prompt template as a string

    Returns:
        JSONResponse: Hierarchical clustering results with metadata

    Raises:
        HTTPException: If file format is invalid or processing fails
    """
    start_time = time.time()
    try:
        # First check if Ollama server is reachable at all (without verbose logging)
        if not check_ollama_server_reachable(verbose=False):
            raise HTTPException(
                status_code=503,
                detail="Ollama server is not reachable. Please ensure Ollama is running and accessible.",
            )

        # Verify model availability before processing
        # This handles the case where models might be deleted from Ollama after the app has started
        logger.info(f"Verifying availability of LLM model: '{llm_model}'")
        settings = get_settings()
        # Use the provided model or get from settings
        llm_model = llm_model or settings.llm_model
        llm_model = ensure_ollama_model(llm_model, fallback_model=settings.llm_model)

        # Use the custom prompt if provided as a string parameter, otherwise use the default
        # This approach avoids issues with file uploads
        current_prompt = custom_prompt if custom_prompt else PROMPT_TEMPLATE

        logger.info(
            f"Processing - Chunks with LLM model: {llm_model}, Embedder model: {embedder_model}, "
            f"Threshold tokens: {threshold_tokens}, Temperature: {temperature}, "
            f"Context window: {context_window}, Custom prompt: {True if custom_prompt else False}"
        )

        # Validate file extension - only accept JSON files
        filename = file.filename.lower()
        if not filename.endswith(".json"):
            error_msg = (
                f"Invalid file format: {filename}. Only JSON files are accepted."
            )
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        content = await file.read()
        try:
            data = json.loads(content)
            if "chunks" not in data or not isinstance(data["chunks"], list):
                error_msg = (
                    "Invalid JSON format: The file must contain a 'chunks' array."
                )
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            chunks = [chunk["text"] for chunk in data["chunks"]]
            logger.info(
                f"Processing - {len(chunks)} chunks ready for RAPTOR processing"
            )
        except json.JSONDecodeError:
            error_msg = "Invalid JSON: The file could not be parsed as JSON."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except KeyError as e:
            error_msg = f"Invalid JSON structure: Missing required field {str(e)}."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # RAPTOR clustering level 1
        level1_start = time.time()
        logger.info("Level 1 - Starting RAPTOR clustering")
        chunks_embedded = get_embeddings(chunks, model=embedder_model)
        clusters = perform_clustering(chunks_embedded, dim=10, threshold=0.2)
        clusters_list = [int(arr.item()) for arr in clusters]
        df = pd.DataFrame({"Chunk": chunks, "Cluster": clusters_list})
        logger.info(f"Level 1 - Created {len(set(clusters_list))} clusters")

        level_nodes = []
        for cluster in sorted(set(clusters_list)):
            cluster_chunks = df[df["Cluster"] == cluster]["Chunk"].tolist()
            level_nodes.append(cluster_chunks)

        nodes = concatenate_strings(level_nodes)
        summaries_1 = []

        # Generate summaries for level 1
        for chunk in tqdm(nodes, desc="Level 1 LLM Summarization", unit="chunk"):
            summary = generate_summary(
                chunk,
                model=llm_model,
                prompt=current_prompt,
                temperature=temperature,
                context_window=context_window,
            )
            summaries_1.append(summary)

        # Count tokens in parallel for level 1
        token_counts_1 = parallel_count_tokens(summaries_1)

        # Apply chunk optimization to level 1 summaries
        clustering_1 = apply_chunk_optimization(
            summaries_1,
            token_counts_1,
            threshold_tokens,
            level="Level 1",
        )

        level1_time = time.time() - level1_start
        logger.info(
            f"Level 1 - Generated {len(clustering_1)} summaries in {level1_time:.2f} seconds"
        )

        # RAPTOR clustering level 2
        level2_start = time.time()
        logger.info("Level 2 - Starting RAPTOR clustering")
        chunks_embedded = get_embeddings(
            [s.summary for s in clustering_1], model=embedder_model
        )
        clusters = perform_clustering(chunks_embedded, dim=10, threshold=0.2)
        clusters_list = [int(arr.item()) for arr in clusters]
        df = pd.DataFrame(
            {"Chunk": [s.summary for s in clustering_1], "Cluster": clusters_list}
        )
        logger.info(f"Level 2 - Created {len(set(clusters_list))} clusters")

        level_nodes = []
        for cluster in sorted(set(clusters_list)):
            cluster_chunks = df[df["Cluster"] == cluster]["Chunk"].tolist()
            level_nodes.append(cluster_chunks)

        nodes = concatenate_strings(level_nodes)
        summaries_2 = []

        # Generate summaries for level 2
        for chunk in tqdm(nodes, desc="Level 2 LLM Summarization", unit="chunk"):
            summary = generate_summary(
                chunk,
                model=llm_model,
                prompt=current_prompt,
                temperature=temperature,
                context_window=context_window,
            )
            summaries_2.append(summary)

        # Count tokens in parallel for level 2
        token_counts_2 = parallel_count_tokens(summaries_2)

        # Apply chunk optimization to level 2 summaries
        clustering_2 = apply_chunk_optimization(
            summaries_2,
            token_counts_2,
            threshold_tokens,
            level="Level 2",
        )

        level2_time = time.time() - level2_start
        logger.info(
            f"Level 2 - Generated {len(clustering_2)} summaries in {level2_time:.2f} seconds"
        )

        # RAPTOR clustering level 3
        level3_start = time.time()
        logger.info("Level 3 - Starting RAPTOR clustering")
        node = " ".join([s.summary for s in clustering_2])

        # Use tqdm for the final summary generation
        with tqdm(total=1, desc="Level 3 LLM Summarization", unit="chunk") as pbar:
            final_summary = generate_summary(
                node,
                model=llm_model,
                prompt=current_prompt,
                temperature=temperature,
                context_window=context_window,
            )
            pbar.update(1)

        # Count tokens for final summary
        token_count_3 = parallel_count_tokens([final_summary])[0]

        # Apply chunk optimization to level 3 summary using the common function
        clustering_3 = apply_chunk_optimization(
            summaries=[final_summary],  # Pass as a single-element list
            token_counts=[token_count_3],  # Pass as a single-element list
            threshold_tokens=threshold_tokens,
            level="Level 3 ",  # Set level for appropriate logging
        )

        level3_time = time.time() - level3_start
        logger.info(f"Level 3 - Generated final summary in {level3_time:.2f} seconds")

        # Calculate total processing time
        total_time = time.time() - start_time

        # Create chunks output with the new format
        all_chunks = []
        chunk_id = 1

        # Add level 1 chunks
        for summary in clustering_1:
            all_chunks.append(
                ChunkOutput(
                    text=summary.summary,
                    token_count=summary.token_count,
                    cluster_level=1,
                    id=chunk_id,
                )
            )
            chunk_id += 1

        # Add level 2 chunks
        for summary in clustering_2:
            all_chunks.append(
                ChunkOutput(
                    text=summary.summary,
                    token_count=summary.token_count,
                    cluster_level=2,
                    id=chunk_id,
                )
            )
            chunk_id += 1

        # Add level 3 chunks
        for summary in clustering_3:
            all_chunks.append(
                ChunkOutput(
                    text=summary.summary,
                    token_count=summary.token_count,
                    cluster_level=3,
                    id=chunk_id,
                )
            )
            chunk_id += 1

        # Create clustering results with the new format
        clustering_results = ClusteringResult(
            chunks=all_chunks,
            metadata=ClusteringMetadata(
                input_chunks=len(chunks),
                level_1_clusters=len(clustering_1),
                level_2_clusters=len(clustering_2),
                level_3_clusters=len(clustering_3),
                total_clusters=len(clustering_1)
                + len(clustering_2)
                + len(clustering_3),
                reduction_ratio=round(
                    1
                    - (len(clustering_1) + len(clustering_2) + len(clustering_3))
                    / len(chunks),
                    2,
                ),
                llm_model=llm_model,
                embedder_model=embedder_model,
                temperature=temperature,
                context_window=context_window,
                # Add a flag indicating if a custom prompt was used
                custom_prompt_used=custom_prompt is not None,
                processing_time=ProcessingTimeMetadata(
                    total=round(total_time, 2),
                    level_1=round(level1_time, 2),
                    level_2=round(level2_time, 2),
                    level_3=round(level3_time, 2),
                ),
            ),
        )
        logger.info(
            f"Summary - Successfully completed RAPTOR clustering process in {total_time:.2f} seconds"
        )
        return JSONResponse(content=clustering_results.dict())

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            f"Error processing chunks after {total_time:.2f} seconds: {str(e)}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing chunks: {str(e)}"},
        )
