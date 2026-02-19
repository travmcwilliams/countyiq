"""
Embedding generation for RAG pipeline.
Uses Azure OpenAI text-embedding-ada-002 with fallback to sentence-transformers.
# DP-100: Feature representation - Embeddings as dense vector representations for semantic search.
"""

import os
import time
from functools import lru_cache
from typing import Any

from loguru import logger

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# DP-100: Model configuration - Embedding model selection and fallback strategy
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# Fallback model
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 1536  # text-embedding-ada-002 dimension


class Embedder:
    """
    Generates embeddings for text using Azure OpenAI or sentence-transformers fallback.
    Caches embeddings in memory (LRU cache, max 1000).
    """

    def __init__(self) -> None:
        """Initialize embedder with Azure OpenAI or fallback model."""
        self._azure_client: Any = None
        self._fallback_model: Any = None
        self._model_used: str = "unknown"
        self._dimension: int = EMBEDDING_DIMENSION

        # Try Azure OpenAI first
        if AzureOpenAI and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
            try:
                self._azure_client = AzureOpenAI(
                    api_key=AZURE_OPENAI_API_KEY,
                    api_version=AZURE_OPENAI_API_VERSION,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                )
                self._model_used = AZURE_OPENAI_EMBEDDING_MODEL
                logger.info("Initialized Azure OpenAI embedder: {}", AZURE_OPENAI_EMBEDDING_MODEL)
            except Exception as e:
                logger.warning("Failed to initialize Azure OpenAI embedder: {}", e)
                self._azure_client = None

        # Fallback to sentence-transformers
        if self._azure_client is None:
            if SentenceTransformer:
                try:
                    self._fallback_model = SentenceTransformer(FALLBACK_MODEL_NAME)
                    self._dimension = self._fallback_model.get_sentence_embedding_dimension()
                    self._model_used = FALLBACK_MODEL_NAME
                    logger.info("Initialized sentence-transformers embedder: {} (dim={})", FALLBACK_MODEL_NAME, self._dimension)
                except Exception as e:
                    logger.error("Failed to initialize sentence-transformers embedder: {}", e)
                    raise RuntimeError("No embedding model available")
            else:
                raise RuntimeError("No embedding model available - install sentence-transformers or configure Azure OpenAI")

    @lru_cache(maxsize=1000)
    def _cached_embed(self, text: str) -> tuple[list[float], str]:
        """
        Cached embedding (returns tuple for cache compatibility).
        Internal method - use embed() instead.
        """
        return self._embed_uncached(text), self._model_used

    def _embed_uncached(self, text: str) -> list[float]:
        """Generate embedding without cache."""
        if self._azure_client:
            return self._embed_azure(text)
        return self._embed_fallback(text)

    def _embed_azure(self, text: str) -> list[float]:
        """Generate embedding using Azure OpenAI."""
        try:
            response = self._azure_client.embeddings.create(
                input=text,
                model=AZURE_OPENAI_EMBEDDING_MODEL,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("Azure OpenAI embedding failed: {}", e)
            raise

    def _embed_fallback(self, text: str) -> list[float]:
        """Generate embedding using sentence-transformers."""
        if self._fallback_model is None:
            raise RuntimeError("Fallback model not initialized")
        embedding = self._fallback_model.encode(text, convert_to_numpy=False, normalize_embeddings=True)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector (list of floats).
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to embedder")
            return [0.0] * self._dimension

        start_time = time.time()
        embedding, _ = self._cached_embed(text.strip())
        latency_ms = int((time.time() - start_time) * 1000)

        logger.debug("Generated embedding (dim={}, latency={}ms, model={})", len(embedding), latency_ms, self._model_used)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        start_time = time.time()

        # Filter empty texts
        valid_texts = [t.strip() if t else "" for t in texts]
        valid_indices = [i for i, t in enumerate(valid_texts) if t]

        if not valid_indices:
            return [[0.0] * self._dimension] * len(texts)

        embeddings: list[list[float]] = []

        if self._azure_client:
            # Azure OpenAI supports batch
            try:
                batch_texts = [valid_texts[i] for i in valid_indices]
                response = self._azure_client.embeddings.create(
                    input=batch_texts,
                    model=AZURE_OPENAI_EMBEDDING_MODEL,
                )
                batch_embeddings = [item.embedding for item in response.data]
            except Exception as e:
                logger.error("Azure OpenAI batch embedding failed: {}", e)
                # Fallback to individual calls
                batch_embeddings = [self._embed_uncached(valid_texts[i]) for i in valid_indices]
        else:
            # sentence-transformers batch
            batch_texts = [valid_texts[i] for i in valid_indices]
            batch_embeddings_raw = self._fallback_model.encode(
                batch_texts, convert_to_numpy=False, normalize_embeddings=True
            )
            batch_embeddings = [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in batch_embeddings_raw]

        # Map back to original indices
        valid_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                embeddings.append(batch_embeddings[valid_idx])
                valid_idx += 1
            else:
                embeddings.append([0.0] * self._dimension)

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info("Generated {} embeddings (latency={}ms, model={})", len(embeddings), latency_ms, self._model_used)

        return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def model_used(self) -> str:
        """Get model name being used."""
        return self._model_used
