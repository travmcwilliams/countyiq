"""
Document retrieval for RAG pipeline.
Hybrid search: vector similarity + keyword search with optional filters.
# DP-100: Retrieval - Hybrid search combining vector similarity and keyword matching.
"""

import os
import time
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from loguru import logger
from pydantic import BaseModel, Field

from rag.embeddings.embedder import Embedder

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "countyiq-documents")


class RetrievedDocument(BaseModel):
    """Retrieved document from search."""

    doc_id: str = Field(..., description="Document ID")
    fips: str = Field(..., description="County FIPS code")
    county_name: str = Field(..., description="County name")
    category: str = Field(..., description="Document category")
    content: str = Field(..., description="Document content")
    source_url: str | None = Field(None, description="Source URL")
    score: float = Field(..., ge=0.0, le=1.0, description="Retrieval score")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


# DP-100: Similarity search - Vector and keyword hybrid retrieval
class Retriever:
    """
    Retrieves relevant documents using hybrid search (vector + keyword).
    Supports filtering by FIPS and category.
    """

    def __init__(self, index_name: str | None = None) -> None:
        """
        Initialize retriever.

        Args:
            index_name: Index name (default: from env or "countyiq-documents").
        """
        self.index_name = index_name or AZURE_SEARCH_INDEX_NAME
        self._search_client: SearchClient | None = None
        self._embedder = Embedder()

        if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
            raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be set")

        try:
            credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
            self._search_client = SearchClient(
                endpoint=AZURE_SEARCH_ENDPOINT,
                index_name=self.index_name,
                credential=credential,
            )
            logger.info("Initialized Azure AI Search retriever for index: {}", self.index_name)
        except Exception as e:
            logger.error("Failed to initialize Azure AI Search retriever: {}", e)
            raise

    def retrieve(
        self,
        query: str,
        fips: str | None = None,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents using hybrid search.

        Args:
            query: Search query text.
            fips: Optional county FIPS filter.
            category: Optional document category filter.
            top_k: Number of results to return (default: 5).

        Returns:
            List of RetrievedDocument instances sorted by score.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to retriever")
            return []

        start_time = time.time()

        # Generate query embedding
        query_embedding = self._embedder.embed(query)

        # Build filter expression
        filter_expr = self._build_filter(fips, category)

        # DP-100: Hybrid search - Combining vector similarity and keyword search
        try:
            # Check if index supports vector search by checking for embedding field
            try:
                from azure.search.documents.indexes import SearchIndexClient
                from azure.core.credentials import AzureKeyCredential
                import os
                
                index_client = SearchIndexClient(
                    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
                    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY", "")),
                )
                index_def = index_client.get_index(self.index_name)
                has_vector_field = any(
                    hasattr(f, 'vector_search_dimensions') and f.vector_search_dimensions
                    for f in index_def.fields
                    if f.name == "embedding"
                )
            except Exception:
                has_vector_field = False
            
            # Try hybrid search if vector field exists, otherwise keyword-only
            if has_vector_field and query_embedding:
                vector_query = VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=top_k,
                    fields="embedding",
                )
                # Hybrid search: vector + keyword
                results = self._search_client.search(
                    search_text=query,  # Keyword search
                    vector_queries=[vector_query],  # Vector search
                    filter=filter_expr,
                    top=top_k,
                    select=["id", "fips", "county_name", "category", "content", "source_url", "metadata"],
                )
            else:
                # Keyword-only search (fallback when vector search not available)
                logger.debug("Using keyword-only search (vector search not available)")
                results = self._search_client.search(
                    search_text=query,  # Keyword search only
                    filter=filter_expr,
                    top=top_k,
                    select=["id", "fips", "county_name", "category", "content", "source_url", "metadata"],
                )

            retrieved_docs: list[RetrievedDocument] = []
            for result in results:
                # Normalize score to 0-1 range
                score = float(result.get("@search.score", 0.0))
                # Azure AI Search scores can be > 1, normalize
                normalized_score = min(1.0, score / 10.0) if score > 1.0 else score

                try:
                    metadata_str = result.get("metadata", "{}")
                    import json

                    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                except Exception:
                    metadata = {}

                retrieved_docs.append(
                    RetrievedDocument(
                        doc_id=str(result.get("id", "")),
                        fips=str(result.get("fips", "")),
                        county_name=str(result.get("county_name", "")),
                        category=str(result.get("category", "")),
                        content=str(result.get("content", "")),
                        source_url=result.get("source_url"),
                        score=normalized_score,
                        metadata=metadata,
                    )
                )

            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "Retrieved {} documents (query='{}', fips={}, category={}, latency={}ms)",
                len(retrieved_docs),
                query[:50],
                fips,
                category,
                latency_ms,
            )

            return retrieved_docs
        except Exception as e:
            logger.error("Retrieval failed: {}", e)
            return []

    def _build_filter(self, fips: str | None, category: str | None) -> str | None:
        """
        Build OData filter expression.

        Args:
            fips: Optional FIPS filter.
            category: Optional category filter.

        Returns:
            OData filter string or None.
        """
        filters: list[str] = []

        if fips:
            filters.append(f"fips eq '{fips}'")

        if category:
            filters.append(f"category eq '{category}'")

        if not filters:
            return None

        return " and ".join(filters)
