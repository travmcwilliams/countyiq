"""
Document indexing for Azure AI Search.
Indexes CountyDocuments with embeddings for vector similarity search.
# DP-100: Vector indexing - Storing embeddings in Azure AI Search for scalable retrieval.
"""

import json
import os
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    VectorSearchVectorizer,
    VectorSearchVectorizerKind,
)
from loguru import logger
from pydantic import BaseModel, Field

from data.schemas.document import CountyDocument

# Azure AI Search configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "countyiq-documents")

EMBEDDING_DIMENSION = 1536  # text-embedding-ada-002


class IndexResult(BaseModel):
    """Result of indexing operation."""

    indexed: int = Field(..., ge=0, description="Number of documents successfully indexed")
    failed: int = Field(..., ge=0, description="Number of documents that failed to index")
    errors: list[str] = Field(default_factory=list, description="Error messages for failed documents")


# DP-100: Vector indexing - Creating search index with vector fields
class DocumentIndexer:
    """
    Indexes CountyDocuments in Azure AI Search with embeddings.
    Creates index if it doesn't exist.
    """

    def __init__(self, index_name: str | None = None) -> None:
        """
        Initialize indexer.

        Args:
            index_name: Index name (default: from env or "countyiq-documents").
        """
        self.index_name = index_name or AZURE_SEARCH_INDEX_NAME

        if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
            raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be set")

        self._search_client: SearchClient | None = None
        self._index_client: SearchIndexClient | None = None

        try:
            credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
            self._search_client = SearchClient(
                endpoint=AZURE_SEARCH_ENDPOINT,
                index_name=self.index_name,
                credential=credential,
            )
            self._index_client = SearchIndexClient(
                endpoint=AZURE_SEARCH_ENDPOINT,
                credential=credential,
            )
            logger.info("Initialized Azure AI Search client for index: {}", self.index_name)
        except Exception as e:
            logger.error("Failed to initialize Azure AI Search client: {}", e)
            raise

        # Ensure index exists
        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        """Create index if it doesn't exist."""
        try:
            existing_indexes = [idx.name for idx in self._index_client.list_indexes()]
            if self.index_name in existing_indexes:
                logger.debug("Index {} already exists", self.index_name)
                return

            logger.info("Creating index: {}", self.index_name)

            # DP-100: Vector search - Defining vector field and search configuration
            index = SearchIndex(
                name=self.index_name,
                fields=[
                    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                    SimpleField(name="fips", type=SearchFieldDataType.String, filterable=True, facetable=True),
                    SimpleField(name="county_name", type=SearchFieldDataType.String, filterable=True),
                    SimpleField(name="state_abbr", type=SearchFieldDataType.String, filterable=True),
                    SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                    SimpleField(name="source_url", type=SearchFieldDataType.String),
                    SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
                    SearchField(
                        name="embedding",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True,
                        vector_search_dimensions=384,  # sentence-transformers dimension
                        vector_search_profile_name="default-vector-profile",
                    ),
                    SimpleField(name="metadata", type=SearchFieldDataType.String),
                    SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True),
                ],
                vector_search=VectorSearch(
                    algorithms=[
                        VectorSearchAlgorithmConfiguration(
                            name="default-algorithm",
                            kind="hnsw",
                        )
                    ],
                    profiles=[
                        VectorSearchProfile(
                            name="default-vector-profile",
                            algorithm_configuration_name="default-algorithm",
                        )
                    ],
                ),
            )

            self._index_client.create_index(index)
            logger.success("Created index: {}", self.index_name)
        except Exception as e:
            logger.error("Failed to create index with vectorizer: {}", e)
            # Try creating without vectorizer (free tier limitation)
            logger.info("Retrying index creation without vectorizer...")
            try:
                index_no_vectorizer = SearchIndex(
                    name=self.index_name,
                    fields=[
                        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                        SimpleField(name="fips", type=SearchFieldDataType.String, filterable=True, facetable=True),
                        SimpleField(name="county_name", type=SearchFieldDataType.String, filterable=True),
                        SimpleField(name="state_abbr", type=SearchFieldDataType.String, filterable=True),
                        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                        SimpleField(name="source_url", type=SearchFieldDataType.String),
                        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
                        SearchField(
                            name="embedding",
                            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True,
                            vector_search_dimensions=384,  # sentence-transformers dimension
                            vector_search_profile_name="default-vector-profile",
                        ),
                        SimpleField(name="metadata", type=SearchFieldDataType.String),
                        SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True),
                    ],
                    vector_search=VectorSearch(
                        algorithms=[
                            VectorSearchAlgorithmConfiguration(
                                name="default-algorithm",
                                kind="hnsw",
                            )
                        ],
                        profiles=[
                            VectorSearchProfile(
                                name="default-vector-profile",
                                algorithm_configuration_name="default-algorithm",
                            )
                        ],
                    ),
                )
                self._index_client.create_index(index_no_vectorizer)
                logger.success("Created index without vectorizer: {}", self.index_name)
            except Exception as e2:
                logger.error("Failed to create index without vectorizer: {}", e2)
            # Try creating a basic index without vector search (for free tier compatibility)
            logger.info("Retrying with basic index (no vector search)...")
            try:
                basic_index = SearchIndex(
                    name=self.index_name,
                    fields=[
                        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                        SimpleField(name="fips", type=SearchFieldDataType.String, filterable=True, facetable=True),
                        SimpleField(name="county_name", type=SearchFieldDataType.String, filterable=True),
                        SimpleField(name="state_abbr", type=SearchFieldDataType.String, filterable=True),
                        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                        SimpleField(name="source_url", type=SearchFieldDataType.String),
                        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
                        # Note: Embedding field removed for basic index - will use keyword search only
                        SimpleField(name="metadata", type=SearchFieldDataType.String),
                        SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True),
                    ],
                )
                self._index_client.create_index(basic_index)
                logger.success("Created basic index (no vector search): {}", self.index_name)
            except Exception as e3:
                logger.error("Failed to create basic index: {}", e3)
                raise

    def _document_to_search_doc(self, doc: CountyDocument) -> dict[str, Any]:
        """Convert CountyDocument to Azure AI Search document format."""
        content = doc.processed_content if doc.processed_content else doc.raw_content

        # Format datetime for Azure AI Search (requires timezone-aware ISO format)
        created_at_str = None
        if doc.created_at:
            # Ensure timezone-aware datetime
            if doc.created_at.tzinfo is None:
                from datetime import timezone
                created_at = doc.created_at.replace(tzinfo=timezone.utc)
            else:
                created_at = doc.created_at
            # Format as ISO 8601 with timezone
            created_at_str = created_at.isoformat()
        
        doc_dict = {
            "id": str(doc.id),
            "fips": doc.fips,
            "county_name": doc.county_name,
            "state_abbr": doc.state_abbr,
            "category": doc.category.value if hasattr(doc.category, "value") else str(doc.category),
            "source_url": doc.source_url or "",
            "content": content or "",
            "metadata": json.dumps(doc.metadata) if doc.metadata else "{}",
            "created_at": created_at_str,
        }
        
        # Only include embedding if index supports it (has vector search config)
        # Check if index has embedding field by trying to get index definition
        try:
            index_def = self._index_client.get_index(self.index_name)
            embedding_fields = [f.name for f in index_def.fields if hasattr(f, 'vector_search_dimensions') and f.vector_search_dimensions]
            if "embedding" in embedding_fields and doc.embedding:
                doc_dict["embedding"] = doc.embedding
        except Exception:
            # Index might not exist yet or doesn't support embeddings - skip embedding field
            pass
        
        return doc_dict

    def index(self, documents: list[CountyDocument]) -> IndexResult:
        """
        Index documents in Azure AI Search.

        Args:
            documents: List of CountyDocument instances to index.

        Returns:
            IndexResult with indexed count, failed count, and errors.
        """
        if not documents:
            return IndexResult(indexed=0, failed=0, errors=[])

        search_docs = [self._document_to_search_doc(doc) for doc in documents]
        indexed_count = 0
        failed_count = 0
        errors: list[str] = []

        try:
            # Upload documents
            result = self._search_client.upload_documents(documents=search_docs)
            for r in result:
                if r.succeeded:
                    indexed_count += 1
                else:
                    failed_count += 1
                    errors.append(f"Document {r.key}: {r.error_message or 'Unknown error'}")

            logger.info("Indexed {} documents, {} failed", indexed_count, failed_count)
        except Exception as e:
            logger.error("Indexing failed: {}", e)
            failed_count = len(documents)
            errors.append(str(e))

        return IndexResult(indexed=indexed_count, failed=failed_count, errors=errors)

    def index_batch(self, documents: list[CountyDocument], batch_size: int = 100) -> IndexResult:
        """
        Index documents in batches.

        Args:
            documents: List of CountyDocument instances to index.
            batch_size: Batch size for indexing (default: 100).

        Returns:
            IndexResult aggregated across all batches.
        """
        if not documents:
            return IndexResult(indexed=0, failed=0, errors=[])

        total_indexed = 0
        total_failed = 0
        all_errors: list[str] = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            result = self.index(batch)
            total_indexed += result.indexed
            total_failed += result.failed
            all_errors.extend(result.errors)

        logger.info("Batch indexing complete: {} indexed, {} failed across {} batches", total_indexed, total_failed, (len(documents) + batch_size - 1) // batch_size)

        return IndexResult(indexed=total_indexed, failed=total_failed, errors=all_errors)

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the index.

        Args:
            doc_id: Document ID to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        try:
            result = self._search_client.delete_documents(documents=[{"id": doc_id}])
            success = result[0].succeeded if result else False
            if success:
                logger.debug("Deleted document: {}", doc_id)
            else:
                logger.warning("Failed to delete document: {}", doc_id)
            return success
        except Exception as e:
            logger.error("Delete failed for document {}: {}", doc_id, e)
            return False
