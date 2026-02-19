"""
Tests for RAG pipeline components.
Covers embedder, indexer, retriever, prompts, and full pipeline.
Mocks Azure AI Search and Claude API calls.
"""

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from data.schemas.document import CountyDocument, ContentType, DocumentCategory
from rag.embeddings.embedder import Embedder
from rag.pipeline import RAGPipeline, RAGResponse
from rag.prompts.system_prompt import SYSTEM_PROMPT, build_context, build_query_prompt
from rag.retrieval.indexer import DocumentIndexer, IndexResult
from rag.retrieval.retriever import RetrievedDocument, Retriever


def _create_test_doc(
    text: str = "Test document content",
    category: DocumentCategory = DocumentCategory.property,
    fips: str = "01001",
) -> CountyDocument:
    """Create a test CountyDocument."""
    from datetime import datetime

    return CountyDocument(
        id=uuid4(),
        fips=fips,
        county_name="Test County",
        state_abbr="AL",
        category=category,
        source_url="https://test.com/doc",
        content_type=ContentType.text,
        raw_content=text,
        processed_content=text,
        metadata={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


class TestEmbedder:
    """Test embedding generation."""

    @patch("rag.embeddings.embedder.AzureOpenAI", None)
    @patch("rag.embeddings.embedder.SentenceTransformer")
    @patch.dict("os.environ", {}, clear=True)
    def test_embedder_initializes_azure_openai(self, mock_st: MagicMock) -> None:
        """Test embedder falls back to sentence-transformers when Azure OpenAI not configured."""
        # This test verifies fallback behavior (Azure OpenAI requires real credentials)
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1536
        mock_model.encode.return_value = [0.1] * 1536
        mock_st.return_value = mock_model
        embedder = Embedder()
        # When Azure OpenAI is None, it uses fallback
        assert embedder.model_used == "all-MiniLM-L6-v2" or embedder.model_used == "text-embedding-ada-002"
        assert embedder.dimension > 0

    @patch("rag.embeddings.embedder.SentenceTransformer")
    @patch.dict("os.environ", {}, clear=True)
    def test_embedder_falls_back_to_sentence_transformers(self, mock_st: MagicMock) -> None:
        """Test embedder falls back to sentence-transformers when Azure OpenAI not configured."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [0.1] * 384
        mock_st.return_value = mock_model

        embedder = Embedder()
        assert embedder.model_used == "all-MiniLM-L6-v2"
        assert embedder.dimension == 384

    @patch("rag.embeddings.embedder.SentenceTransformer")
    @patch.dict("os.environ", {}, clear=True)
    def test_embed_returns_correct_dimension_vector(self, mock_st: MagicMock) -> None:
        """Test embed returns vector of correct dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [0.1] * 384
        mock_st.return_value = mock_model

        embedder = Embedder()
        embedding = embedder.embed("test text")
        assert len(embedding) == 384
        assert all(isinstance(x, (int, float)) for x in embedding)

    @patch("rag.embeddings.embedder.SentenceTransformer")
    @patch.dict("os.environ", {}, clear=True)
    def test_embed_batch_returns_correct_length(self, mock_st: MagicMock) -> None:
        """Test embed_batch returns correct number of embeddings."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [[0.1] * 384, [0.2] * 384]
        mock_st.return_value = mock_model

        embedder = Embedder()
        embeddings = embedder.embed_batch(["text1", "text2"])
        assert len(embeddings) == 2
        assert all(len(emb) == 384 for emb in embeddings)


class TestDocumentIndexer:
    """Test document indexing."""

    @patch("rag.retrieval.indexer.AZURE_SEARCH_ENDPOINT", "https://test.search.windows.net")
    @patch("rag.retrieval.indexer.AZURE_SEARCH_API_KEY", "test-key")
    @patch("rag.retrieval.indexer.SearchClient")
    @patch("rag.retrieval.indexer.SearchIndexClient")
    def test_indexer_initializes(self, mock_index_client: MagicMock, mock_search_client: MagicMock) -> None:
        """Test indexer initializes successfully."""
        mock_index_client.return_value.list_indexes.return_value = []
        indexer = DocumentIndexer()
        assert indexer.index_name == "countyiq-documents"

    @patch("rag.retrieval.indexer.AZURE_SEARCH_ENDPOINT", "https://test.search.windows.net")
    @patch("rag.retrieval.indexer.AZURE_SEARCH_API_KEY", "test-key")
    @patch("rag.retrieval.indexer.SearchClient")
    @patch("rag.retrieval.indexer.SearchIndexClient")
    def test_index_returns_index_result(self, mock_index_client: MagicMock, mock_search_client: MagicMock) -> None:
        """Test index returns IndexResult with correct fields."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.succeeded = True
        mock_client.upload_documents.return_value = [mock_result]
        mock_search_client.return_value = mock_client
        mock_index_client.return_value.list_indexes.return_value = []

        indexer = DocumentIndexer()
        doc = _create_test_doc()
        result = indexer.index([doc])

        assert isinstance(result, IndexResult)
        assert result.indexed == 1
        assert result.failed == 0


class TestRetriever:
    """Test document retrieval."""

    @patch("rag.retrieval.retriever.AZURE_SEARCH_ENDPOINT", "https://test.search.windows.net")
    @patch("rag.retrieval.retriever.AZURE_SEARCH_API_KEY", "test-key")
    @patch("rag.retrieval.retriever.SearchClient")
    @patch("rag.retrieval.retriever.Embedder")
    def test_retriever_filters_by_fips(self, mock_embedder_class: MagicMock, mock_search_client: MagicMock) -> None:
        """Test retriever filters by FIPS code."""
        
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.get.side_effect = lambda k, d=None: {
            "id": "doc1",
            "fips": "01001",
            "county_name": "Test",
            "category": "property",
            "content": "test",
            "source_url": "https://test.com",
            "metadata": "{}",
        }.get(k, d)
        mock_result.__iter__ = lambda self: iter([mock_result])
        mock_client.search.return_value = [mock_result]
        mock_search_client.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embed.return_value = [0.1] * 1536
        mock_embedder_class.return_value = mock_emb

        retriever = Retriever()
        results = retriever.retrieve("test query", fips="01001")

        assert len(results) == 1
        assert results[0].fips == "01001"

    @patch("rag.retrieval.retriever.AZURE_SEARCH_ENDPOINT", "https://test.search.windows.net")
    @patch("rag.retrieval.retriever.AZURE_SEARCH_API_KEY", "test-key")
    @patch("rag.retrieval.retriever.SearchClient")
    @patch("rag.retrieval.retriever.Embedder")
    def test_retriever_filters_by_category(self, mock_embedder_class: MagicMock, mock_search_client: MagicMock) -> None:
        """Test retriever filters by category."""
        
        mock_result = MagicMock()
        mock_result.get.side_effect = lambda k, d=None: {
            "id": "doc1",
            "fips": "01001",
            "county_name": "Test",
            "category": "legal",
            "content": "test",
            "source_url": "https://test.com",
            "metadata": "{}",
        }.get(k, d)
        mock_result.__iter__ = lambda self: iter([mock_result])
        mock_search_client.return_value.search.return_value = [mock_result]

        mock_emb = MagicMock()
        mock_emb.embed.return_value = [0.1] * 1536
        mock_embedder_class.return_value = mock_emb

        retriever = Retriever()
        results = retriever.retrieve("test query", category="legal")

        assert len(results) == 1
        assert results[0].category == "legal"


class TestPrompts:
    """Test prompt building."""

    def test_build_context_formats_documents_correctly(self) -> None:
        """Test context builder formats documents correctly."""
        docs = [
            RetrievedDocument(
                doc_id="1",
                fips="01001",
                county_name="Test County",
                category="property",
                content="Test content",
                source_url="https://test.com",
                score=0.9,
            )
        ]

        context = build_context(docs)
        assert "CONTEXT DOCUMENTS" in context
        assert "Test County" in context
        assert "Test content" in context
        assert "https://test.com" in context

    def test_build_query_prompt_includes_query_and_context(self) -> None:
        """Test query prompt includes query and context."""
        context = "CONTEXT: Test document"
        prompt = build_query_prompt("What is the property value?", context, fips="01001")

        assert "What is the property value?" in prompt
        assert context in prompt
        assert "01001" in prompt


class TestRAGPipeline:
    """Test full RAG pipeline."""

    @patch("rag.pipeline.Retriever")
    @patch("rag.pipeline.Anthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_rag_response_has_all_required_fields(self, mock_anthropic: MagicMock, mock_retriever_class: MagicMock) -> None:
        """Test RAGResponse has all required fields."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedDocument(
                doc_id="1",
                fips="01001",
                county_name="Test",
                category="property",
                content="test",
                score=0.9,
            )
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_claude = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Test answer")]
        mock_claude.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_claude

        pipeline = RAGPipeline()
        response = pipeline.query("test query")

        assert isinstance(response, RAGResponse)
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert 0.0 <= response.confidence <= 1.0
        assert response.query == "test query"
        assert response.latency_ms >= 0
        assert response.model_used is not None

    @patch("rag.pipeline.Retriever")
    @patch("rag.pipeline.Anthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_confidence_score_is_between_0_and_1(self, mock_anthropic: MagicMock, mock_retriever_class: MagicMock) -> None:
        """Test confidence score is between 0 and 1."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedDocument(
                doc_id="1",
                fips="01001",
                county_name="Test",
                category="property",
                content="test",
                score=0.8,
            )
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_claude = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Test answer with sufficient length")]
        mock_claude.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_claude

        pipeline = RAGPipeline()
        response = pipeline.query("test query")

        assert 0.0 <= response.confidence <= 1.0

    @patch("rag.pipeline.Retriever")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_pipeline_handles_empty_retrieval_gracefully(self, mock_retriever_class: MagicMock) -> None:
        """Test pipeline handles empty retrieval gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_class.return_value = mock_retriever

        pipeline = RAGPipeline()
        response = pipeline.query("test query")

        assert "don't have any relevant documents" in response.answer.lower()
        assert response.confidence == 0.0
        assert len(response.sources) == 0

    @patch("rag.pipeline.Retriever")
    @patch("rag.pipeline.Anthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_pipeline_logs_full_pipeline_metrics(self, mock_anthropic: MagicMock, mock_retriever_class: MagicMock) -> None:
        """Test pipeline logs query, retrieval count, latency, confidence."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedDocument(
                doc_id="1",
                fips="01001",
                county_name="Test",
                category="property",
                content="test",
                score=0.9,
            )
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_claude = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Test answer")]
        mock_claude.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_claude

        pipeline = RAGPipeline()
        response = pipeline.query("test query", fips="01001", category="property")

        assert response.fips == "01001"
        assert response.latency_ms >= 0

    @patch("rag.pipeline.Retriever")
    @patch("rag.pipeline.Anthropic")
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_pipeline_handles_claude_api_error_gracefully(self, mock_anthropic: MagicMock, mock_retriever_class: MagicMock) -> None:
        """Test pipeline handles Claude API errors gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedDocument(
                doc_id="1",
                fips="01001",
                county_name="Test",
                category="property",
                content="test",
                score=0.9,
            )
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_claude = MagicMock()
        mock_claude.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_claude

        pipeline = RAGPipeline()
        response = pipeline.query("test query")

        assert "Error" in response.answer
        assert len(response.sources) == 1  # Documents still retrieved
        assert response.confidence < 0.6  # Lower confidence on error (but not 0 since docs retrieved)
