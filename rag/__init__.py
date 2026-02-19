"""RAG package: embeddings, retrieval, and prompt logic."""

from rag.embeddings.embedder import Embedder
from rag.pipeline import RAGPipeline, RAGResponse
from rag.retrieval.indexer import DocumentIndexer, IndexResult
from rag.retrieval.retriever import RetrievedDocument, Retriever

__all__ = [
    "DocumentIndexer",
    "Embedder",
    "IndexResult",
    "RAGPipeline",
    "RAGResponse",
    "RetrievedDocument",
    "Retriever",
]
