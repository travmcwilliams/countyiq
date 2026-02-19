"""Live end-to-end RAG test: index Autauga County document and query it."""

import json
import pathlib
import uuid
from datetime import datetime

from data.schemas.document import CountyDocument, ContentType, DocumentCategory
from rag.embeddings.embedder import Embedder
from rag.pipeline import RAGPipeline
from rag.retrieval.indexer import DocumentIndexer

# Load saved crawl document
raw_path = pathlib.Path("data/raw/01001/property")
files = list(raw_path.glob("*.json"))
print(f"Found {len(files)} crawled documents")

if not files:
    print("No documents found!")
    exit(1)

with open(files[0]) as f:
    data = json.load(f)

doc = CountyDocument(**data)
print(f"Loaded document: {doc.id}, category: {doc.category}, content length: {len(doc.raw_content)}")

# Generate embedding if not present
embedder = Embedder()
if not doc.embedding:
    text = doc.processed_content if doc.processed_content else doc.raw_content
    try:
        doc.embedding = embedder.embed(text)
        print(f"Generated embedding: {len(doc.embedding)} dims, model: {embedder.model_used}")
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        print("Falling back to sentence-transformers...")
        # Force fallback by clearing Azure client
        embedder._azure_client = None
        if embedder._fallback_model is None:
            from sentence_transformers import SentenceTransformer
            embedder._fallback_model = SentenceTransformer("all-MiniLM-L6-v2")
            embedder._dimension = embedder._fallback_model.get_sentence_embedding_dimension()
            embedder._model_used = "all-MiniLM-L6-v2"
        doc.embedding = embedder.embed(text)
        print(f"Generated embedding (fallback): {len(doc.embedding)} dims, model: {embedder.model_used}")

# Index it
print("\n--- Indexing document ---")
indexer = DocumentIndexer()
result = indexer.index([doc])
print(f"Indexed: {result.indexed}, Failed: {result.failed}")
if result.errors:
    print(f"Errors: {result.errors}")

# Wait a moment for index to be ready
import time

time.sleep(2)

# Query it
print("\n--- Querying RAG pipeline ---")
pipeline = RAGPipeline()
response = pipeline.query(
    user_query="What property information is available for Autauga County Alabama?",
    fips="01001",
)

print(f"\nAnswer: {response.answer[:500]}")
print(f"\nConfidence: {response.confidence:.3f}")
print(f"Sources: {len(response.sources)}")
print(f"Latency: {response.latency_ms}ms")
print(f"Model used: {response.model_used}")

if response.sources:
    print("\n--- Source Documents ---")
    for i, source in enumerate(response.sources, 1):
        print(f"{i}. {source.county_name} ({source.fips}) - {source.category}")
        print(f"   Score: {source.score:.3f}")
        if source.source_url:
            print(f"   URL: {source.source_url}")
