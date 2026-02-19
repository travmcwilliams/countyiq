"""
RAG pipeline orchestrator.
Embeds query → retrieves documents → builds prompt → calls Claude API → parses response.
# DP-100: Model deployment - End-to-end inference pipeline for production RAG system.
"""

import os
import time
from typing import Any

from anthropic import Anthropic
from loguru import logger
from pydantic import BaseModel, Field

from data.schemas.document import DocumentCategory
from rag.embeddings.embedder import Embedder
from rag.prompts.system_prompt import SYSTEM_PROMPT, build_context, build_query_prompt
from rag.retrieval.retriever import RetrievedDocument, Retriever

# Claude API configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")


class RAGResponse(BaseModel):
    """Response from RAG pipeline."""

    answer: str = Field(..., description="Generated answer")
    sources: list[RetrievedDocument] = Field(default_factory=list, description="Source documents used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    query: str = Field(..., description="Original query")
    fips: str | None = Field(None, description="County FIPS if specified")
    latency_ms: int = Field(..., ge=0, description="Total pipeline latency in milliseconds")
    model_used: str = Field(..., description="Claude model used")


# DP-100: Inference pipeline - Orchestrating retrieval, prompt building, and generation
class RAGPipeline:
    """
    End-to-end RAG pipeline.
    Orchestrates embedding, retrieval, prompt building, and Claude API call.
    """

    def __init__(self) -> None:
        """Initialize RAG pipeline."""
        self._retriever = Retriever()
        self._embedder = Embedder()

        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set - Claude API calls will fail")

        try:
            self._claude_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
            logger.info("Initialized RAG pipeline (Claude model: {})", CLAUDE_MODEL)
        except Exception as e:
            logger.error("Failed to initialize Claude client: {}", e)
            self._claude_client = None

    def _calculate_confidence(
        self,
        retrieved_docs: list[RetrievedDocument],
        answer: str,
    ) -> float:
        """
        Calculate confidence score based on retrieval scores, answer length, and source count.

        Args:
            retrieved_docs: Retrieved documents.
            answer: Generated answer.

        Returns:
            Confidence score 0.0-1.0.
        """
        if not retrieved_docs:
            return 0.0

        # Average retrieval score (weight: 0.5)
        avg_score = sum(doc.score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
        score_component = avg_score * 0.5

        # Answer length factor (weight: 0.2) - longer answers suggest more information
        answer_length_factor = min(1.0, len(answer) / 500.0) * 0.2

        # Source count factor (weight: 0.3) - more sources = higher confidence
        source_count_factor = min(1.0, len(retrieved_docs) / 5.0) * 0.3

        confidence = score_component + answer_length_factor + source_count_factor
        return min(1.0, max(0.0, confidence))

    def query(
        self,
        user_query: str,
        fips: str | None = None,
        category: str | None = None,
        top_k: int = 5,
    ) -> RAGResponse:
        """
        Process a user query through the RAG pipeline.

        Args:
            user_query: User's question.
            fips: Optional county FIPS filter.
            category: Optional document category filter.
            top_k: Number of documents to retrieve (default: 5).

        Returns:
            RAGResponse with answer, sources, confidence, and metadata.
        """
        pipeline_start = time.time()

        if not user_query or not user_query.strip():
            return RAGResponse(
                answer="Please provide a valid query.",
                sources=[],
                confidence=0.0,
                query=user_query or "",
                fips=fips,
                latency_ms=0,
                model_used=CLAUDE_MODEL,
            )

        # Step 1: Retrieve documents
        retrieved_docs = self._retriever.retrieve(
            query=user_query,
            fips=fips,
            category=category,
            top_k=top_k,
        )

        if not retrieved_docs:
            return RAGResponse(
                answer="I don't have any relevant documents to answer your question.",
                sources=[],
                confidence=0.0,
                query=user_query,
                fips=fips,
                latency_ms=int((time.time() - pipeline_start) * 1000),
                model_used=CLAUDE_MODEL,
            )

        # Step 2: Build prompt
        context = build_context(retrieved_docs)
        user_prompt = build_query_prompt(user_query, context, fips)

        # Step 3: Call Claude API
        if self._claude_client is None:
            logger.error("Claude client not initialized")
            return RAGResponse(
                answer="Error: Claude API not configured.",
                sources=retrieved_docs,
                confidence=0.0,
                query=user_query,
                fips=fips,
                latency_ms=int((time.time() - pipeline_start) * 1000),
                model_used=CLAUDE_MODEL,
            )

        try:
            # DP-100: Model inference - Calling Claude API for text generation
            message = self._claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            answer = message.content[0].text if message.content else "No response generated."
        except Exception as e:
            logger.error("Claude API call failed: {}", e)
            answer = f"Error generating answer: {e}"

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(retrieved_docs, answer)

        latency_ms = int((time.time() - pipeline_start) * 1000)

        logger.info(
            "RAG pipeline complete (query='{}', retrieved={}, confidence={:.2f}, latency={}ms)",
            user_query[:50],
            len(retrieved_docs),
            confidence,
            latency_ms,
        )

        return RAGResponse(
            answer=answer,
            sources=retrieved_docs,
            confidence=confidence,
            query=user_query,
            fips=fips,
            latency_ms=latency_ms,
            model_used=CLAUDE_MODEL,
        )
