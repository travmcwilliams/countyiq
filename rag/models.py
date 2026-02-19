"""RAG response models and document categories.

# DP-100: Responsible AI - Structured response models for confidence scoring and source grounding.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentCategory(str, Enum):
    """Document categories for county data."""

    PROPERTY = "property"
    LEGAL = "legal"
    DEMOGRAPHICS = "demographics"
    TAX = "tax"
    ZONING = "zoning"
    PERMITS = "permits"
    COURTS = "courts"


class SourceCitation(BaseModel):
    """Source document citation with relevance score."""

    document_id: str
    county_fips: str = Field(..., min_length=5, max_length=5, description="5-digit county FIPS code")
    source_url: str
    page_number: Optional[int] = None
    excerpt: str = Field(..., max_length=1000)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    category: Optional[DocumentCategory] = None
    timestamp: Optional[datetime] = None  # When document was crawled


class RAGResponse(BaseModel):
    """RAG response with source grounding and confidence scoring.

    # DP-100: Responsible AI - Response model includes confidence score and source citations
    for transparency and accountability.
    """

    answer: str
    sources: List[SourceCitation] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    county_fips: str = Field(..., min_length=5, max_length=5, description="5-digit county FIPS code")
    hallucination_detected: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: int = Field(default=0, ge=0, description="Query latency in milliseconds")
    category: Optional[DocumentCategory] = None

    @property
    def confidence(self) -> float:
        """Alias for confidence_score (for compatibility with RAGResponseLike protocol)."""
        return self.confidence_score
