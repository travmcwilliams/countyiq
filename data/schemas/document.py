"""
Pydantic v2 models for county documents (crawled content and user uploads).
"""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentCategory(str, Enum):
    """Category of county data or upload."""

    property = "property"
    legal = "legal"
    demographics = "demographics"
    permits = "permits"
    zoning = "zoning"
    courts = "courts"
    tax = "tax"
    user_upload = "user_upload"


class ContentType(str, Enum):
    """Format of stored content."""

    html = "html"
    pdf = "pdf"
    structured = "structured"
    text = "text"


# DP-100: Data asset schema - Document as the unit for indexing and RAG
class CountyDocument(BaseModel):
    """
    Base unit stored after crawling or user upload.
    Used for indexing, embeddings, and RAG retrieval.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique document identifier")
    fips: str = Field(..., min_length=5, max_length=5)
    county_name: str = Field(..., min_length=1)
    state_abbr: str = Field(..., min_length=2, max_length=2)
    category: DocumentCategory = Field(...)
    source_url: str | None = Field(None)
    content_type: ContentType = Field(...)
    raw_content: str = Field("", description="Original crawled or uploaded content")
    processed_content: str | None = Field(None, description="Cleaned/normalized text for retrieval")
    embedding: list[float] | None = Field(None, description="Vector embedding for similarity search")
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"str_strip_whitespace": True, "use_enum_values": True}
