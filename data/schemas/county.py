"""
Pydantic v2 models for county registry and crawl tracking.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

CrawlStatusValue = Literal["pending", "active", "complete", "failed"]


# DP-100: Data schema - Structured definition for county and crawl metadata
class DataSources(BaseModel):
    """URLs for each data category; null if unknown or not applicable."""

    property: str | None = None
    legal: str | None = None
    demographics: str | None = None
    permits: str | None = None
    zoning: str | None = None
    courts: str | None = None
    tax: str | None = None


class CrawlStatus(BaseModel):
    """Per-category crawl status for a county."""

    property: CrawlStatusValue = "pending"
    legal: CrawlStatusValue = "pending"
    demographics: CrawlStatusValue = "pending"
    permits: CrawlStatusValue = "pending"
    zoning: CrawlStatusValue = "pending"
    courts: CrawlStatusValue = "pending"
    tax: CrawlStatusValue = "pending"


class County(BaseModel):
    """
    County registry entry with FIPS, name, state, population, and crawl metadata.
    """

    fips: str = Field(..., min_length=5, max_length=5, description="5-digit FIPS code")
    county_name: str = Field(..., min_length=1)
    state_name: str = Field(..., min_length=1)
    state_abbr: str = Field(..., min_length=2, max_length=2)
    population: int | None = Field(None, ge=0, description="Census population (approximate OK)")
    data_sources: DataSources = Field(default_factory=DataSources)
    crawl_status: CrawlStatus = Field(default_factory=CrawlStatus)
    last_crawled: datetime | None = Field(None, description="ISO timestamp of last crawl or null")

    model_config = {"str_strip_whitespace": True, "extra": "forbid"}


class CrawlRecord(BaseModel):
    """
    Log entry for a single crawl attempt (one category, one county).
    """

    fips: str = Field(..., min_length=5, max_length=5)
    category: str = Field(..., description="One of: property, legal, demographics, permits, zoning, courts, tax")
    url: str | None = Field(None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(...)
    record_count: int | None = Field(None, ge=0)
    error_message: str | None = Field(None)

    model_config = {"str_strip_whitespace": True}
