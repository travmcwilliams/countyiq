"""Confidence filter: ensures responses meet minimum quality thresholds.

# DP-100: Responsible AI - Model confidence thresholds prevent low-quality responses
from being served to users. Critical for maintaining trust and accuracy.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from rag.models import RAGResponse, SourceCitation


class FilteredResponse(BaseModel):
    """Response after confidence filtering."""

    original_response: RAGResponse
    filtered_answer: str
    confidence_sufficient: bool = Field(default=False)
    disclaimer_added: bool = Field(default=False)
    filter_reason: Optional[str] = Field(default=None)


class ConfidenceFilter:
    """
    Filters RAG responses based on confidence thresholds and data freshness.

    # DP-100: Responsible AI - Confidence filtering ensures users receive
    only responses that meet quality standards. Below-threshold responses
    are replaced with "I don't have sufficient information" messages.
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
        min_source_count: int = 1,
        outdated_days: int = 90,
    ):
        """
        Initialize confidence filter.

        Args:
            min_confidence: Minimum confidence score (0.0-1.0) to serve response.
            min_source_count: Minimum number of sources required.
            outdated_days: Days after which data is considered outdated.
        """
        self.min_confidence = min_confidence
        self.min_source_count = min_source_count
        self.outdated_days = outdated_days

    def filter(self, response: RAGResponse) -> FilteredResponse:
        """
        Filter response based on confidence and source requirements.

        # DP-100: Responsible AI - Confidence filtering prevents serving
        low-quality responses that could mislead users.

        Args:
            response: Original RAG response.

        Returns:
            FilteredResponse with filtered answer and metadata.
        """
        # Check minimum confidence threshold
        if response.confidence_score < self.min_confidence:
            logger.warning(
                "Response filtered: confidence {} below threshold {}",
                response.confidence_score,
                self.min_confidence,
            )
            return FilteredResponse(
                original_response=response,
                filtered_answer="I don't have sufficient information to answer this question accurately. Please consult the county directly or try rephrasing your query.",
                confidence_sufficient=False,
                disclaimer_added=False,
                filter_reason=f"confidence {response.confidence_score:.2f} < {self.min_confidence}",
            )

        # Check minimum source count
        if len(response.sources) < self.min_source_count:
            logger.warning(
                "Response filtered: {} sources below minimum {}",
                len(response.sources),
                self.min_source_count,
            )
            return FilteredResponse(
                original_response=response,
                filtered_answer="I don't have sufficient information to answer this question accurately. Please consult the county directly or try rephrasing your query.",
                confidence_sufficient=False,
                disclaimer_added=False,
                filter_reason=f"source count {len(response.sources)} < {self.min_source_count}",
            )

        # Apply disclaimers based on confidence level and data freshness
        answer = response.answer
        disclaimers: List[str] = []
        disclaimer_added = False

        # Medium confidence disclaimer (0.3-0.6)
        if 0.3 <= response.confidence_score < 0.6:
            disclaimer = "This answer is based on limited available data and may be incomplete."
            answer = f"{answer}\n\n*Note: {disclaimer}*"
            disclaimers.append(disclaimer)
            disclaimer_added = True
            logger.info("Added medium confidence disclaimer for confidence {}", response.confidence_score)

        # Check for outdated data
        if self._is_data_outdated(response.sources):
            disclaimer = "This information may be outdated. Please verify with the county directly."
            answer = f"{answer}\n\n*Note: {disclaimer}*"
            disclaimers.append(disclaimer)
            disclaimer_added = True
            logger.info("Added outdated data disclaimer for county {}", response.county_fips)

        return FilteredResponse(
            original_response=response,
            filtered_answer=answer,
            confidence_sufficient=True,
            disclaimer_added=disclaimer_added,
            filter_reason=None,
        )

    def is_sufficient(self, response: RAGResponse) -> bool:
        """
        Check if response meets minimum quality thresholds.

        Args:
            response: RAG response to check.

        Returns:
            True if response meets thresholds, False otherwise.
        """
        if response.confidence_score < self.min_confidence:
            return False
        if len(response.sources) < self.min_source_count:
            return False
        return True

    def _is_data_outdated(self, sources: List[SourceCitation]) -> bool:
        """
        Check if any source data is older than outdated_days threshold.

        Args:
            sources: List of source citations.

        Returns:
            True if any source is outdated, False otherwise.
        """
        if not sources:
            return False

        cutoff_date = datetime.utcnow() - timedelta(days=self.outdated_days)

        for source in sources:
            # If timestamp is missing, assume data is recent (conservative)
            if source.timestamp is None:
                continue
            if source.timestamp < cutoff_date:
                return True

        return False
