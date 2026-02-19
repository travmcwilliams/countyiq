"""Bias detection: flags potentially biased or discriminatory language.

# DP-100: Responsible AI - Fairness and bias detection ensures responses
do not perpetuate stereotypes or exclude demographic groups.
"""

import re
from typing import List, Optional, Set

from loguru import logger
from pydantic import BaseModel, Field

from pipelines.ingest.county_registry import get_counties_by_state
from rag.models import RAGResponse


class BiasReport(BaseModel):
    """Report of bias detection results."""

    bias_detected: bool = Field(default=False)
    bias_type: Optional[str] = Field(default=None, description="Type of bias detected")
    flagged_terms: List[str] = Field(default_factory=list)
    severity: str = Field(default="low", description="low, medium, or high")
    recommendation: Optional[str] = Field(default=None)


class BalanceReport(BaseModel):
    """Report on demographic balance in data coverage."""

    fips: str = Field(..., min_length=5, max_length=5)
    categories_covered: List[str] = Field(default_factory=list)
    missing_categories: List[str] = Field(default_factory=list)
    balance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="0.0 = no balance, 1.0 = perfect balance")


class BiasDetector:
    """
    Detects potentially biased language and demographic imbalances.

    # DP-100: Responsible AI - Bias detection prevents discriminatory
    language and ensures fair representation across demographic groups.
    """

    def __init__(self):
        """Initialize bias detector with flagged terms."""
        # Terms that may indicate loaded language or stereotypes
        # DP-100: Responsible AI - Flagged terms help identify potentially
        # biased language that should trigger human review.
        self.flagged_terms: Set[str] = {
            # Race/ethnicity loaded terms
            "inner city",
            "urban crime",
            "ghetto",
            "thug",
            "illegal alien",
            "anchor baby",
            # Income/class loaded terms
            "welfare queen",
            "deadbeat",
            "lazy",
            "entitled",
            # Crime stereotypes
            "criminal element",
            "gang-infested",
            "high-crime area",
            # Geographic stereotypes
            "bad neighborhood",
            "rough area",
            "sketchy",
        }

        # Demographic categories to check for balance
        self.demographic_categories = [
            "race",
            "ethnicity",
            "income",
            "age",
            "education",
            "housing",
        ]

    def check(self, response: RAGResponse) -> BiasReport:
        """
        Check response for potentially biased language.

        # DP-100: Responsible AI - Bias checking ensures responses
        do not contain discriminatory or stereotypical language.

        Args:
            response: RAG response to check.

        Returns:
            BiasReport with detection results.
        """
        answer_lower = response.answer.lower()
        flagged: List[str] = []
        bias_types: List[str] = []

        # Check for flagged terms
        for term in self.flagged_terms:
            # Use word boundaries to avoid false positives
            pattern = r"\b" + re.escape(term.lower()) + r"\b"
            if re.search(pattern, answer_lower):
                flagged.append(term)
                logger.warning("Flagged term detected in response: {}", term)

        # Check for demographic generalizations
        generalization_patterns = [
            r"all\s+\w+\s+people",
            r"all\s+people",
            r"every\s+\w+\s+person",
            r"every\s+person",
            r"\w+\s+are\s+always",
            r"\w+\s+never\s+\w+",
        ]
        for pattern in generalization_patterns:
            if re.search(pattern, answer_lower):
                bias_types.append("demographic_generalization")
                logger.warning("Demographic generalization detected in response")
                break  # Only need to detect once

        # Determine severity
        severity = "low"
        if len(flagged) > 2 or (len(bias_types) > 0 and len(flagged) > 0):
            # High: multiple flagged terms OR generalization + flagged term
            severity = "high"
        elif len(flagged) > 0 or len(bias_types) > 0:
            severity = "medium"

        # Build recommendation
        recommendation = None
        if bias_detected := len(flagged) > 0 or len(bias_types) > 0:
            recommendation = (
                "This response contains language that may be biased or discriminatory. "
                "Please review and ensure fair, accurate representation of all demographic groups."
            )

        return BiasReport(
            bias_detected=bias_detected,
            bias_type=bias_types[0] if bias_types else None,
            flagged_terms=flagged,
            severity=severity,
            recommendation=recommendation,
        )

    def check_demographic_balance(self, fips: str) -> BalanceReport:
        """
        Check if query results systematically exclude certain demographic groups.

        # DP-100: Responsible AI - Demographic balance checking ensures
        fair representation across all groups in data coverage.

        Args:
            fips: County FIPS code to check.

        Returns:
            BalanceReport with coverage analysis.
        """
        # In production, this would query the actual data store to check
        # which demographic categories are represented in retrieved documents.
        # For now, we simulate by checking if sources mention demographic categories.

        # Placeholder: In real implementation, would query vector store
        # for documents mentioning each demographic category
        categories_covered: List[str] = []
        missing_categories: List[str] = []

        # Simulate checking coverage (in production, query actual data)
        # This is a simplified version - real implementation would check
        # document content for demographic category mentions
        for category in self.demographic_categories:
            # Placeholder logic - in production, check actual document content
            categories_covered.append(category)  # Simplified: assume all covered

        # Calculate balance score
        total_categories = len(self.demographic_categories)
        if total_categories == 0:
            balance_score = 1.0
        else:
            balance_score = len(categories_covered) / total_categories

        return BalanceReport(
            fips=fips,
            categories_covered=categories_covered,
            missing_categories=missing_categories,
            balance_score=balance_score,
        )
