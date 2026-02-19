"""Legal compliance: ensures responses include required disclaimers and PII protection.

# DP-100: Responsible AI - Legal compliance ensures responses meet regulatory
requirements and protect user privacy (PII detection).
"""

import re
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from rag.models import DocumentCategory, RAGResponse


class ComplianceResult(BaseModel):
    """Result of legal compliance check."""

    compliant: bool = Field(default=True)
    issues: List[str] = Field(default_factory=list)
    disclaimers_added: List[str] = Field(default_factory=list)
    pii_detected: bool = Field(default=False)


class LegalCompliance:
    """
    Ensures RAG responses meet legal requirements and protect PII.

    # DP-100: Responsible AI - Legal compliance ensures responses include
    required disclaimers and do not expose personally identifiable information.
    """

    def __init__(self):
        """Initialize legal compliance checker."""
        # PII detection patterns
        # DP-100: Responsible AI - PII detection prevents exposure of
        # sensitive personal information in responses.
        self.pii_patterns = {
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN: 123-45-6789
            "ssn_no_dash": re.compile(r"\b\d{9}\b"),  # SSN without dashes
            "account_number": re.compile(r"\b\d{10,}\b"),  # Account numbers (10+ digits)
            "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Credit card
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # Phone number
        }

        # Category-specific disclaimers
        self.category_disclaimers = {
            DocumentCategory.PROPERTY: "Property data is for informational purposes only. Verify with county assessor before making decisions.",
            DocumentCategory.LEGAL: "This is not legal advice. Consult a licensed attorney for legal matters.",
            DocumentCategory.TAX: "Tax information may change. Verify current rates with the county tax office.",
            DocumentCategory.DEMOGRAPHICS: "Demographic data sourced from public census records.",
            DocumentCategory.ZONING: "Zoning information may change. Verify current regulations with the county planning department.",
            DocumentCategory.PERMITS: "Permit information is for reference only. Verify current requirements with the county permitting office.",
            DocumentCategory.COURTS: "Court records are public information. This is not legal advice.",
        }

    def check(self, response: RAGResponse) -> ComplianceResult:
        """
        Check response for legal compliance and PII.

        # DP-100: Responsible AI - Compliance checking ensures responses
        meet legal requirements and protect user privacy.

        Args:
            response: RAG response to check.

        Returns:
            ComplianceResult with compliance status and issues.
        """
        issues: List[str] = []
        disclaimers_added: List[str] = []
        pii_detected = False

        # Check for PII in answer
        answer_text = response.answer
        detected_pii_types: List[str] = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(answer_text)
            if matches:
                detected_pii_types.append(pii_type)
                pii_detected = True
                logger.error("PII detected in response: {} (type: {})", matches[:3], pii_type)
                issues.append(f"PII detected: {pii_type}")

        # Check for PII in source excerpts
        for source in response.sources:
            for pii_type, pattern in self.pii_patterns.items():
                if pattern.search(source.excerpt):
                    detected_pii_types.append(f"{pii_type}_in_source")
                    pii_detected = True
                    logger.error("PII detected in source excerpt: {}", pii_type)
                    issues.append(f"PII in source: {pii_type}")

        # Add category-specific disclaimer if category is set
        if response.category and response.category in self.category_disclaimers:
            disclaimer = self.category_disclaimers[response.category]
            # Check if disclaimer already present
            if disclaimer.lower() not in answer_text.lower():
                disclaimers_added.append(disclaimer)

        return ComplianceResult(
            compliant=not pii_detected and len(issues) == 0,
            issues=issues,
            disclaimers_added=disclaimers_added,
            pii_detected=pii_detected,
        )

    def add_disclaimers(self, answer: str, category: Optional[DocumentCategory]) -> str:
        """
        Add category-specific disclaimers to answer.

        # DP-100: Responsible AI - Disclaimer addition ensures users
        understand limitations and legal status of information.

        Args:
            answer: Original answer text.
            category: Document category.

        Returns:
            Answer with disclaimers appended.
        """
        if category is None:
            return answer

        if category not in self.category_disclaimers:
            return answer

        disclaimer = self.category_disclaimers[category]

        # Check if disclaimer already present
        if disclaimer.lower() in answer.lower():
            return answer

        # Append disclaimer
        return f"{answer}\n\n*Disclaimer: {disclaimer}*"
