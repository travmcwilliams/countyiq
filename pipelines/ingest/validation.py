"""
Data validation framework for CountyIQ documents.
Validates content, FIPS, category, URLs, metadata, and flags PII (SSN, credit card, email).
# DP-100: Data quality and validation - Quality gates before indexing and model consumption.
"""

import re
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, Field

from data.schemas.document import CountyDocument, DocumentCategory

# DP-100: Data quality - Validation gates before indexing and model consumption

MIN_CONTENT_LENGTH = 50
FIPS_PATTERN = re.compile(r"^\d{5}$")
# SSN: xxx-xx-xxxx (with optional spaces)
SSN_PATTERN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
# Simple credit card: 4 groups of 4 digits, optional spaces/dashes
CC_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
# Email (basic)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
# URL (basic)
URL_PATTERN = re.compile(
    r"^https?://[^\s/$.?#].[^\s]*$",
    re.IGNORECASE,
)


class ValidationResult(BaseModel):
    """Result of validating a single CountyDocument."""

    is_valid: bool = Field(..., description="True if document passed all validation rules")
    errors: list[str] = Field(default_factory=list, description="Blocking validation errors")
    warnings: list[str] = Field(default_factory=list, description="Non-blocking warnings (e.g. PII)")
    doc_id: UUID = Field(..., description="Document id")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Quality score 0.0-1.0")


class BatchValidationResult(BaseModel):
    """Result of validating a batch of CountyDocuments."""

    total: int = Field(..., ge=0)
    valid: int = Field(..., ge=0)
    invalid: int = Field(..., ge=0)
    results: list[ValidationResult] = Field(default_factory=list)


def _content_for_validation(doc: CountyDocument) -> str:
    """Text to validate: processed_content if set, else raw_content."""
    if doc.processed_content and doc.processed_content.strip():
        return doc.processed_content.strip()
    return (doc.raw_content or "").strip()


def _quality_score(doc: CountyDocument) -> float:
    """
    Compute quality score 0.0-1.0 from:
    content length, metadata richness, has processed_content, has source_url.
    """
    content = _content_for_validation(doc)
    score = 0.0
    # Content length: up to 0.4 (e.g. 500+ chars = 0.4)
    length_score = min(1.0, len(content) / 500.0) * 0.4
    score += length_score
    # Has processed_content: 0.2
    if doc.processed_content and doc.processed_content.strip():
        score += 0.2
    # Has source_url (non-empty): 0.2
    if doc.source_url and doc.source_url.strip():
        score += 0.2
    # Metadata richness: 0.2 (e.g. 5+ keys = 0.2)
    meta_keys = len(doc.metadata) if isinstance(doc.metadata, dict) else 0
    score += min(1.0, meta_keys / 5.0) * 0.2
    return round(min(1.0, score), 2)


def _check_pii(content: str) -> list[str]:
    """Flag SSN, credit card, and email patterns; return list of warning messages."""
    warnings: list[str] = []
    if SSN_PATTERN.search(content):
        warnings.append("Possible SSN pattern detected in content")
    if CC_PATTERN.search(content):
        warnings.append("Possible credit card pattern detected in content")
    if EMAIL_PATTERN.search(content):
        warnings.append("Email address(es) detected in content")
    return warnings


def _is_valid_url(s: str | None) -> bool:
    """Return True if s is empty or a valid HTTP/HTTPS URL."""
    if not s or not str(s).strip():
        return True
    return bool(URL_PATTERN.match(str(s).strip()))


class DocumentValidator:
    """
    Validates CountyDocuments: content, FIPS, category, source_url, metadata, PII.
    PII detection flags but does not block.
    """

    def __init__(self, min_content_length: int = MIN_CONTENT_LENGTH) -> None:
        self.min_content_length = min_content_length

    def validate(self, doc: CountyDocument) -> ValidationResult:
        """
        Validate a single document.
        Checks: content not empty, length >= min, fips 5-digit, category valid,
        source_url valid or empty, metadata dict, PII (warnings only).
        """
        errors: list[str] = []
        warnings: list[str] = []

        content = _content_for_validation(doc)

        if not content:
            errors.append("Content is empty")
        elif len(content) < self.min_content_length:
            errors.append(f"Content length {len(content)} is below minimum {self.min_content_length}")

        if not FIPS_PATTERN.match(doc.fips):
            errors.append("fips must be a 5-digit string")

        try:
            DocumentCategory(doc.category)
        except (ValueError, TypeError):
            errors.append("category is not a valid DocumentCategory")

        if not _is_valid_url(doc.source_url):
            errors.append("source_url must be a valid URL or empty")

        if not isinstance(doc.metadata, dict):
            errors.append("metadata must be a dict")

        warnings.extend(_check_pii(content))

        is_valid = len(errors) == 0
        quality = _quality_score(doc)

        if not is_valid:
            logger.debug("Validation failed for doc {}: {}", doc.id, errors)

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            doc_id=doc.id,
            quality_score=quality,
        )

    def validate_batch(self, docs: list[CountyDocument]) -> BatchValidationResult:
        """Validate a list of documents and return aggregate summary."""
        results = [self.validate(doc) for doc in docs]
        valid_count = sum(1 for r in results if r.is_valid)
        return BatchValidationResult(
            total=len(docs),
            valid=valid_count,
            invalid=len(docs) - valid_count,
            results=results,
        )
