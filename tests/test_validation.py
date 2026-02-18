"""
Tests for document validation framework.
Covers empty/short content, FIPS, category, source_url, PII detection, quality score, batch validation.
"""

import pytest

from data.schemas.document import CountyDocument, ContentType, DocumentCategory
from pipelines.ingest.validation import (
    DocumentValidator,
    ValidationResult,
    BatchValidationResult,
    _check_pii,
    _content_for_validation,
    _quality_score,
    _is_valid_url,
    MIN_CONTENT_LENGTH,
)


def _doc(
    raw_content: str = "A" * 60,
    processed_content: str | None = None,
    fips: str = "01001",
    county_name: str = "Autauga",
    state_abbr: str = "AL",
    category: DocumentCategory = DocumentCategory.property,
    source_url: str | None = "https://example.com/doc",
    metadata: dict | None = None,
) -> CountyDocument:
    """Build a CountyDocument for tests. Use processed_content=None for default (raw strip)."""
    use_processed = raw_content.strip() if processed_content is None else processed_content
    return CountyDocument(
        fips=fips,
        county_name=county_name,
        state_abbr=state_abbr,
        category=category,
        source_url=source_url,
        content_type=ContentType.text,
        raw_content=raw_content,
        processed_content=use_processed or None,
        metadata=metadata or {},
    )


# ---- _content_for_validation ----


class TestContentForValidation:
    """Test content selection for validation."""

    def test_uses_processed_content_when_present(self) -> None:
        doc = _doc(raw_content="x" * 60, processed_content="processed text here")
        assert _content_for_validation(doc) == "processed text here"

    def test_uses_raw_content_when_processed_empty(self) -> None:
        doc = _doc(raw_content="raw only content here", processed_content="")
        assert "raw only" in _content_for_validation(doc)


# ---- _quality_score ----


class TestQualityScore:
    """Test quality score calculation (0.0-1.0)."""

    def test_long_content_and_metadata_high_score(self) -> None:
        doc = _doc(
            raw_content="x" * 600,
            processed_content="y" * 600,
            source_url="https://example.com/p",
            metadata={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        )
        s = _quality_score(doc)
        assert 0.8 <= s <= 1.0

    def test_short_content_low_score(self) -> None:
        doc = _doc(raw_content="short", processed_content=None, source_url=None, metadata={})
        s = _quality_score(doc)
        assert s < 0.5

    def test_has_processed_content_increases_score(self) -> None:
        doc_empty_processed = _doc(raw_content="x" * 100, processed_content="", metadata={})
        doc_with_processed = _doc(raw_content="x" * 100, processed_content="y" * 100, metadata={})
        assert _quality_score(doc_with_processed) > _quality_score(doc_empty_processed)


# ---- _check_pii ----


class TestCheckPII:
    """Test PII detection (flag only, no block)."""

    def test_ssn_pattern_flagged(self) -> None:
        w = _check_pii("Contact SSN 123-45-6789 for verification.")
        assert any("SSN" in x for x in w)

    def test_ssn_no_dashes_flagged(self) -> None:
        w = _check_pii("SSN: 123456789")
        assert any("SSN" in x for x in w)

    def test_credit_card_pattern_flagged(self) -> None:
        w = _check_pii("Card 1234-5678-9012-3456")
        assert any("credit" in x.lower() for x in w)

    def test_email_flagged(self) -> None:
        w = _check_pii("Email user@example.com for help.")
        assert any("Email" in x for x in w)

    def test_no_pii_no_warnings(self) -> None:
        w = _check_pii("Just normal county property text.")
        assert len(w) == 0


# ---- _is_valid_url ----


class TestIsValidUrl:
    """Test URL validation."""

    def test_empty_allowed(self) -> None:
        assert _is_valid_url("") is True
        assert _is_valid_url(None) is True

    def test_http_url_valid(self) -> None:
        assert _is_valid_url("http://example.com/path") is True

    def test_https_url_valid(self) -> None:
        assert _is_valid_url("https://county.gov/data") is True

    def test_invalid_not_url(self) -> None:
        assert _is_valid_url("not a url") is False


# ---- DocumentValidator.validate ----


class TestDocumentValidatorValidate:
    """Test single-document validation."""

    def test_empty_content_fails(self) -> None:
        v = DocumentValidator()
        doc = _doc(raw_content="", processed_content="")
        r = v.validate(doc)
        assert r.is_valid is False
        assert any("empty" in e.lower() for e in r.errors)

    def test_short_content_fails(self) -> None:
        v = DocumentValidator()
        doc = _doc(raw_content="short", processed_content="short")
        r = v.validate(doc)
        assert r.is_valid is False
        assert any("length" in e.lower() or "minimum" in e.lower() for e in r.errors)

    def test_content_above_min_passes_length_check(self) -> None:
        v = DocumentValidator()
        doc = _doc(raw_content="A" * (MIN_CONTENT_LENGTH + 1), processed_content="A" * (MIN_CONTENT_LENGTH + 1))
        # May still fail on other checks; at least no "content length" error
        r = v.validate(doc)
        content_errors = [e for e in r.errors if "length" in e.lower() or "empty" in e.lower()]
        assert len(content_errors) == 0

    def test_invalid_fips_fails(self) -> None:
        v = DocumentValidator()
        doc = _doc(fips="12ab5", raw_content="x" * 60)  # non-digit
        r = v.validate(doc)
        assert r.is_valid is False
        assert any("fips" in e.lower() for e in r.errors)

    def test_valid_fips_passes(self) -> None:
        v = DocumentValidator()
        doc = _doc(fips="01001", raw_content="x" * 60)
        r = v.validate(doc)
        assert not any("fips" in e.lower() for e in r.errors)

    def test_fips_00000_valid_format(self) -> None:
        v = DocumentValidator()
        doc = _doc(fips="00000", raw_content="x" * 60)
        r = v.validate(doc)
        assert not any("fips" in e.lower() for e in r.errors)

    def test_invalid_source_url_fails(self) -> None:
        v = DocumentValidator()
        doc = _doc(raw_content="x" * 60, source_url="not a url")
        r = v.validate(doc)
        assert r.is_valid is False
        assert any("url" in e.lower() for e in r.errors)

    def test_empty_source_url_passes(self) -> None:
        v = DocumentValidator()
        doc = _doc(raw_content="x" * 60, source_url="")
        r = v.validate(doc)
        assert not any("url" in e.lower() for e in r.errors)

    def test_pii_detection_adds_warning_does_not_fail(self) -> None:
        v = DocumentValidator()
        doc = _doc(raw_content="x" * 60 + " SSN 123-45-6789")
        r = v.validate(doc)
        assert any("SSN" in w for w in r.warnings)
        # Still valid (PII does not block)
        assert r.is_valid is True

    def test_result_has_doc_id_and_quality_score(self) -> None:
        v = DocumentValidator()
        doc = _doc(raw_content="x" * 60)
        r = v.validate(doc)
        assert r.doc_id == doc.id
        assert 0.0 <= r.quality_score <= 1.0


# ---- DocumentValidator.validate_batch ----


class TestDocumentValidatorBatch:
    """Test batch validation summary counts."""

    def test_batch_summary_counts(self) -> None:
        v = DocumentValidator()
        valid_doc = _doc(raw_content="x" * 60)
        invalid_doc = _doc(raw_content="short")  # too short
        batch = v.validate_batch([valid_doc, invalid_doc])
        assert batch.total == 2
        assert batch.valid == 1
        assert batch.invalid == 1
        assert len(batch.results) == 2

    def test_batch_all_valid(self) -> None:
        v = DocumentValidator()
        docs = [_doc(raw_content="x" * 60) for _ in range(3)]
        batch = v.validate_batch(docs)
        assert batch.total == 3
        assert batch.valid == 3
        assert batch.invalid == 0

    def test_batch_all_invalid(self) -> None:
        v = DocumentValidator()
        docs = [_doc(raw_content="x") for _ in range(2)]
        batch = v.validate_batch(docs)
        assert batch.total == 2
        assert batch.valid == 0
        assert batch.invalid == 2

    def test_batch_empty_list(self) -> None:
        v = DocumentValidator()
        batch = v.validate_batch([])
        assert batch.total == 0
        assert batch.valid == 0
        assert batch.invalid == 0
        assert batch.results == []


# ---- ValidationResult / BatchValidationResult models ----


class TestValidationModels:
    """Test Pydantic validation result models."""

    def test_validation_result_serialization(self) -> None:
        from uuid import uuid4

        r = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Possible SSN"],
            doc_id=uuid4(),
            quality_score=0.75,
        )
        assert r.is_valid is True
        assert r.quality_score == 0.75
        assert len(r.warnings) == 1

    def test_batch_validation_result_serialization(self) -> None:
        b = BatchValidationResult(total=5, valid=4, invalid=1, results=[])
        assert b.total == 5
        assert b.valid == 4
        assert b.invalid == 1
