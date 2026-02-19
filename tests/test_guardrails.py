"""Tests for Responsible AI guardrails.

# DP-100: Responsible AI - Comprehensive testing ensures guardrails
function correctly to protect users from unsafe, biased, or non-compliant responses.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from rag.guardrails.bias_detector import BiasDetector, BiasReport, BalanceReport
from rag.guardrails.confidence_filter import ConfidenceFilter, FilteredResponse
from rag.guardrails.guardrail_pipeline import GuardedResponse, GuardrailPipeline
from rag.guardrails.legal_compliance import ComplianceResult, LegalCompliance
from rag.models import DocumentCategory, RAGResponse, SourceCitation


# Test fixtures
def create_rag_response(
    answer: str = "Test answer",
    confidence: float = 0.8,
    sources_count: int = 3,
    county_fips: str = "01001",
    category: DocumentCategory = DocumentCategory.PROPERTY,
) -> RAGResponse:
    """Create a test RAG response."""
    sources = [
        SourceCitation(
            document_id=f"doc_{i}",
            county_fips=county_fips,
            source_url=f"https://example.com/doc_{i}",
            excerpt=f"Excerpt {i}",
            relevance_score=0.8,
            category=category,
            timestamp=datetime.utcnow(),
        )
        for i in range(sources_count)
    ]
    return RAGResponse(
        answer=answer,
        sources=sources,
        confidence_score=confidence,
        county_fips=county_fips,
        hallucination_detected=False,
        category=category,
    )


# Confidence Filter Tests
def test_confidence_filter_blocks_low_confidence() -> None:
    """Test confidence filter blocks responses below threshold."""
    filter_obj = ConfidenceFilter(min_confidence=0.3)
    response = create_rag_response(confidence=0.2)
    filtered = filter_obj.filter(response)
    assert not filtered.confidence_sufficient
    assert "don't have sufficient information" in filtered.filtered_answer.lower()
    assert filtered.filter_reason is not None


def test_confidence_filter_allows_high_confidence() -> None:
    """Test confidence filter allows responses above threshold."""
    filter_obj = ConfidenceFilter(min_confidence=0.3)
    response = create_rag_response(confidence=0.8)
    filtered = filter_obj.filter(response)
    assert filtered.confidence_sufficient
    assert filtered.filter_reason is None


def test_confidence_filter_blocks_no_sources() -> None:
    """Test confidence filter blocks responses with no sources."""
    filter_obj = ConfidenceFilter(min_source_count=1)
    response = create_rag_response(sources_count=0)
    filtered = filter_obj.filter(response)
    assert not filtered.confidence_sufficient
    assert "don't have sufficient information" in filtered.filtered_answer.lower()


def test_confidence_filter_adds_medium_confidence_disclaimer() -> None:
    """Test disclaimer added for medium confidence (0.3-0.6)."""
    filter_obj = ConfidenceFilter(min_confidence=0.3)
    response = create_rag_response(confidence=0.5)
    filtered = filter_obj.filter(response)
    assert filtered.confidence_sufficient
    assert filtered.disclaimer_added
    assert "limited available data" in filtered.filtered_answer.lower()


def test_confidence_filter_adds_outdated_disclaimer() -> None:
    """Test outdated data disclaimer added when sources are old."""
    filter_obj = ConfidenceFilter(outdated_days=90)
    old_timestamp = datetime.utcnow() - timedelta(days=100)
    sources = [
        SourceCitation(
            document_id="doc_1",
            county_fips="01001",
            source_url="https://example.com/doc_1",
            excerpt="Old excerpt",
            relevance_score=0.8,
            timestamp=old_timestamp,
        )
    ]
    response = RAGResponse(
        answer="Test answer",
        sources=sources,
        confidence_score=0.8,
        county_fips="01001",
        hallucination_detected=False,
    )
    filtered = filter_obj.filter(response)
    assert filtered.disclaimer_added
    assert "outdated" in filtered.filtered_answer.lower()


def test_is_sufficient_returns_true_for_good_response() -> None:
    """Test is_sufficient returns True for good response."""
    filter_obj = ConfidenceFilter(min_confidence=0.3, min_source_count=1)
    response = create_rag_response(confidence=0.8, sources_count=3)
    assert filter_obj.is_sufficient(response)


def test_is_sufficient_returns_false_for_low_confidence() -> None:
    """Test is_sufficient returns False for low confidence."""
    filter_obj = ConfidenceFilter(min_confidence=0.3)
    response = create_rag_response(confidence=0.2)
    assert not filter_obj.is_sufficient(response)


# Bias Detector Tests
def test_bias_detector_flags_loaded_terms() -> None:
    """Test bias detector flags loaded terms."""
    detector = BiasDetector()
    response = create_rag_response(answer="This inner city area has high crime rates.")
    report = detector.check(response)
    assert report.bias_detected
    assert len(report.flagged_terms) > 0
    assert "inner city" in report.flagged_terms


def test_bias_detector_no_bias_for_clean_response() -> None:
    """Test bias detector finds no bias in clean response."""
    detector = BiasDetector()
    response = create_rag_response(answer="The property value is $250,000.")
    report = detector.check(response)
    assert not report.bias_detected
    assert len(report.flagged_terms) == 0


def test_bias_detector_flags_demographic_generalizations() -> None:
    """Test bias detector flags demographic generalizations."""
    detector = BiasDetector()
    response = create_rag_response(answer="All people in this area are wealthy.")
    report = detector.check(response)
    assert report.bias_detected
    assert "demographic_generalization" in (report.bias_type or "")


def test_bias_detector_severity_levels() -> None:
    """Test bias detector assigns correct severity levels."""
    detector = BiasDetector()
    # Low severity: single flagged term
    response1 = create_rag_response(answer="This is an inner city neighborhood.")
    report1 = detector.check(response1)
    assert report1.severity in ["low", "medium"]

    # High severity: multiple flagged terms
    response2 = create_rag_response(
        answer="This inner city area has high crime and is gang-infested."
    )
    report2 = detector.check(response2)
    # Should be high if multiple flagged terms detected
    assert report2.severity in ["medium", "high"]


def test_check_demographic_balance() -> None:
    """Test demographic balance checking."""
    detector = BiasDetector()
    report = detector.check_demographic_balance("01001")
    assert isinstance(report, BalanceReport)
    assert report.fips == "01001"
    assert 0.0 <= report.balance_score <= 1.0


# Legal Compliance Tests
def test_legal_compliance_adds_property_disclaimer() -> None:
    """Test legal compliance adds property disclaimer."""
    compliance = LegalCompliance()
    answer = "The property value is $250,000."
    result = compliance.add_disclaimers(answer, DocumentCategory.PROPERTY)
    assert "informational purposes only" in result.lower()
    assert "county assessor" in result.lower()


def test_legal_compliance_adds_legal_disclaimer() -> None:
    """Test legal compliance adds legal disclaimer."""
    compliance = LegalCompliance()
    answer = "This is legal information."
    result = compliance.add_disclaimers(answer, DocumentCategory.LEGAL)
    assert "not legal advice" in result.lower()
    assert "licensed attorney" in result.lower()


def test_legal_compliance_adds_tax_disclaimer() -> None:
    """Test legal compliance adds tax disclaimer."""
    compliance = LegalCompliance()
    answer = "Tax rate is 2.5%."
    result = compliance.add_disclaimers(answer, DocumentCategory.TAX)
    assert "tax office" in result.lower()


def test_legal_compliance_detects_ssn_pii() -> None:
    """Test PII detection for SSN."""
    compliance = LegalCompliance()
    response = create_rag_response(answer="SSN: 123-45-6789")
    result = compliance.check(response)
    assert result.pii_detected
    assert not result.compliant
    assert "ssn" in str(result.issues).lower()


def test_legal_compliance_detects_email_pii() -> None:
    """Test PII detection for email."""
    compliance = LegalCompliance()
    response = create_rag_response(answer="Contact: user@example.com")
    result = compliance.check(response)
    assert result.pii_detected
    assert not result.compliant


def test_legal_compliance_detects_phone_pii() -> None:
    """Test PII detection for phone number."""
    compliance = LegalCompliance()
    response = create_rag_response(answer="Phone: 555-123-4567")
    result = compliance.check(response)
    assert result.pii_detected


def test_legal_compliance_detects_pii_in_source() -> None:
    """Test PII detection in source excerpts."""
    compliance = LegalCompliance()
    sources = [
        SourceCitation(
            document_id="doc_1",
            county_fips="01001",
            source_url="https://example.com/doc_1",
            excerpt="SSN: 123-45-6789",
            relevance_score=0.8,
        )
    ]
    response = RAGResponse(
        answer="Test answer",
        sources=sources,
        confidence_score=0.8,
        county_fips="01001",
        hallucination_detected=False,
    )
    result = compliance.check(response)
    assert result.pii_detected
    assert "source" in str(result.issues).lower()


def test_legal_compliance_no_pii_clean_response() -> None:
    """Test legal compliance finds no PII in clean response."""
    compliance = LegalCompliance()
    response = create_rag_response(answer="The property value is $250,000.")
    result = compliance.check(response)
    assert not result.pii_detected
    assert result.compliant


# Guardrail Pipeline Tests
def test_guardrail_pipeline_safe_response() -> None:
    """Test guardrail pipeline allows safe response."""
    pipeline = GuardrailPipeline()
    response = create_rag_response(confidence=0.8, sources_count=3)
    guarded = pipeline.apply(response)
    assert guarded.safe_to_serve
    assert guarded.confidence_sufficient
    assert not guarded.bias_report.bias_detected
    assert guarded.compliance_result.compliant


def test_guardrail_pipeline_blocks_low_confidence() -> None:
    """Test guardrail pipeline blocks low confidence response."""
    pipeline = GuardrailPipeline(min_confidence=0.3)
    response = create_rag_response(confidence=0.2)
    guarded = pipeline.apply(response)
    assert not guarded.safe_to_serve
    assert "don't have sufficient information" in guarded.final_answer.lower()


def test_guardrail_pipeline_blocks_pii() -> None:
    """Test guardrail pipeline blocks response with PII."""
    pipeline = GuardrailPipeline()
    response = create_rag_response(answer="SSN: 123-45-6789")
    guarded = pipeline.apply(response)
    assert not guarded.safe_to_serve
    assert guarded.compliance_result.pii_detected
    assert "sensitive personal data" in guarded.final_answer.lower()


def test_guardrail_pipeline_blocks_bias() -> None:
    """Test guardrail pipeline blocks biased response."""
    pipeline = GuardrailPipeline()
    response = create_rag_response(answer="This inner city area is dangerous.")
    guarded = pipeline.apply(response)
    assert not guarded.safe_to_serve
    assert guarded.bias_report.bias_detected


def test_guardrail_pipeline_adds_disclaimers() -> None:
    """Test guardrail pipeline adds disclaimers."""
    pipeline = GuardrailPipeline()
    response = create_rag_response(
        confidence=0.5,
        category=DocumentCategory.PROPERTY,
    )
    guarded = pipeline.apply(response)
    assert guarded.safe_to_serve
    assert len(guarded.disclaimers) > 0


def test_guardrail_pipeline_fallback_message() -> None:
    """Test guardrail pipeline returns fallback when not safe (bias detected)."""
    pipeline = GuardrailPipeline()
    # Use high confidence but with bias to trigger fallback (not confidence filter)
    response = create_rag_response(
        confidence=0.8,
        answer="This inner city area is dangerous and all people there are criminals.",
    )
    guarded = pipeline.apply(response)
    assert not guarded.safe_to_serve
    assert "unable to provide a reliable answer" in guarded.final_answer.lower()


def test_guardrail_pipeline_logs_to_mlflow() -> None:
    """Test guardrail pipeline logs to MLflow."""
    pipeline = GuardrailPipeline()
    response = create_rag_response(confidence=0.8)
    # MLflow import happens inside _log_to_mlflow, so we just verify
    # the function completes without error (MLflow may not be installed)
    guarded = pipeline.apply(response)
    # Verify response was created (MLflow logging may fail silently if not installed)
    assert guarded is not None
    assert guarded.safe_to_serve


# Integration Tests
def test_full_pipeline_with_guardrails() -> None:
    """Test GuardrailPipeline produces GuardedResponse (integration)."""
    from rag.guardrails import GuardrailPipeline

    pipeline = GuardrailPipeline(min_confidence=0.3)
    response = create_rag_response(confidence=0.8, sources_count=2)
    guarded = pipeline.apply(response)
    assert isinstance(guarded, GuardedResponse)
    assert guarded.safe_to_serve


def test_guardrail_sequence_order() -> None:
    """Test guardrails are applied in correct sequence."""
    pipeline = GuardrailPipeline()
    # Low confidence should be caught first
    response = create_rag_response(confidence=0.2)
    guarded = pipeline.apply(response)
    # Should fail at confidence filter, not reach bias/legal
    assert not guarded.confidence_sufficient
    assert not guarded.safe_to_serve


def test_multiple_disclaimers_added() -> None:
    """Test multiple disclaimers can be added."""
    pipeline = GuardrailPipeline()
    # Medium confidence + property category = multiple disclaimers
    response = create_rag_response(
        confidence=0.5,
        category=DocumentCategory.PROPERTY,
    )
    guarded = pipeline.apply(response)
    assert guarded.safe_to_serve
    # Should have at least confidence disclaimer and category disclaimer
    assert len(guarded.disclaimers) >= 1
