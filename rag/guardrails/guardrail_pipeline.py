"""Guardrail pipeline: orchestrates all responsible AI checks.

# DP-100: Responsible AI - Pipeline orchestrates confidence filtering,
bias detection, and legal compliance to ensure safe, compliant responses.
"""

from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from rag.guardrails.bias_detector import BiasDetector, BiasReport
from rag.guardrails.confidence_filter import ConfidenceFilter, FilteredResponse
from rag.guardrails.legal_compliance import ComplianceResult, LegalCompliance
from rag.models import RAGResponse


class GuardedResponse(BaseModel):
    """Final response after all guardrails applied.

    # DP-100: Responsible AI - GuardedResponse ensures all safety checks
    have been applied before serving to users.
    """

    original_response: RAGResponse
    final_answer: str
    confidence_sufficient: bool = Field(default=False)
    bias_report: BiasReport
    compliance_result: ComplianceResult
    disclaimers: List[str] = Field(default_factory=list)
    safe_to_serve: bool = Field(default=False)


class GuardrailPipeline:
    """
    Orchestrates all responsible AI guardrails in sequence.

    # DP-100: Responsible AI - Pipeline ensures responses pass confidence,
    bias, and legal compliance checks before being served to users.
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
        min_source_count: int = 1,
        outdated_days: int = 90,
    ):
        """
        Initialize guardrail pipeline.

        Args:
            min_confidence: Minimum confidence threshold.
            min_source_count: Minimum source count required.
            outdated_days: Days after which data is considered outdated.
        """
        self.confidence_filter = ConfidenceFilter(
            min_confidence=min_confidence,
            min_source_count=min_source_count,
            outdated_days=outdated_days,
        )
        self.bias_detector = BiasDetector()
        self.legal_compliance = LegalCompliance()

    def apply(self, response: RAGResponse) -> GuardedResponse:
        """
        Apply all guardrails to response in sequence.

        # DP-100: Responsible AI - Sequential guardrail application ensures
        comprehensive safety checks: confidence → bias → legal compliance.

        Args:
            response: Original RAG response.

        Returns:
            GuardedResponse with all guardrails applied.
        """
        logger.info("Applying guardrails to response for county {}", response.county_fips)

        # Step 1: Confidence filtering
        filtered = self.confidence_filter.filter(response)
        if not filtered.confidence_sufficient:
            logger.warning("Response failed confidence filter: {}", filtered.filter_reason)
            return self._create_unsafe_response(
                response=response,
                final_answer=filtered.filtered_answer,
                confidence_sufficient=False,
            )

        # Step 2: Bias detection
        bias_report = self.bias_detector.check(response)
        if bias_report.bias_detected:
            logger.warning("Bias detected in response: {}", bias_report.bias_type)

        # Step 3: Legal compliance
        compliance_result = self.legal_compliance.check(response)
        if compliance_result.pii_detected:
            logger.error("PII detected in response - blocking")
            return self._create_unsafe_response(
                response=response,
                final_answer="I cannot provide this information as it may contain sensitive personal data. Please consult the county directly.",
                confidence_sufficient=filtered.confidence_sufficient,
                compliance_result=compliance_result,
            )

        # Step 4: Add legal disclaimers
        answer = filtered.filtered_answer
        disclaimers: List[str] = []

        # Extract confidence disclaimers (already in filtered_answer if added)
        if filtered.disclaimer_added:
            # Extract disclaimers from filtered answer (format: "*Note: {disclaimer}*")
            if "*Note:" in answer:
                note_parts = answer.split("*Note:")
                for part in note_parts[1:]:  # Skip first part before first Note
                    if "*" in part:
                        note_text = part.split("*")[0].strip()
                        if note_text:
                            disclaimers.append(note_text)

        # Add legal disclaimers
        if compliance_result.disclaimers_added:
            answer = self.legal_compliance.add_disclaimers(answer, response.category)
            disclaimers.extend(compliance_result.disclaimers_added)

        # Determine if safe to serve
        safe_to_serve = (
            filtered.confidence_sufficient
            and not bias_report.bias_detected
            and compliance_result.compliant
            and not compliance_result.pii_detected
        )

        # If not safe (but passed confidence filter), return fallback
        # Note: If confidence filter already failed, we return early above
        if not safe_to_serve:
            logger.warning("Response not safe to serve - returning fallback")
            return self._create_unsafe_response(
                response=response,
                final_answer=self._get_fallback_message(),
                confidence_sufficient=filtered.confidence_sufficient,
                bias_report=bias_report,
                compliance_result=compliance_result,
            )

        # Log to MLflow
        self._log_to_mlflow(response, bias_report, compliance_result, safe_to_serve)

        return GuardedResponse(
            original_response=response,
            final_answer=answer,
            confidence_sufficient=filtered.confidence_sufficient,
            bias_report=bias_report,
            compliance_result=compliance_result,
            disclaimers=disclaimers,
            safe_to_serve=safe_to_serve,
        )

    def _create_unsafe_response(
        self,
        response: RAGResponse,
        final_answer: str,
        confidence_sufficient: bool,
        bias_report: Optional[BiasReport] = None,
        compliance_result: Optional[ComplianceResult] = None,
    ) -> GuardedResponse:
        """Create unsafe response with fallback message."""
        if bias_report is None:
            bias_report = BiasReport()
        if compliance_result is None:
            compliance_result = ComplianceResult()

        self._log_to_mlflow(response, bias_report, compliance_result, safe_to_serve=False)

        return GuardedResponse(
            original_response=response,
            final_answer=final_answer,
            confidence_sufficient=confidence_sufficient,
            bias_report=bias_report,
            compliance_result=compliance_result,
            disclaimers=[],
            safe_to_serve=False,
        )

    def _get_fallback_message(self) -> str:
        """Get safe fallback message when response is not safe to serve."""
        return (
            "I'm unable to provide a reliable answer to this question. "
            "Please consult the county directly or try rephrasing your query with more specific details."
        )

    def _log_to_mlflow(
        self,
        response: RAGResponse,
        bias_report: BiasReport,
        compliance_result: ComplianceResult,
        safe_to_serve: bool,
    ) -> None:
        """
        Log guardrail decisions to MLflow.

        # DP-100: Model monitoring - Logging guardrail decisions enables
        tracking of safety metrics and compliance rates.
        """
        try:
            import mlflow

            mlflow.set_experiment("countyiq-guardrails")
            with mlflow.start_run(run_name=f"guardrail_{response.county_fips}", nested=True):
                mlflow.log_param("county_fips", response.county_fips)
                mlflow.log_param("category", str(response.category) if response.category else None)
                mlflow.log_metric("confidence_score", response.confidence_score)
                mlflow.log_metric("source_count", len(response.sources))
                mlflow.log_metric("bias_detected", 1 if bias_report.bias_detected else 0)
                mlflow.log_metric("pii_detected", 1 if compliance_result.pii_detected else 0)
                mlflow.log_metric("safe_to_serve", 1 if safe_to_serve else 0)
                mlflow.log_metric("compliant", 1 if compliance_result.compliant else 0)
                if bias_report.flagged_terms:
                    mlflow.log_param("flagged_terms", ",".join(bias_report.flagged_terms))
        except Exception as e:
            logger.warning("Failed to log guardrail metrics to MLflow: {}", e)
