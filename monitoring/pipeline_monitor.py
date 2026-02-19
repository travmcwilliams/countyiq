"""Pipeline monitoring: MLflow metrics for crawl and RAG.

# DP-100: Model monitoring - Logging pipeline metrics for observability and alerting.
"""

from pathlib import Path
from typing import Any, List, Optional, Protocol

from loguru import logger
from pydantic import BaseModel, Field

from pipelines.ingest.crawl_orchestrator import CrawlSummary, OrchestrationResult
from pipelines.ingest.county_registry import load_counties

EXPERIMENT_NAME = "countyiq-monitoring"
# data/processed/crawl_reports relative to project root (parent of monitoring/)
REPORTS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "crawl_reports"


class RAGResponseLike(Protocol):
    """Protocol for RAG response objects (e.g. RAGResponse from rag.pipeline)."""

    @property
    def latency_ms(self) -> int:
        ...

    @property
    def confidence(self) -> float:
        ...

    @property
    def sources(self) -> List[Any]:
        ...


class HealthReport(BaseModel):
    """Aggregated health of the crawl and RAG pipeline."""

    total_counties: int = Field(default=0, ge=0)
    crawled_counties: int = Field(default=0, ge=0)
    pending_counties: int = Field(default=0, ge=0)
    failed_counties: int = Field(default=0, ge=0)
    total_documents: int = Field(default=0, ge=0)
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_crawl_time: Optional[str] = Field(default=None)


class PipelineMonitor:
    """
    Logs crawl and RAG metrics to MLflow and provides health summary.

    # DP-100: Model monitoring - Centralized metric logging for ML pipelines.
    """

    def __init__(self, experiment_name: str = EXPERIMENT_NAME, reports_dir: Path | None = None):
        self.experiment_name = experiment_name
        self.reports_dir = reports_dir or REPORTS_DIR
        self._rag_confidences: List[float] = []

    def log_crawl_metrics(self, summary: CrawlSummary) -> None:
        """Log a single county crawl summary to MLflow."""
        # DP-100: Experiment tracking - Logging pipeline run metrics.
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=f"crawl_{summary.fips}", nested=True):
                mlflow.log_param("fips", summary.fips)
                mlflow.log_param("county_name", summary.county_name)
                mlflow.log_metric("documents_crawled", summary.documents_crawled)
                mlflow.log_metric("duration_seconds", summary.duration_seconds)
                mlflow.log_metric("categories_attempted", len(summary.categories_attempted))
                mlflow.log_metric("categories_succeeded", len(summary.categories_succeeded))
                mlflow.log_metric("error_count", len(summary.errors))
        except Exception as e:
            logger.warning("MLflow log_crawl_metrics failed: {}", e)

    def log_rag_query(self, response: RAGResponseLike) -> None:
        """Log RAG query latency, confidence, and source count to MLflow."""
        # DP-100: Model monitoring - Logging inference latency and quality metrics.
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name="rag_query", nested=True):
                mlflow.log_metric("rag_latency_ms", response.latency_ms)
                mlflow.log_metric("rag_confidence", response.confidence)
                mlflow.log_metric("rag_sources_count", len(response.sources))
            self._rag_confidences.append(response.confidence)
        except Exception as e:
            logger.warning("MLflow log_rag_query failed: {}", e)

    def get_crawl_health(self) -> HealthReport:
        """
        Build health report from latest crawl report and registry.

        Reads data/processed/crawl_reports/latest.json if present.
        """
        # DP-100: Model monitoring - Aggregating pipeline health for CI/CD and alerting.
        total_counties = len(load_counties())
        crawled_counties = 0
        failed_counties = 0
        total_documents = 0
        last_crawl_time: Optional[str] = None

        latest_path = self.reports_dir / "latest.json"
        if latest_path.exists():
            try:
                import json
                from datetime import datetime
                with open(latest_path, encoding="utf-8") as f:
                    data = json.load(f)
                result = OrchestrationResult.model_validate(data)
                crawled_counties = result.succeeded
                failed_counties = result.failed
                total_documents = result.total_documents
                last_crawl_time = datetime.fromtimestamp(latest_path.stat().st_mtime).isoformat()
            except Exception as e:
                logger.warning("Could not parse latest crawl report: {}", e)

        pending_counties = max(0, total_counties - crawled_counties - failed_counties)
        avg_conf = sum(self._rag_confidences) / len(self._rag_confidences) if self._rag_confidences else 0.0

        return HealthReport(
            total_counties=total_counties,
            crawled_counties=crawled_counties,
            pending_counties=pending_counties,
            failed_counties=failed_counties,
            total_documents=total_documents,
            avg_confidence=round(avg_conf, 4),
            last_crawl_time=last_crawl_time,
        )
