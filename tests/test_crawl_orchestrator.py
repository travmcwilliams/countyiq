"""Tests for CrawlOrchestrator and related models."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from pipelines.ingest.crawl_orchestrator import (
    CrawlOrchestrator,
    CrawlSummary,
    OrchestrationResult,
)
from pipelines.ingest.county_registry import load_counties


def test_crawl_summary_model() -> None:
    s = CrawlSummary(
        fips="01001",
        county_name="Autauga",
        categories_attempted=["property"],
        categories_succeeded=["property"],
        documents_crawled=5,
        duration_seconds=1.2,
        errors=[],
    )
    assert s.fips == "01001"
    assert s.documents_crawled == 5


def test_crawl_summary_requires_fips() -> None:
    with pytest.raises(ValidationError):
        CrawlSummary(county_name="Autauga")


def test_orchestration_result_model() -> None:
    r = OrchestrationResult(
        total_counties=100,
        succeeded=80,
        failed=10,
        skipped=10,
        total_documents=500,
        duration_seconds=120.0,
        summaries=[],
    )
    assert r.total_counties == 100


def test_orchestrator_run_county_calls_crawler() -> None:
    mock_crawler = MagicMock()
    mock_crawler.crawl.return_value = [{"x": 1}, {"x": 2}]
    with patch("pipelines.ingest.crawl_orchestrator.get_crawler", return_value=mock_crawler):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            orch = CrawlOrchestrator(raw_base=raw)
            summary = orch.run_county("01001", categories=["property"])
    assert summary.fips == "01001"
    assert summary.documents_crawled == 2
    mock_crawler.crawl.assert_called_once()


def test_orchestrator_run_county_handles_no_crawler() -> None:
    with patch("pipelines.ingest.crawl_orchestrator.get_crawler", return_value=None):
        orch = CrawlOrchestrator()
        summary = orch.run_county("01001", categories=["property"])
    assert summary.documents_crawled == 0
    assert len(summary.errors) > 0


def test_run_state_filters_by_state() -> None:
    """With full registry AL has many counties; only assert we get AL FIPS (01xx)."""
    with patch("pipelines.ingest.crawl_orchestrator.get_crawler") as m:
        m.return_value = MagicMock(crawl=MagicMock(return_value=[]), save=MagicMock())
        orch = CrawlOrchestrator()
        summaries = orch.run_state("AL", categories=["property"])
    assert len(summaries) >= 1
    assert all(s.fips.startswith("01") for s in summaries)


def test_run_all_uses_thread_pool() -> None:
    with patch("pipelines.ingest.crawl_orchestrator.get_crawler") as m:
        crawler = MagicMock()
        crawler.crawl.return_value = [{}]
        crawler.save.return_value = None
        m.return_value = crawler
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            reports = Path(tmp) / "reports"
            orch = CrawlOrchestrator(max_workers=2, raw_base=raw, reports_dir=reports)
            result = orch.run_all(categories=["property"], max_workers=2)
            assert result.total_counties == len(load_counties())
            assert (reports / "latest.json").exists()


def test_orchestrator_saves_report() -> None:
    with patch("pipelines.ingest.crawl_orchestrator.get_crawler") as m:
        m.return_value = MagicMock(crawl=MagicMock(return_value=[]), save=MagicMock())
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "crawl_reports"
            orch = CrawlOrchestrator(reports_dir=reports)
            orch.run_all(max_workers=2)
            latest = reports / "latest.json"
            data = json.loads(latest.read_text())
            assert "total_counties" in data
            assert "summaries" in data


def test_failed_crawls_recorded() -> None:
    with patch("pipelines.ingest.crawl_orchestrator.get_crawler") as m:
        crawler = MagicMock()
        crawler.crawl.side_effect = RuntimeError("network error")
        m.return_value = crawler
        orch = CrawlOrchestrator()
        summary = orch.run_county("01001", categories=["property"])
    assert summary.documents_crawled == 0
    assert any("network" in e for e in summary.errors)


def test_get_pending_counties_without_run() -> None:
    orch = CrawlOrchestrator()
    pending = orch.get_pending_counties()
    assert len(pending) == len(load_counties())


def test_get_pending_counties_after_run() -> None:
    with patch("pipelines.ingest.crawl_orchestrator.get_crawler") as m:
        crawler = MagicMock()
        crawler.crawl.return_value = [{}]
        crawler.save.return_value = None
        m.return_value = crawler
        with tempfile.TemporaryDirectory() as tmp:
            orch = CrawlOrchestrator(reports_dir=Path(tmp) / "reports", raw_base=Path(tmp) / "raw")
            orch.run_all(max_workers=2)
            pending = orch.get_pending_counties()
    assert len(pending) == 0


def test_get_failed_counties() -> None:
    with patch("pipelines.ingest.crawl_orchestrator.get_crawler") as m:
        crawler = MagicMock()
        crawler.crawl.side_effect = RuntimeError("fail")
        m.return_value = crawler
        with tempfile.TemporaryDirectory() as tmp:
            orch = CrawlOrchestrator(reports_dir=Path(tmp) / "reports", raw_base=Path(tmp) / "raw")
            orch.run_all(max_workers=2)
            failed = orch.get_failed_counties()
    assert len(failed) == len(load_counties())
