"""Crawl orchestrator: schedule and run county crawls with parallel workers.

# DP-100: Pipeline orchestration - Fan-out across counties with rate limiting and reporting.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipelines.ingest.county_registry import County, load_counties, get_counties_by_state, get_county_by_fips
from pipelines.ingest.crawler_factory import get_crawler


# Default categories when none specified
DEFAULT_CATEGORIES = ["property", "legal", "demographics"]
REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "crawl_reports"
RAW_BASE = Path(__file__).resolve().parents[2] / "data" / "raw"


class CrawlSummary(BaseModel):
    """Result of crawling a single county."""

    fips: str = Field(..., min_length=5, max_length=5)
    county_name: str = Field(default="")
    categories_attempted: List[str] = Field(default_factory=list)
    categories_succeeded: List[str] = Field(default_factory=list)
    documents_crawled: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    errors: List[str] = Field(default_factory=list)


class OrchestrationResult(BaseModel):
    """Result of a full or partial orchestration run."""

    total_counties: int = Field(..., ge=0)
    succeeded: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    total_documents: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    summaries: List[CrawlSummary] = Field(default_factory=list)


class CrawlOrchestrator:
    """
    Orchestrates county crawls with a thread pool and persists reports.

    # DP-100: Pipeline orchestration - Parallel execution with max_workers to avoid
    overwhelming county servers; idempotent runs; structured reporting.
    """

    def __init__(
        self,
        max_workers: int = 10,
        reports_dir: Path | None = None,
        raw_base: Path | None = None,
    ):
        self.max_workers = max_workers
        self.reports_dir = reports_dir or REPORTS_DIR
        self.raw_base = raw_base or RAW_BASE
        self._last_result: Optional[OrchestrationResult] = None

    def run_county(
        self,
        fips: str,
        categories: List[str] | None = None,
    ) -> CrawlSummary:
        """
        Crawl a single county for the given categories.

        Args:
            fips: 5-digit FIPS code.
            categories: List of categories to crawl; default DEFAULT_CATEGORIES.

        Returns:
            CrawlSummary for this county.
        """
        # DP-100: Pipeline orchestration - Single-node crawl unit.
        cats = categories or DEFAULT_CATEGORIES
        county = get_county_by_fips(fips)
        county_name = county.county_name if county else ""
        errors: List[str] = []
        succeeded: List[str] = []
        total_docs = 0
        start = time.perf_counter()

        for category in cats:
            crawler = get_crawler(fips, category)
            if crawler is None:
                errors.append(f"no crawler for {category}")
                continue
            out_dir = self.raw_base / fips / category
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "crawl.json"
            try:
                data = crawler.crawl()
                if data:
                    crawler.save(data, str(out_path))
                    succeeded.append(category)
                    total_docs += len(data)
            except Exception as e:
                errors.append(f"{category}: {e!s}")

        duration = time.perf_counter() - start
        summary = CrawlSummary(
            fips=fips,
            county_name=county_name,
            categories_attempted=cats,
            categories_succeeded=succeeded,
            documents_crawled=total_docs,
            duration_seconds=round(duration, 2),
            errors=errors,
        )
        return summary

    def run_state(
        self,
        state_abbr: str,
        categories: List[str] | None = None,
    ) -> List[CrawlSummary]:
        """Run crawls for all counties in a state. Sequential (no thread pool)."""
        counties = get_counties_by_state(state_abbr)
        summaries: List[CrawlSummary] = []
        for c in counties:
            summaries.append(self.run_county(c.fips, categories))
        return summaries

    def run_all(
        self,
        categories: List[str] | None = None,
        max_workers: int | None = None,
    ) -> OrchestrationResult:
        """
        Run crawls for all counties in the registry with a thread pool.

        Logs progress every 10 counties and saves report to data/processed/crawl_reports/.
        """
        # DP-100: Pipeline orchestration - Fan-out across many counties with bounded concurrency.
        workers = max_workers or self.max_workers
        cats = categories or DEFAULT_CATEGORIES
        counties = load_counties()
        total = len(counties)
        succeeded = 0
        failed = 0
        skipped = 0
        total_documents = 0
        summaries: List[CrawlSummary] = []
        start = time.perf_counter()

        def run_one(c: County) -> CrawlSummary:
            return self.run_county(c.fips, cats)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_county = {executor.submit(run_one, c): c for c in counties}
            done = 0
            for future in as_completed(future_to_county):
                done += 1
                if done % 10 == 0:
                    logger.info("Crawl progress: {}/{} counties", done, total)
                try:
                    summary = future.result()
                    summaries.append(summary)
                    if summary.errors and not summary.categories_succeeded:
                        failed += 1
                    elif summary.categories_succeeded:
                        succeeded += 1
                        total_documents += summary.documents_crawled
                    else:
                        skipped += 1
                except Exception as e:
                    failed += 1
                    c = future_to_county[future]
                    summaries.append(
                        CrawlSummary(
                            fips=c.fips,
                            county_name=c.county_name,
                            categories_attempted=cats,
                            errors=[str(e)],
                        )
                    )

        duration = time.perf_counter() - start
        result = OrchestrationResult(
            total_counties=total,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            total_documents=total_documents,
            duration_seconds=round(duration, 2),
            summaries=summaries,
        )
        self._last_result = result
        self._save_report(result)
        return result

    def _save_report(self, result: OrchestrationResult) -> None:
        """Write orchestration report to data/processed/crawl_reports/."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.reports_dir / f"crawl_report_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        latest = self.reports_dir / "latest.json"
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        logger.info("Saved crawl report to {} and latest.json", path)

    def get_pending_counties(self, category: str | None = None) -> List[County]:
        """
        Return counties that have not been successfully crawled (for category if given).

        Uses latest report: counties that are not in succeeded set are pending.
        If no report exists, all registry counties are pending.
        """
        # DP-100: Pipeline orchestration - Identifying work remaining.
        all_counties = load_counties()
        if not self._last_result:
            return all_counties
        succeeded_fips = {
            s.fips
            for s in self._last_result.summaries
            if s.categories_succeeded
            and (category is None or category in s.categories_succeeded)
        }
        return [c for c in all_counties if c.fips not in succeeded_fips]

    def get_failed_counties(self) -> List[County]:
        """Return counties that failed on the last run (no category succeeded)."""
        if not self._last_result:
            return []
        failed_fips = {
            s.fips
            for s in self._last_result.summaries
            if s.errors and not s.categories_succeeded
        }
        counties = load_counties()
        return [c for c in counties if c.fips in failed_fips]
