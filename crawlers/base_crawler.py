"""
Base crawler class for all county data crawlers.
Production implementation with robots.txt checking, rate limiting, retries, and registry updates.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from loguru import logger

from data.schemas.county import CrawlRecord
from data.schemas.document import ContentType, CountyDocument, DocumentCategory
from data.schemas.registry_loader import get_county, update_crawl_status


# DP-100: Data pipeline - Crawler as ETL component extracting county data
class BaseCrawler(ABC):
    """
    Abstract base class for all county data crawlers.
    
    Features:
    - robots.txt checking (cached per domain)
    - Rate limiting per domain
    - Retry logic with exponential backoff
    - Session management with realistic headers
    - Idempotent save() via source_url hash
    - Crawl logging to JSONL
    - Registry crawl_status updates
    """

    def __init__(
        self,
        fips: str,
        category: DocumentCategory,
        rate_limit_seconds: float = 2.0,
        max_retries: int = 3,
        timeout: int = 30,
    ) -> None:
        """
        Initialize base crawler.

        Args:
            fips: 5-digit county FIPS code.
            category: Document category (property, legal, etc.).
            rate_limit_seconds: Minimum seconds between requests to same domain (default 2.0).
            max_retries: Maximum retry attempts for failed requests (default 3).
            timeout: Request timeout in seconds (default 30).
        """
        self.fips = str(fips).strip().zfill(5)
        self.category = category
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.timeout = timeout

        # Get county info from registry
        county = get_county(self.fips)
        if not county:
            raise ValueError(f"County FIPS {self.fips} not found in registry")
        self.county_name = county.county_name
        self.state_abbr = county.state_abbr

        # Session with realistic headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CountyIQ/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

        # Rate limiting: track last request time per domain
        self._last_request_time: dict[str, float] = {}

        # robots.txt cache: domain -> RobotFileParser
        self._robots_cache: dict[str, RobotFileParser] = {}

        # Output paths
        self._raw_dir = Path("data/raw") / self.fips / self.category.value
        self._crawl_log_path = Path("data/raw") / self.fips / "crawl_log.jsonl"

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting and robots.txt."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _check_robots(self, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt.
        Caches RobotFileParser per domain.

        Args:
            url: URL to check.

        Returns:
            True if allowed, False if disallowed.
        """
        domain = self._get_domain(url)
        if domain not in self._robots_cache:
            robots_url = urljoin(domain, "/robots.txt")
            rp = RobotFileParser()
            try:
                rp.set_url(robots_url)
                rp.read()
                self._robots_cache[domain] = rp
                logger.debug("Loaded robots.txt for {}", domain)
            except Exception as e:
                logger.warning("Could not load robots.txt from {}: {}", robots_url, e)
                # If robots.txt fails, allow access (fail open)
                rp = RobotFileParser()
                rp.set_url(robots_url)
                self._robots_cache[domain] = rp

        rp = self._robots_cache[domain]
        return rp.can_fetch("CountyIQ/1.0", url)

    def _enforce_rate_limit(self, url: str) -> None:
        """
        Enforce rate limiting per domain.
        Sleeps if needed to maintain minimum gap between requests.
        """
        domain = self._get_domain(url)
        now = time.time()
        if domain in self._last_request_time:
            elapsed = now - self._last_request_time[domain]
            if elapsed < self.rate_limit_seconds:
                sleep_time = self.rate_limit_seconds - elapsed
                logger.debug("Rate limiting: sleeping {:.2f}s for {}", sleep_time, domain)
                time.sleep(sleep_time)
        self._last_request_time[domain] = time.time()

    def fetch(self, url: str, **kwargs: Any) -> requests.Response | None:
        """
        Fetch a URL with robots.txt check, rate limiting, and retry logic.

        Args:
            url: URL to fetch (absolute).
            **kwargs: Additional arguments to pass to requests.get.

        Returns:
            Response object or None if all retries failed or robots.txt disallowed.
        """
        # Check robots.txt
        if not self._check_robots(url):
            logger.warning("robots.txt disallows: {}", url)
            return None

        # Enforce rate limit
        self._enforce_rate_limit(url)

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout, **kwargs)
                # DP-100: Error handling - Retry on 429 (rate limit) and 503 (service unavailable)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning("Rate limited (429), retrying after {}s", retry_after)
                    time.sleep(retry_after)
                    continue
                if response.status_code == 503:
                    wait_time = 2 ** attempt
                    logger.warning("Service unavailable (503), retrying after {}s", wait_time)
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response
            except requests.exceptions.ConnectionError as e:
                wait_time = 2 ** attempt
                logger.warning("Connection error (attempt {}/{}): {}, retrying after {}s", 
                             attempt + 1, self.max_retries, e, wait_time)
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error("Failed to fetch {} after {} attempts", url, self.max_retries)
                    return None
            except requests.exceptions.RequestException as e:
                logger.warning("Request failed (attempt {}/{}): {}", attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Failed to fetch {} after {} attempts", url, self.max_retries)
                    return None
        return None

    def fetch_pdf(self, url: str, **kwargs: Any) -> bytes | None:
        """
        Fetch a URL and return raw bytes if response is PDF.
        Uses same robots.txt, rate limiting, and retry logic as fetch().

        Args:
            url: URL to fetch (absolute).
            **kwargs: Additional arguments to pass to requests.get.

        Returns:
            PDF bytes if Content-Type is application/pdf; None otherwise or on failure.
        """
        response = self.fetch(url, **kwargs)
        if not response:
            return None
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "application/pdf" not in content_type:
            logger.warning("fetch_pdf: URL did not return PDF (Content-Type: {})", content_type)
            return None
        return response.content

    def _source_url_hash(self, source_url: str | None) -> str:
        """Generate hash of source_url for idempotency check."""
        if not source_url:
            return ""
        return hashlib.md5(source_url.encode()).hexdigest()

    def save(self, documents: list[CountyDocument]) -> None:
        """
        Save documents to data/raw/{fips}/{category}/ as JSON files.
        Idempotent: skips if file with same source_url hash already exists.

        Args:
            documents: List of CountyDocument to save.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        skipped_count = 0

        for doc in documents:
            # Generate filename from source_url hash for idempotency
            url_hash = self._source_url_hash(doc.source_url)
            if url_hash:
                filename = f"{url_hash}.json"
            else:
                # Fallback: use document ID
                filename = f"{doc.id}.json"

            file_path = self._raw_dir / filename

            # Idempotency check: skip if file exists
            if file_path.exists():
                logger.debug("Skipping {} (already exists)", file_path)
                skipped_count += 1
                continue

            # Save document as JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(doc.model_dump(mode="json"), f, indent=2, ensure_ascii=False, default=str)
            saved_count += 1

        logger.info("Saved {} documents, skipped {} (idempotent) to {}", 
                   saved_count, skipped_count, self._raw_dir)

    def log_crawl_record(self, success: bool, record_count: int | None, error: str | None) -> None:
        """
        Write CrawlRecord to data/raw/{fips}/crawl_log.jsonl.

        Args:
            success: Whether crawl succeeded.
            record_count: Number of records crawled (None if failed).
            error: Error message if failed (None if succeeded).
        """
        self._crawl_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = CrawlRecord(
            fips=self.fips,
            category=self.category.value,
            url=None,  # Could be set by subclass
            timestamp=datetime.utcnow(),
            success=success,
            record_count=record_count,
            error_message=error,
        )
        with open(self._crawl_log_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        logger.debug("Logged crawl record: success={}, count={}", success, record_count)

    @abstractmethod
    def crawl(self, fips: str) -> list[CountyDocument]:
        """
        Main crawl method to be implemented by subclasses.
        Must return list of CountyDocument instances.

        Args:
            fips: 5-digit county FIPS code (same as self.fips, provided for convenience).

        Returns:
            List of CountyDocument instances.
        """
        ...

    def run(self) -> list[CountyDocument]:
        """
        Run the crawler: update registry status, crawl, save, log, update registry again.
        Call this instead of crawl() directly.

        Returns:
            List of CountyDocument instances that were crawled.
        """
        logger.info("Starting crawl: FIPS={}, category={}", self.fips, self.category.value)
        
        # Update registry: set status to "active"
        try:
            update_crawl_status(self.fips, self.category.value, "active")
        except Exception as e:
            logger.warning("Failed to update registry status to active: {}", e)

        documents: list[CountyDocument] = []
        error_msg: str | None = None

        try:
            # Run subclass crawl() implementation
            documents = self.crawl(self.fips)
            logger.info("Crawled {} documents for FIPS {} category {}", 
                       len(documents), self.fips, self.category.value)
            
            # Save documents
            if documents:
                self.save(documents)
            
            # Log success
            self.log_crawl_record(success=True, record_count=len(documents), error=None)
            
            # Update registry: set status to "complete"
            try:
                update_crawl_status(self.fips, self.category.value, "complete")
            except Exception as e:
                logger.warning("Failed to update registry status to complete: {}", e)

        except Exception as e:
            error_msg = str(e)
            logger.error("Crawl failed for FIPS {} category {}: {}", 
                        self.fips, self.category.value, e)
            
            # Log failure
            self.log_crawl_record(success=False, record_count=None, error=error_msg)
            
            # Update registry: set status to "failed"
            try:
                update_crawl_status(self.fips, self.category.value, "failed")
            except Exception as update_error:
                logger.warning("Failed to update registry status to failed: {}", update_error)
            
            raise

        return documents
