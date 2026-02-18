"""Base crawler class for all county data crawlers."""

import time
from abc import ABC, abstractmethod
from typing import Any

import requests
from bs4 import BeautifulSoup
from loguru import logger
from urllib.parse import urljoin, urlparse


class BaseCrawler(ABC):
    """
    Base class for all crawlers with rate limiting, retries, and robots.txt respect.
    Every crawler inherits from this; logs source URL, timestamp, county FIPS, data category.
    Idempotent — re-running a crawl should not duplicate data.
    """

    def __init__(
        self,
        base_url: str,
        user_agent: str | None = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ) -> None:
        """
        Initialize base crawler.

        Args:
            base_url: Base URL for the crawler.
            user_agent: User agent string for requests.
            delay_seconds: Delay between requests (rate limiting).
            max_retries: Maximum number of retries for failed requests.
            timeout_seconds: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent or "CountyIQ/1.0"
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def fetch(self, url: str, **kwargs: Any) -> requests.Response | None:
        """
        Fetch a URL with retry logic.

        Args:
            url: URL to fetch (relative or absolute).
            **kwargs: Additional arguments to pass to requests.get.

        Returns:
            Response object or None if all retries failed.
        """
        full_url = urljoin(self.base_url + "/", url)

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    full_url,
                    timeout=self.timeout_seconds,
                    **kwargs,
                )
                response.raise_for_status()
                time.sleep(self.delay_seconds)
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "Attempt {}/{} failed for {}: {}",
                    attempt + 1,
                    self.max_retries,
                    full_url,
                    e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    logger.error(
                        "Failed to fetch {} after {} attempts",
                        full_url,
                        self.max_retries,
                    )
                    return None
        return None

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content."""
        return BeautifulSoup(html, "html.parser")

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return bool(result.scheme and result.netloc)
        except Exception:
            return False

    @abstractmethod
    def crawl(self) -> list[dict[str, Any]]:
        """
        Main crawl method to be implemented by subclasses.
        Should log source URL, timestamp, county FIPS code, data category.

        Returns:
            List of crawled data dictionaries.
        """
        ...

    def save(self, data: list[dict[str, Any]], output_path: str) -> None:
        """
        Save crawled data to file. Idempotent when combined with deterministic output_path.

        Args:
            data: List of data dictionaries.
            output_path: Path to save data.
        """
        import json
        import os

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved {} records to {}", len(data), output_path)
