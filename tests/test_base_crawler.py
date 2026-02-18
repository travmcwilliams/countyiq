"""Tests for base crawler."""

import pytest

from crawlers.base_crawler import BaseCrawler


class TestCrawler(BaseCrawler):
    """Concrete crawler for tests."""

    def crawl(self) -> list[dict]:
        """Return minimal test data."""
        return [{"test": "data"}]


def test_base_crawler_initialization() -> None:
    """Test base crawler initialization."""
    crawler = TestCrawler(base_url="https://example.com")
    assert crawler.base_url == "https://example.com"
    assert crawler.delay_seconds == 1.0
    assert crawler.max_retries == 3


def test_is_valid_url() -> None:
    """Test URL validation."""
    crawler = TestCrawler(base_url="https://example.com")
    assert crawler.is_valid_url("https://example.com/page") is True
    assert crawler.is_valid_url("invalid-url") is False


def test_crawl() -> None:
    """Test crawl method."""
    crawler = TestCrawler(base_url="https://example.com")
    results = crawler.crawl()
    assert len(results) == 1
    assert results[0]["test"] == "data"
