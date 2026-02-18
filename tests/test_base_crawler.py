"""
Comprehensive tests for BaseCrawler.
Uses unittest.mock to avoid real network requests.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from crawlers.base_crawler import BaseCrawler
from data.schemas.document import ContentType, CountyDocument, DocumentCategory


class StubCrawler(BaseCrawler):
    """Concrete crawler implementation for testing."""

    def crawl(self, fips: str) -> list[CountyDocument]:
        """Return test documents."""
        return [
            CountyDocument(
                fips=fips,
                county_name="Test County",
                state_abbr="AL",
                category=DocumentCategory.property,
                source_url="https://example.com/property/1",
                content_type=ContentType.html,
                raw_content="<html>test</html>",
            )
        ]


class TestBaseCrawler:
    """Test suite for BaseCrawler."""

    def test_initialization(self) -> None:
        """Test crawler initialization."""
        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        assert crawler.fips == "01001"
        assert crawler.category == DocumentCategory.property
        assert crawler.rate_limit_seconds == 2.0
        assert crawler.max_retries == 3
        assert crawler.timeout == 30

    def test_initialization_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        crawler = StubCrawler(
            fips="01001",
            category=DocumentCategory.property,
            rate_limit_seconds=5.0,
            max_retries=5,
            timeout=60,
        )
        assert crawler.rate_limit_seconds == 5.0
        assert crawler.max_retries == 5
        assert crawler.timeout == 60

    def test_get_domain(self) -> None:
        """Test domain extraction."""
        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        assert crawler._get_domain("https://example.com/page") == "https://example.com"
        assert crawler._get_domain("http://test.org/path") == "http://test.org"

    @patch("crawlers.base_crawler.RobotFileParser")
    def test_robots_txt_allowed(self, mock_robot_parser_class: MagicMock) -> None:
        """Test robots.txt check when URL is allowed."""
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        mock_robot_parser_class.return_value = mock_rp

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        # Clear cache to force fresh check
        crawler._robots_cache.clear()
        result = crawler._check_robots("https://example.com/page")

        assert result is True
        mock_rp.set_url.assert_called_once()
        mock_rp.read.assert_called_once()
        mock_rp.can_fetch.assert_called_once_with("CountyIQ/1.0", "https://example.com/page")

    @patch("crawlers.base_crawler.RobotFileParser")
    def test_robots_txt_disallowed(self, mock_robot_parser_class: MagicMock) -> None:
        """Test robots.txt check when URL is disallowed."""
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = False
        mock_robot_parser_class.return_value = mock_rp

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        # Clear cache to force fresh check
        crawler._robots_cache.clear()
        result = crawler._check_robots("https://example.com/page")

        assert result is False

    def test_rate_limiting_enforced(self) -> None:
        """Test that rate limiting enforces minimum gap between requests."""
        crawler = StubCrawler(fips="01001", category=DocumentCategory.property, rate_limit_seconds=0.5)
        url = "https://example.com/page"

        start = time.time()
        crawler._enforce_rate_limit(url)
        first_call = time.time() - start

        start = time.time()
        crawler._enforce_rate_limit(url)
        second_call = time.time() - start

        # First call should be fast (no wait)
        assert first_call < 0.1
        # Second call should wait at least rate_limit_seconds
        assert second_call >= 0.5

    def test_rate_limiting_different_domains(self) -> None:
        """Test that rate limiting is per-domain."""
        crawler = StubCrawler(fips="01001", category=DocumentCategory.property, rate_limit_seconds=0.5)

        start = time.time()
        crawler._enforce_rate_limit("https://example.com/page1")
        crawler._enforce_rate_limit("https://other.com/page1")  # Different domain
        elapsed = time.time() - start

        # Should not wait between different domains
        assert elapsed < 0.1

    @patch("crawlers.base_crawler.BaseCrawler._check_robots")
    @patch("crawlers.base_crawler.BaseCrawler._enforce_rate_limit")
    def test_fetch_success(
        self,
        mock_rate_limit: MagicMock,
        mock_robots: MagicMock,
    ) -> None:
        """Test successful fetch."""
        mock_robots.return_value = True
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        crawler.session.get = Mock(return_value=mock_response)
        result = crawler.fetch("https://example.com/page")

        assert result == mock_response
        mock_robots.assert_called_once_with("https://example.com/page")
        mock_rate_limit.assert_called_once_with("https://example.com/page")
        crawler.session.get.assert_called_once()

    @patch("crawlers.base_crawler.BaseCrawler._check_robots")
    def test_fetch_robots_disallowed(self, mock_robots: MagicMock) -> None:
        """Test fetch returns None when robots.txt disallows."""
        mock_robots.return_value = False

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        result = crawler.fetch("https://example.com/page")

        assert result is None

    @patch("crawlers.base_crawler.BaseCrawler._check_robots")
    @patch("crawlers.base_crawler.BaseCrawler._enforce_rate_limit")
    def test_fetch_retry_on_503(
        self,
        mock_rate_limit: MagicMock,
        mock_robots: MagicMock,
    ) -> None:
        """Test retry logic triggers on 503 status."""
        mock_robots.return_value = True

        # First attempt: 503, second attempt: success
        mock_response_503 = Mock()
        mock_response_503.status_code = 503
        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.raise_for_status = Mock()

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property, max_retries=3)
        crawler.session.get = Mock(side_effect=[mock_response_503, mock_response_200])
        with patch("time.sleep"):  # Mock sleep to speed up test
            result = crawler.fetch("https://example.com/page")

        assert result == mock_response_200
        assert crawler.session.get.call_count == 2

    @patch("crawlers.base_crawler.BaseCrawler._check_robots")
    @patch("crawlers.base_crawler.BaseCrawler._enforce_rate_limit")
    def test_fetch_retry_on_429(
        self,
        mock_rate_limit: MagicMock,
        mock_robots: MagicMock,
    ) -> None:
        """Test retry logic on 429 (rate limit) status."""
        mock_robots.return_value = True

        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}
        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.raise_for_status = Mock()

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        crawler.session.get = Mock(side_effect=[mock_response_429, mock_response_200])
        with patch("time.sleep"):
            result = crawler.fetch("https://example.com/page")

        assert result == mock_response_200
        assert crawler.session.get.call_count == 2

    @patch("crawlers.base_crawler.BaseCrawler._check_robots")
    @patch("crawlers.base_crawler.BaseCrawler._enforce_rate_limit")
    def test_fetch_retry_on_connection_error(
        self,
        mock_rate_limit: MagicMock,
        mock_robots: MagicMock,
    ) -> None:
        """Test retry logic on connection error."""
        mock_robots.return_value = True

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property, max_retries=3)
        crawler.session.get = Mock(side_effect=[
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response,
        ])
        with patch("time.sleep"):
            result = crawler.fetch("https://example.com/page")

        assert result == mock_response
        assert crawler.session.get.call_count == 2

    def test_source_url_hash(self) -> None:
        """Test source URL hash generation."""
        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        hash1 = crawler._source_url_hash("https://example.com/page")
        hash2 = crawler._source_url_hash("https://example.com/page")
        hash3 = crawler._source_url_hash("https://example.com/other")

        assert hash1 == hash2  # Same URL produces same hash
        assert hash1 != hash3  # Different URLs produce different hashes
        assert hash1 != ""  # Non-empty hash

    def test_source_url_hash_empty(self) -> None:
        """Test source URL hash with None/empty URL."""
        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        assert crawler._source_url_hash(None) == ""
        assert crawler._source_url_hash("") == ""

    def test_save_idempotent(self, tmp_path: Path) -> None:
        """Test that save() is idempotent (skips existing files)."""
        mock_raw_dir = tmp_path / "data" / "raw" / "01001" / "property"

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        crawler._raw_dir = mock_raw_dir

        doc = CountyDocument(
            fips="01001",
            county_name="Test County",
            state_abbr="AL",
            category=DocumentCategory.property,
            source_url="https://example.com/property/1",
            content_type=ContentType.html,
            raw_content="<html>test</html>",
        )

        # First save
        crawler.save([doc])
        first_count = len(list(mock_raw_dir.glob("*.json")))

        # Second save (should skip)
        crawler.save([doc])
        second_count = len(list(mock_raw_dir.glob("*.json")))

        assert first_count == 1
        assert second_count == 1  # No new file created

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Test that save() creates directory if it doesn't exist."""
        mock_raw_dir = tmp_path / "data" / "raw" / "01001" / "property"

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        crawler._raw_dir = mock_raw_dir

        doc = CountyDocument(
            fips="01001",
            county_name="Test County",
            state_abbr="AL",
            category=DocumentCategory.property,
            source_url="https://example.com/property/1",
            content_type=ContentType.html,
            raw_content="<html>test</html>",
        )

        assert not mock_raw_dir.exists()
        crawler.save([doc])
        assert mock_raw_dir.exists()

    def test_log_crawl_record(self, tmp_path: Path) -> None:
        """Test that log_crawl_record writes to crawl_log.jsonl."""
        mock_log_path = tmp_path / "crawl_log.jsonl"

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        crawler._crawl_log_path = mock_log_path

        crawler.log_crawl_record(success=True, record_count=5, error=None)

        assert mock_log_path.exists()
        with open(mock_log_path, encoding="utf-8") as f:
            line = f.readline()
            record = json.loads(line)

        assert record["fips"] == "01001"
        assert record["category"] == "property"
        assert record["success"] is True
        assert record["record_count"] == 5
        assert record["error_message"] is None

    def test_log_crawl_record_failure(self, tmp_path: Path) -> None:
        """Test logging failed crawl record."""
        mock_log_path = tmp_path / "crawl_log.jsonl"

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        crawler._crawl_log_path = mock_log_path

        crawler.log_crawl_record(success=False, record_count=None, error="Connection timeout")

        with open(mock_log_path, encoding="utf-8") as f:
            line = f.readline()
            record = json.loads(line)

        assert record["success"] is False
        assert record["record_count"] is None
        assert record["error_message"] == "Connection timeout"

    @patch("crawlers.base_crawler.update_crawl_status")
    @patch("crawlers.base_crawler.get_county")
    def test_run_updates_registry_status(
        self,
        mock_get_county: MagicMock,
        mock_update_status: MagicMock,
    ) -> None:
        """Test that run() updates registry status to active, then complete."""
        from data.schemas.county import County, CrawlStatus, DataSources

        mock_county = County(
            fips="01001",
            county_name="Test County",
            state_name="Alabama",
            state_abbr="AL",
            data_sources=DataSources(),
            crawl_status=CrawlStatus(),
        )
        mock_get_county.return_value = mock_county

        crawler = StubCrawler(fips="01001", category=DocumentCategory.property)
        with patch.object(crawler, "save"), patch.object(crawler, "log_crawl_record"):
            crawler.run()

        # Should update to active, then complete
        assert mock_update_status.call_count == 2
        mock_update_status.assert_any_call("01001", "property", "active")
        mock_update_status.assert_any_call("01001", "property", "complete")

    @patch("crawlers.base_crawler.update_crawl_status")
    @patch("crawlers.base_crawler.get_county")
    def test_run_updates_registry_on_failure(
        self,
        mock_get_county: MagicMock,
        mock_update_status: MagicMock,
    ) -> None:
        """Test that run() updates registry status to failed on exception."""
        from data.schemas.county import County, CrawlStatus, DataSources

        mock_county = County(
            fips="01001",
            county_name="Test County",
            state_name="Alabama",
            state_abbr="AL",
            data_sources=DataSources(),
            crawl_status=CrawlStatus(),
        )
        mock_get_county.return_value = mock_county

        class FailingCrawler(BaseCrawler):
            def crawl(self, fips: str) -> list[CountyDocument]:
                raise ValueError("Test error")

        crawler = FailingCrawler(fips="01001", category=DocumentCategory.property)
        with patch.object(crawler, "log_crawl_record"):
            with pytest.raises(ValueError):
                crawler.run()

        # Should update to active, then failed
        assert mock_update_status.call_count == 2
        mock_update_status.assert_any_call("01001", "property", "active")
        mock_update_status.assert_any_call("01001", "property", "failed")
