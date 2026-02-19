"""Generic county crawler: fallback when no category-specific crawler exists.

Fetches official county page for a category and saves raw HTML.
# DP-100: Data ingestion - Fallback crawler for counties without dedicated crawlers.
"""

import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger

from crawlers.base_crawler import BaseCrawler
from pipelines.ingest.county_registry import County


class GenericCrawler(BaseCrawler):
    """
    Fallback crawler that fetches a single URL and saves raw HTML.

    In production, base_url could be discovered via Google Custom Search API
    (e.g. query: "{county_name} county {state} official {category}").
    """

    def __init__(
        self,
        base_url: str,
        fips: str,
        category: str,
        county_name: str = "",
        state_abbr: str = "",
        user_agent: Optional[str] = None,
        delay_seconds: float = 1.0,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ):
        super().__init__(
            base_url=base_url,
            user_agent=user_agent,
            delay_seconds=delay_seconds,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        self.fips = fips
        self.category = category
        self.county_name = county_name
        self.state_abbr = state_abbr

    @classmethod
    def for_county(
        cls,
        fips: str,
        county: County,
        category: str,
        base_url: Optional[str] = None,
    ) -> "GenericCrawler":
        """
        Build a GenericCrawler for a county and category.

        If base_url is not provided, uses a placeholder (in production,
        would be resolved via Google search or a URL registry).
        """
        url = base_url or _placeholder_url(county, category)
        return cls(
            base_url=url,
            fips=fips,
            category=category,
            county_name=county.county_name,
            state_abbr=county.state_abbr,
        )

    def crawl(self) -> List[Dict[str, Any]]:
        """
        Fetch the base URL and return a single document with raw HTML.

        Returns:
            List of one dict: { "raw_html", "url", "fips", "category", "county_name", "state_abbr" }.
        """
        response = self.fetch(self.base_url)
        if response is None:
            return []
        doc = {
            "raw_html": response.text,
            "url": self.base_url,
            "fips": self.fips,
            "category": self.category,
            "county_name": self.county_name,
            "state_abbr": self.state_abbr,
        }
        return [doc]

    def save(self, data: List[Dict], output_path: str) -> None:
        """Save crawled data (raw HTML docs) to output_path as JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved {} record(s) to {}", len(data), output_path)


def _placeholder_url(county: County, category: str) -> str:
    """Return a placeholder URL when no specific URL is configured."""
    # In production, resolve via Google Custom Search or URL registry
    safe_name = county.county_name.replace(" ", "").lower()
    return f"https://www.{safe_name}county{county.state_abbr.lower()}.us/{category}"
