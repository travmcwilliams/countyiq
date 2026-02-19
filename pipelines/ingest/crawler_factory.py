"""Crawler factory: map FIPS + category to crawler instance.

# DP-100: Pipeline orchestration - Routing crawl jobs to the correct crawler implementation.
"""

from typing import Dict, Optional

from loguru import logger

from crawlers.base_crawler import BaseCrawler

# Import generic crawler from same package to avoid circular import
from pipelines.ingest.generic_crawler import GenericCrawler
from pipelines.ingest.county_registry import get_county_by_fips


# Registry: (state_abbr, category) -> crawler class (add specific crawlers here when available)
_CRAWLER_REGISTRY: Dict[str, type] = {
    "generic": GenericCrawler,
}


def get_crawler(fips: str, category: str) -> Optional[BaseCrawler]:
    """
    Return the appropriate crawler for the given county and category.

    Maps fips + category to the correct crawler class. Falls back to GenericCrawler
    if no specific crawler exists for that county/category.

    Args:
        fips: 5-digit county FIPS code.
        category: Data category (e.g. 'property', 'legal', 'demographics').

    Returns:
        BaseCrawler instance, or None if county unknown or category invalid.
    """
    county = get_county_by_fips(fips)
    if not county:
        logger.warning("Unknown FIPS for crawler: {}", fips)
        return None

    # TODO: lookup (state_abbr, category) in _CRAWLER_REGISTRY for dedicated crawlers
    crawler_class = _CRAWLER_REGISTRY.get("generic", GenericCrawler)
    try:
        instance = crawler_class.for_county(fips=fips, county=county, category=category)
        return instance
    except Exception as e:
        logger.error("Failed to create crawler for fips={} category={}: {}", fips, category, e)
        return None
