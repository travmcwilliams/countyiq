"""
Census Bureau demographics ingestion for CountyIQ.
Fetches 2020 Decennial Census data by county FIPS and returns CountyDocuments.
"""

from typing import Any

import requests
from loguru import logger

from data.schemas.document import ContentType, CountyDocument, DocumentCategory
from data.schemas.registry_loader import get_county, update_county_population
from pipelines.ingest.structured_processor import StructuredProcessor


# DP-100: Data pipeline - External API as data source for ML/analytics
class CensusDemographicsProcessor:
    """
    Fetches US Census Bureau 2020 Decennial Census demographics for a county
    and returns CountyDocuments with category=demographics.
    Updates population in county registry.
    """

    BASE_URL = "https://api.census.gov/data/2020/dec/pl"
    VARIABLES = "NAME,P1_001N,P1_003N,P1_004N,P1_005N,P1_006N"  # total, white, black, aian, asian

    def __init__(self, timeout: int = 30) -> None:
        """
        Initialize Census demographics processor.

        Args:
            timeout: Request timeout in seconds (default 30).
        """
        self.timeout = timeout
        self._structured = StructuredProcessor()

    def _fips_parts(self, fips: str) -> tuple[str, str]:
        """Return (state_fips, county_fips) from 5-digit FIPS."""
        fips = str(fips).strip().zfill(5)
        return fips[:2], fips[2:]

    def fetch(self, fips: str) -> dict[str, Any] | list[Any] | None:
        """
        Fetch Census API response for the given county FIPS.

        Args:
            fips: 5-digit county FIPS code.

        Returns:
            Parsed JSON (list of lists from Census) or None on failure.
        """
        state_fips, county_fips = self._fips_parts(fips)
        url = (
            f"{self.BASE_URL}?get={self.VARIABLES}"
            f"&for=county:{county_fips}&in=state:{state_fips}"
        )
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            logger.info("Fetched Census demographics for FIPS {} ({} rows)", fips, len(data) if isinstance(data, list) else 0)
            return data
        except requests.RequestException as e:
            logger.warning("Census API request failed for FIPS {}: {}", fips, e)
            return None

    def process(self, fips: str) -> list[CountyDocument]:
        """
        Fetch Census demographics for the county and return CountyDocuments.
        Updates county registry population when total is available.

        Args:
            fips: 5-digit county FIPS code.

        Returns:
            List of CountyDocument with category=demographics.
        """
        fips = str(fips).strip().zfill(5)
        county = get_county(fips)
        if not county:
            logger.warning("County FIPS {} not in registry", fips)
            return []

        response = self.fetch(fips)
        if not response or not isinstance(response, list) or len(response) < 2:
            logger.warning("No Census data for FIPS {}", fips)
            return []

        # Census returns [headers, row1, row2, ...]; row is list of values
        headers: list[str] = [str(h) for h in response[0]]
        rows = response[1:]

        # Convert to list of dicts for structured processor
        records: list[dict[str, Any]] = []
        for row in rows:
            rec = {}
            for i, val in enumerate(row):
                if i < len(headers):
                    key = headers[i]
                    rec[key] = val if val is not None else None
            records.append(rec)

        # Extract total population (P1_001N) and update registry
        if records:
            first = records[0]
            total_pop = first.get("P1_001N")
            if total_pop is not None:
                try:
                    pop_int = int(total_pop)
                    update_county_population(fips, pop_int)
                except (ValueError, TypeError):
                    pass

        # Build API-like payload for process_api (record_path=None, we pass list)
        source_url = f"{self.BASE_URL}?get=...&for=county:{fips[2:]}&in=state:{fips[:2]}"
        documents = self._structured.process_api(
            source_url=source_url,
            fips=fips,
            category=DocumentCategory.demographics,
            response_json=records,
            record_path=None,
        )

        # process_api expects response_json to be dict or list; we passed list of dicts
        # but process_api with list treats it as list of records. So we're good.
        return documents
