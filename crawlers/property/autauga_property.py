"""
Autauga County AL property records crawler.
Targets county assessor/GIS site for property data.
"""

import json
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger

from crawlers.base_crawler import BaseCrawler
from data.schemas.document import ContentType, DocumentCategory


class AutaugaPropertyCrawler(BaseCrawler):
    """
    Crawler for Autauga County AL property records.
    Extracts: parcel ID, owner name, address, assessed value, land use code, acreage.
    """

    def __init__(self, fips: str = "01001", **kwargs: Any) -> None:
        """
        Initialize Autauga County property crawler.

        Args:
            fips: County FIPS code (default "01001" for Autauga County AL).
            **kwargs: Additional arguments passed to BaseCrawler.
        """
        super().__init__(fips=fips, category=DocumentCategory.property, **kwargs)
        # Autauga County property search URL
        self.base_url = "https://www.autaugaal.org"

    def crawl(self, fips: str) -> list[Any]:
        """
        Crawl Autauga County property records.

        Args:
            fips: County FIPS code (must match self.fips).

        Returns:
            List of CountyDocument instances with property data.
        """
        if fips != self.fips:
            raise ValueError(f"FIPS mismatch: expected {self.fips}, got {fips}")

        documents: list[Any] = []

        # Autauga County property search page
        search_url = f"{self.base_url}/Property_Search.html"

        logger.info("Fetching property records from {}", search_url)
        response = self.fetch(search_url)

        if not response:
            logger.warning("Failed to fetch property search page")
            return documents

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract property records from page
        # Common patterns: table rows, div containers, JSON data embedded in script tags
        # Adjust selectors based on actual site structure

        # Pattern 1: Table rows (common for property listings)
        property_rows = soup.select("table.property-list tr, table.parcel-list tr, .property-row")
        
        # Pattern 2: Div containers
        if not property_rows:
            property_rows = soup.select(".property-card, .parcel-card, [data-parcel-id]")

        # Pattern 3: Embedded JSON in script tag
        if not property_rows:
            script_tags = soup.find_all("script", type="application/json")
            for script in script_tags:
                try:
                    import json
                    data = json.loads(script.string)
                    if isinstance(data, list):
                        property_rows = data
                    elif isinstance(data, dict) and "properties" in data:
                        property_rows = data["properties"]
                except Exception:
                    continue

        for row in property_rows[:100]:  # Limit to first 100 for initial crawl
            try:
                doc = self._extract_property_document(row, search_url)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning("Failed to extract property from row: {}", e)
                continue

        # If no properties found via selectors, save the raw page as a document
        if not documents:
            logger.info("No property records extracted, saving raw page HTML")
            from data.schemas.document import CountyDocument
            doc = CountyDocument(
                fips=self.fips,
                county_name=self.county_name,
                state_abbr=self.state_abbr,
                category=DocumentCategory.property,
                source_url=search_url,
                content_type=ContentType.html,
                raw_content=response.text,
                metadata={"extraction_note": "No structured records found, saved raw HTML"},
            )
            documents.append(doc)

        return documents

    def _extract_property_document(self, element: Any, source_url: str) -> Any | None:
        """
        Extract property data from HTML element and create CountyDocument.

        Args:
            element: BeautifulSoup element or dict containing property data.
            source_url: Source URL for this property.

        Returns:
            CountyDocument instance or None if extraction fails.
        """
        from data.schemas.document import CountyDocument

        # Handle dict (from JSON)
        if isinstance(element, dict):
            parcel_id = str(element.get("parcel_id") or element.get("parcelId") or element.get("id") or "")
            owner_name = str(element.get("owner_name") or element.get("ownerName") or element.get("owner") or "")
            address = str(element.get("address") or element.get("property_address") or "")
            assessed_value = element.get("assessed_value") or element.get("assessedValue") or element.get("value")
            land_use = str(element.get("land_use") or element.get("landUse") or element.get("use_code") or "")
            acreage = element.get("acreage") or element.get("acres") or element.get("land_acres")
            raw_html = json.dumps(element, indent=2) if hasattr(json, "dumps") else str(element)
        else:
            # Handle BeautifulSoup element
            parcel_id = self._extract_text(element, ["td.parcel-id", ".parcel-id", "[data-parcel-id]"])
            owner_name = self._extract_text(element, ["td.owner", ".owner-name", ".owner"])
            address = self._extract_text(element, ["td.address", ".property-address", ".address"])
            assessed_value_text = self._extract_text(element, ["td.value", ".assessed-value", ".value"])
            assessed_value = self._parse_currency(assessed_value_text)
            land_use = self._extract_text(element, ["td.land-use", ".land-use", ".use-code"])
            acreage_text = self._extract_text(element, ["td.acreage", ".acreage", ".acres"])
            acreage = self._parse_float(acreage_text)
            raw_html = str(element)

        # Build metadata dict
        metadata: dict[str, Any] = {}
        if parcel_id:
            metadata["parcel_id"] = parcel_id
        if owner_name:
            metadata["owner_name"] = owner_name
        if address:
            metadata["address"] = address
        if assessed_value is not None:
            metadata["assessed_value"] = assessed_value
        if land_use:
            metadata["land_use_code"] = land_use
        if acreage is not None:
            metadata["acreage"] = acreage

        # Create document
        doc = CountyDocument(
            fips=self.fips,
            county_name=self.county_name,
            state_abbr=self.state_abbr,
            category=DocumentCategory.property,
            source_url=source_url,
            content_type=ContentType.html,
            raw_content=raw_html,
            metadata=metadata,
        )

        return doc

    def _extract_text(self, element: Any, selectors: list[str]) -> str:
        """Extract text from element using list of CSS selectors."""
        if not hasattr(element, "select"):
            return ""
        for selector in selectors:
            found = element.select_one(selector)
            if found:
                return found.get_text(strip=True)
        return ""

    def _parse_currency(self, text: str) -> float | None:
        """Parse currency string to float (e.g., '$123,456.78' -> 123456.78)."""
        if not text:
            return None
        try:
            cleaned = text.replace("$", "").replace(",", "").strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return None

    def _parse_float(self, text: str) -> float | None:
        """Parse float string."""
        if not text:
            return None
        try:
            cleaned = text.replace(",", "").strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return None
