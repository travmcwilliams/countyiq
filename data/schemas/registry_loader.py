"""
Load and update the county registry JSON.
Uses loguru for logging; all values validated via Pydantic County model.
# DP-100: Data ingestion - Loading and validating structured registry for ML/data pipelines.
"""

import json
from pathlib import Path

from loguru import logger

from data.schemas.county import County, CrawlStatus, DataSources  # noqa: I001

# Path to registry relative to project root
_REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "crawlers" / "county_registry.json"

# DP-100: Data validation - Registry loaded and validated as structured data
_CACHE: list[County] | None = None


def _load_raw() -> dict:
    """Load raw JSON from registry file."""
    if not _REGISTRY_PATH.exists():
        logger.warning("Registry file not found: {}", _REGISTRY_PATH)
        return {"counties": [], "metadata": {}}
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_raw(data: dict) -> None:
    """Write registry JSON to file."""
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved registry to {}", _REGISTRY_PATH)


def _county_from_dict(obj: dict) -> County:
    """Build County from registry JSON entry (snake_case or legacy keys)."""
    # Normalize keys: support both "name" and "county_name", "state" and "state_abbr"
    fips = str(obj.get("fips", "")).strip().zfill(5)
    county_name = obj.get("county_name") or obj.get("name") or ""
    state_abbr = (obj.get("state_abbr") or obj.get("state") or "").strip().upper()
    state_name = obj.get("state_name") or ""
    population = obj.get("population")
    if population is not None:
        population = int(population)

    data_sources = obj.get("data_sources")
    if isinstance(data_sources, dict):
        ds = DataSources(
            property=data_sources.get("property"),
            legal=data_sources.get("legal"),
            demographics=data_sources.get("demographics"),
            permits=data_sources.get("permits"),
            zoning=data_sources.get("zoning"),
            courts=data_sources.get("courts"),
            tax=data_sources.get("tax"),
        )
    else:
        ds = DataSources()

    crawl_status = obj.get("crawl_status")
    if isinstance(crawl_status, dict):
        cs = CrawlStatus(**{k: v or "pending" for k, v in crawl_status.items() if k in CrawlStatus.model_fields})
    else:
        cs = CrawlStatus()

    last_crawled = obj.get("last_crawled")
    if last_crawled is not None and isinstance(last_crawled, str):
        from datetime import datetime
        try:
            last_crawled = datetime.fromisoformat(last_crawled.replace("Z", "+00:00"))
        except ValueError:
            last_crawled = None

    schema_version = str(obj.get("schema_version", "1.0"))
    record_count = obj.get("record_count")
    if not isinstance(record_count, dict):
        record_count = {}

    return County(
        schema_version=schema_version,
        fips=fips,
        county_name=county_name,
        state_name=state_name,
        state_abbr=state_abbr,
        population=population,
        record_count=record_count,
        data_sources=ds,
        crawl_status=cs,
        last_crawled=last_crawled,
    )


def load_registry(use_cache: bool = True) -> list[County]:
    """
    Load and validate the county registry from JSON.
    Returns a list of County models. Cached in memory unless use_cache=False.
    """
    global _CACHE
    if use_cache and _CACHE is not None:
        return _CACHE

    raw = _load_raw()
    counties_list = raw.get("counties") or []
    out: list[County] = []
    for i, item in enumerate(counties_list):
        try:
            out.append(_county_from_dict(item))
        except Exception as e:
            logger.error("Invalid county entry at index {}: {} — {}", i, item, e)
            raise
    _CACHE = out
    logger.debug("Loaded {} counties from registry", len(out))
    return out


def get_county(fips: str, use_cache: bool = True) -> County | None:
    """
    Return the county with the given 5-digit FIPS code, or None if not found.
    """
    fips = str(fips).strip().zfill(5)
    for c in load_registry(use_cache=use_cache):
        if c.fips == fips:
            return c
    return None


def update_crawl_status(fips: str, category: str, status: str) -> None:
    """
    Update crawl_status for the given county and category, then persist to JSON.
    category: one of property, legal, demographics, permits, zoning, courts, tax
    status: one of pending, active, complete, failed
    """
    fips = str(fips).strip().zfill(5)
    category = category.strip().lower()
    status = status.strip().lower()

    allowed_categories = {"property", "legal", "demographics", "permits", "zoning", "courts", "tax"}
    allowed_statuses = {"pending", "active", "complete", "failed"}
    if category not in allowed_categories:
        raise ValueError(f"category must be one of {allowed_categories}, got {category!r}")
    if status not in allowed_statuses:
        raise ValueError(f"status must be one of {allowed_statuses}, got {status!r}")

    raw = _load_raw()
    counties_list = raw.get("counties") or []
    for item in counties_list:
        if str(item.get("fips", "")).strip().zfill(5) == fips:
            if "crawl_status" not in item or not isinstance(item["crawl_status"], dict):
                item["crawl_status"] = {}
            item["crawl_status"][category] = status
            raw["counties"] = counties_list
            _save_raw(raw)
            _CACHE = None
            logger.info("Updated crawl_status for FIPS {} category {} -> {}", fips, category, status)
            return

    logger.warning("County FIPS {} not found in registry", fips)


def update_county_population(fips: str, population: int) -> None:
    """
    Update population for the given county in the registry JSON.

    Args:
        fips: 5-digit county FIPS code.
        population: Census population count (non-negative).
    """
    fips = str(fips).strip().zfill(5)
    if population < 0:
        raise ValueError("population must be non-negative")

    raw = _load_raw()
    counties_list = raw.get("counties") or []
    for item in counties_list:
        if str(item.get("fips", "")).strip().zfill(5) == fips:
            item["population"] = population
            raw["counties"] = counties_list
            _save_raw(raw)
            _CACHE = None
            logger.info("Updated population for FIPS {} -> {}", fips, population)
            return

    logger.warning("County FIPS {} not found in registry", fips)


def update_county_record_count(fips: str, category: str, count: int) -> None:
    """
    Update record_count for the given county and category in the registry JSON.

    Args:
        fips: 5-digit county FIPS code.
        category: Document category (e.g. demographics, property).
        count: Number of documents for that category.
    """
    fips = str(fips).strip().zfill(5)
    raw = _load_raw()
    counties_list = raw.get("counties") or []
    for item in counties_list:
        if str(item.get("fips", "")).strip().zfill(5) == fips:
            if "record_count" not in item or not isinstance(item["record_count"], dict):
                item["record_count"] = {}
            item["record_count"][category] = count
            raw["counties"] = counties_list
            _save_raw(raw)
            _CACHE = None
            logger.debug("Updated record_count for FIPS {} category {} -> {}", fips, category, count)
            return

    logger.warning("County FIPS {} not found in registry", fips)
