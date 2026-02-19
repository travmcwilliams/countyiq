"""County registry: load list of counties for crawl orchestration.

Loads from crawlers/county_registry.json which contains all 3,235 US counties.
# DP-100: Data asset - Registry of county entities for pipeline orchestration.
"""

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class County(BaseModel):
    """Single county record (FIPS + name + state)."""

    fips: str = Field(..., min_length=5, max_length=5, description="5-digit FIPS code")
    county_name: str = Field(..., min_length=1)
    state_abbr: str = Field(..., min_length=2, max_length=2)


def get_registry_path() -> Path:
    """Return path to county registry JSON."""
    return Path(__file__).resolve().parents[2] / "crawlers" / "county_registry.json"


def load_counties(path: Path | None = None) -> List[County]:
    """
    Load all counties from registry JSON.

    Supports two formats:
    1. Array format: [{"fips": "...", "county_name": "...", "state_abbr": "..."}, ...]
    2. Object format: {"counties": [{"fips": "...", "name": "...", "state": "..."}, ...]}

    Args:
        path: Override path to JSON file. Defaults to crawlers/county_registry.json.

    Returns:
        List of County models.
    """
    p = path or get_registry_path()
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle object format with "counties" key (crawlers/county_registry.json format)
    if isinstance(data, dict) and "counties" in data:
        counties_data = data["counties"]
        # Map "name" -> "county_name" and "state" -> "state_abbr"
        normalized = []
        for item in counties_data:
            normalized.append({
                "fips": item["fips"],
                "county_name": item.get("name", item.get("county_name", "")),
                "state_abbr": item.get("state", item.get("state_abbr", "")),
            })
        return [County(**item) for item in normalized]
    
    # Handle array format (data/county_registry.json format)
    if isinstance(data, list):
        return [County(**item) for item in data]
    
    return []


def get_counties_by_state(state_abbr: str, path: Path | None = None) -> List[County]:
    """Return counties for a given state abbreviation."""
    counties = load_counties(path)
    return [c for c in counties if c.state_abbr.upper() == state_abbr.upper()]


def get_county_by_fips(fips: str, path: Path | None = None) -> County | None:
    """Return a single county by FIPS code."""
    counties = load_counties(path)
    for c in counties:
        if c.fips == fips:
            return c
    return None
