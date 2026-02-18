"""
Build crawlers/county_registry.json from US Census national_county2020.txt.
Run from project root: python scripts/build_county_registry.py
"""

import json
import re
import urllib.request
from pathlib import Path

# State FIPS/abbr to full name (Census 2020 uses 2-letter state + STATEFP)
STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
    "WY": "Wyoming", "PR": "Puerto Rico", "VI": "Virgin Islands", "GU": "Guam",
    "AS": "American Samoa", "MP": "Northern Mariana Islands",
}

CENSUS_URL = "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "crawlers" / "county_registry.json"


def fetch_census() -> str:
    """Download Census county file."""
    with urllib.request.urlopen(CENSUS_URL, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_census(content: str) -> list[dict]:
    """Parse pipe-delimited Census file; return list of county dicts for registry."""
    lines = content.strip().split("\n")
    if not lines:
        return []
    # Header: STATE|STATEFP|COUNTYFP|COUNTYNS|COUNTYNAME|CLASSFP|FUNCSTAT
    counties: list[dict] = []
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) < 5:
            continue
        state_abbr = parts[0].strip()
        state_fp = parts[1].strip()
        county_fp = parts[2].strip()
        county_name = parts[4].strip()
        fips = state_fp + county_fp
        if len(fips) != 5:
            fips = fips.zfill(5)
        state_name = STATE_NAMES.get(state_abbr, state_abbr)
        counties.append({
            "fips": fips,
            "county_name": county_name,
            "state_name": state_name,
            "state_abbr": state_abbr,
            "population": None,
            "data_sources": {
                "property": None, "legal": None, "demographics": None,
                "permits": None, "zoning": None, "courts": None, "tax": None,
            },
            "crawl_status": {
                "property": "pending", "legal": "pending", "demographics": "pending",
                "permits": "pending", "zoning": "pending", "courts": "pending", "tax": "pending",
            },
            "last_crawled": None,
        })
    return counties


def main() -> None:
    print("Fetching Census national_county2020.txt...")
    content = fetch_census()
    counties = parse_census(content)
    print(f"Parsed {len(counties)} counties.")
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "counties": counties,
        "metadata": {
            "last_updated": "2026-02-18",
            "total_counties": len(counties),
            "source": "US Census national_county2020.txt",
        },
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
