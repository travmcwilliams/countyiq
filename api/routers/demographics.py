"""Demographics data API routes."""

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/")
async def get_demographics(
    county_fips: str = Query(..., description="County FIPS code"),
) -> None:
    """Get demographics data for a county."""
    raise HTTPException(status_code=404, detail="Demographics data not found")


@router.get("/compare")
async def compare_demographics(
    county_fips_list: str = Query(..., description="Comma-separated list of FIPS codes"),
) -> dict:
    """Compare demographics across multiple counties."""
    fips_codes = [f.strip() for f in county_fips_list.split(",")]
    return {
        "counties": fips_codes,
        "comparison": {},
    }
