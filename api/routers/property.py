"""Property data API routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/")
async def list_properties(
    county_fips: Optional[str] = Query(None, description="County FIPS code"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """List properties."""
    return {
        "properties": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{property_id}")
async def get_property(property_id: str) -> None:
    """Get property by ID."""
    raise HTTPException(status_code=404, detail="Property not found")
