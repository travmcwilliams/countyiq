"""Legal data API routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/")
async def list_legal_documents(
    county_fips: Optional[str] = Query(None, description="County FIPS code"),
    document_type: Optional[str] = Query(None, description="Document type"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """List legal documents."""
    return {
        "documents": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{document_id}")
async def get_legal_document(document_id: str) -> None:
    """Get legal document by ID."""
    raise HTTPException(status_code=404, detail="Document not found")
