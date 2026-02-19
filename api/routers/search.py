"""Search API routes for RAG system."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline, RAGResponse

router = APIRouter()

# Auth stub: placeholder for future API key check
# DP-100: Security - Authentication gate before RAG inference
def _auth_stub() -> bool:
    """Placeholder for future API key / auth check. Always returns True for now."""
    return True


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1, description="Search query")
    fips: Optional[str] = Field(None, description="County FIPS code (5 digits)")
    category: Optional[str] = Field(None, description="Document category filter")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


@router.post("/", response_model=RAGResponse)
async def search(request: SearchRequest) -> RAGResponse:
    """
    Search across all county data using RAG pipeline.
    Returns grounded answer with source citations.
    """
    if not _auth_stub():
        logger.warning("Search rejected: auth stub returned False")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    try:
        pipeline = RAGPipeline()
        response = pipeline.query(
            user_query=request.query,
            fips=request.fips,
            category=request.category,
            top_k=request.top_k,
        )

        logger.info(
            "Search complete: query='{}', fips={}, category={}, confidence={:.2f}",
            request.query[:50],
            request.fips,
            request.category,
            response.confidence,
        )

        return response
    except Exception as e:
        logger.exception("Search failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e!s}",
        )


@router.get("/semantic")
async def semantic_search(
    q: str = Query(..., min_length=1, description="Search query"),
    fips: Optional[str] = Query(None, description="County FIPS code"),
    category: Optional[str] = Query(None, description="Document category filter"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
) -> RAGResponse:
    """
    Semantic search endpoint (GET version of POST /search).
    Returns grounded answer with source citations.
    """
    if not _auth_stub():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    try:
        pipeline = RAGPipeline()
        response = pipeline.query(
            user_query=q,
            fips=fips,
            category=category,
            top_k=top_k,
        )

        return response
    except Exception as e:
        logger.exception("Semantic search failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e!s}",
        )
