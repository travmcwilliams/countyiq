"""Search API routes for RAG system."""

import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from data.schemas.document import DocumentCategory
from rag.pipeline import RAGPipeline, RAGResponse

router = APIRouter()

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return singleton RAG pipeline (avoids loading Embedder/model on every request)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

# Security: validated filter values only (OData injection prevention)
FIPS_PATTERN = re.compile(r"^\d{5}$")
VALID_CATEGORIES = frozenset(e.value for e in DocumentCategory)

# Auth stub: placeholder for future API key check
# DP-100: Security - Authentication gate before RAG inference
def _auth_stub() -> bool:
    """Placeholder for future API key / auth check. Always returns True for now."""
    return True


class SearchRequest(BaseModel):
    """Search request model. fips and category are validated to prevent OData injection."""

    query: str = Field(..., min_length=1, description="Search query")
    fips: Optional[str] = Field(None, description="County FIPS code (5 digits)")
    category: Optional[str] = Field(None, description="Document category filter")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")

    @field_validator("fips")
    @classmethod
    def fips_five_digits(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        if not FIPS_PATTERN.match(v):
            raise ValueError("fips must be exactly 5 digits")
        return v

    @field_validator("category")
    @classmethod
    def category_enum(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        if v not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
            )
        return v


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
        pipeline = get_pipeline()
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
    fips: Optional[str] = Query(
        None,
        pattern=r"^\d{5}$",
        description="County FIPS code (exactly 5 digits)",
    ),
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
    if category is not None and category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"category must be one of: {', '.join(sorted(VALID_CATEGORIES))}",
        )

    try:
        pipeline = get_pipeline()
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
