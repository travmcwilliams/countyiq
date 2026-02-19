"""
Upload API routes for user document ingestion.
POST /upload: multipart file upload, process via UploadProcessor, validate via DocumentValidator.
"""

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, status, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from data.schemas.document import DocumentCategory
from pipelines.ingest.upload_processor import UploadProcessor
from pipelines.ingest.validation import BatchValidationResult, DocumentValidator

router = APIRouter()

# Auth stub: placeholder for future API key check
# DP-100: Security - Authentication gate before data ingestion
def _auth_stub() -> bool:
    """Placeholder for future API key / auth check. Always returns True for now."""
    return True


class UploadResponse(BaseModel):
    """Response model for POST /upload."""

    files_processed: int = Field(..., description="Number of files processed")
    documents_created: int = Field(..., description="Total CountyDocuments created")
    validation_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="total, valid, invalid from BatchValidationResult",
    )
    quality_scores: list[float] = Field(
        default_factory=list,
        description="Quality score per document (0.0-1.0)",
    )
    validation_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-document validation (is_valid, errors, warnings, doc_id)",
    )


@router.post("/", response_model=UploadResponse)
async def upload(
    files: list[UploadFile] = File(...),
    fips: str | None = Form(None),
    category: str | None = Form(None),
) -> UploadResponse:
    """
    Accept multipart file upload(s), process via UploadProcessor, validate via DocumentValidator.
    Returns files processed, documents created, validation summary, and quality scores.
    """
    if not _auth_stub():
        logger.warning("Upload rejected: auth stub returned False")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required",
        )

    doc_category = DocumentCategory.user_upload
    if category:
        try:
            doc_category = DocumentCategory(category)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid category: {category}",
            )

    processor = UploadProcessor()
    validator = DocumentValidator()
    all_docs: list[Any] = []
    all_results: list[Any] = []

    for upload_file in files:
        try:
            file_bytes = await upload_file.read()
            filename = upload_file.filename or "unknown"
            logger.info("Processing upload: {} ({} bytes)", filename, len(file_bytes))

            docs = processor.process(
                file_bytes=file_bytes,
                filename=filename,
                fips=fips.strip() if fips and str(fips).strip() else None,
                category=doc_category,
            )
            all_docs.extend(docs)

            if docs:
                batch = validator.validate_batch(docs)
                all_results.extend(batch.results)
        except Exception as e:
            logger.exception("Upload processing failed for {}: {}", upload_file.filename, e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processing failed: {e!s}",
            )

    if not all_docs:
        return UploadResponse(
            files_processed=len(files),
            documents_created=0,
            validation_summary={"total": 0, "valid": 0, "invalid": 0},
            quality_scores=[],
            validation_results=[],
        )

    batch_result = validator.validate_batch(all_docs)
    validation_summary = {
        "total": batch_result.total,
        "valid": batch_result.valid,
        "invalid": batch_result.invalid,
    }
    quality_scores = [r.quality_score for r in batch_result.results]
    validation_results = [
        {
            "is_valid": r.is_valid,
            "errors": r.errors,
            "warnings": r.warnings,
            "doc_id": str(r.doc_id),
            "quality_score": r.quality_score,
        }
        for r in batch_result.results
    ]

    logger.info(
        "Upload complete: {} files, {} documents, {} valid",
        len(files),
        len(all_docs),
        batch_result.valid,
    )

    return UploadResponse(
        files_processed=len(files),
        documents_created=len(all_docs),
        validation_summary=validation_summary,
        quality_scores=quality_scores,
        validation_results=validation_results,
    )
