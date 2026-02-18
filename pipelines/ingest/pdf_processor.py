"""
PDF ingestion pipeline for CountyIQ.
Handles text-based and scanned (image) PDFs; produces chunked CountyDocuments for RAG.
"""

from typing import Any

from loguru import logger

from data.schemas.document import ContentType, CountyDocument, DocumentCategory
from data.schemas.registry_loader import get_county


# DP-100: Feature engineering - Chunking text for RAG retrieval and embedding
class PDFProcessor:
    """
    Process county PDFs (deeds, court records, zoning, tax, permits) into
    CountyDocument chunks for indexing and RAG.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        min_text_length_for_text_pdf: int = 50,
    ) -> None:
        """
        Initialize PDF processor.

        Args:
            chunk_size: Target characters per chunk (default 1000).
            overlap: Overlap between consecutive chunks (default 200).
            min_text_length_for_text_pdf: If extract_text returns fewer chars, treat as scan (default 50).
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_text_length_for_text_pdf = min_text_length_for_text_pdf

    def extract_text(self, pdf_bytes: bytes) -> str:
        """
        Extract text from text-based PDF using pymupdf (fitz).

        Args:
            pdf_bytes: Raw PDF file bytes.

        Returns:
            Extracted text, or empty string if extraction fails.
        """
        try:
            import fitz  # pymupdf

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            parts: list[str] = []
            for page in doc:
                parts.append(page.get_text())
            page_count = len(doc)
            doc.close()
            text = "\n".join(parts).strip()
            logger.debug("Extracted {} chars from {} pages (text)", len(text), page_count)
            return text
        except Exception as e:
            logger.warning("Text extraction failed: {}", e)
            return ""

    def extract_text_ocr(self, pdf_bytes: bytes) -> str:
        """
        Extract text from scanned PDF using OCR (pytesseract + pdf2image).

        Args:
            pdf_bytes: Raw PDF file bytes.

        Returns:
            OCR-extracted text, or empty string if OCR fails.
        """
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(pdf_bytes)
            parts: list[str] = []
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img)
                if text.strip():
                    parts.append(text.strip())
                logger.debug("OCR page {}/{}: {} chars", i + 1, len(images), len(text))
            result = "\n\n".join(parts).strip()
            logger.info("OCR extracted {} chars from {} pages", len(result), len(images))
            return result
        except Exception as e:
            logger.warning("OCR extraction failed: {}", e)
            return ""

    def detect_scan(self, pdf_bytes: bytes) -> bool:
        """
        Return True if PDF has no (or negligible) extractable text (i.e. is a scan).

        Args:
            pdf_bytes: Raw PDF file bytes.

        Returns:
            True if PDF appears to be scanned/image-based; False if text-based.
        """
        text = self.extract_text(pdf_bytes)
        is_scan = len(text.strip()) < self.min_text_length_for_text_pdf
        if is_scan:
            logger.debug("PDF detected as scan (text length {})", len(text))
        return is_scan

    # DP-100: Chunking - Splitting documents for RAG context windows and retrieval
    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """
        Split text into overlapping chunks for RAG.

        Args:
            text: Full text to chunk.
            chunk_size: Characters per chunk (default: instance chunk_size).
            overlap: Overlap between chunks (default: instance overlap).

        Returns:
            List of text chunks (overlapping).
        """
        cs = chunk_size if chunk_size is not None else self.chunk_size
        ov = overlap if overlap is not None else self.overlap
        if ov >= cs:
            ov = max(0, cs // 2)
        text = text.strip()
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + cs
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - ov
        logger.debug("Chunked text into {} chunks (size={}, overlap={})", len(chunks), cs, ov)
        return chunks

    # DP-100: Data pipeline - Ingestion step transforming raw PDF into structured documents
    def process(
        self,
        source_url: str,
        fips: str,
        category: DocumentCategory,
        pdf_bytes: bytes,
        county_name: str | None = None,
        state_abbr: str | None = None,
    ) -> list[CountyDocument]:
        """
        Process a PDF into a list of CountyDocuments (one per chunk).

        Args:
            source_url: URL or path of the PDF.
            fips: 5-digit county FIPS code.
            category: Document category (legal, tax, zoning, etc.).
            pdf_bytes: Raw PDF bytes.
            county_name: Override county name (default: from registry).
            state_abbr: Override state abbr (default: from registry).

        Returns:
            List of CountyDocument instances (one per chunk) with metadata.
        """
        fips = str(fips).strip().zfill(5)
        county = get_county(fips)
        if county:
            cname = county_name or county.county_name
            sabbr = state_abbr or county.state_abbr
        else:
            cname = county_name or "Unknown"
            sabbr = state_abbr or ""

        # Detect scan vs text
        is_scan = self.detect_scan(pdf_bytes)
        if is_scan:
            full_text = self.extract_text_ocr(pdf_bytes)
            if not full_text.strip():
                logger.warning("OCR produced no text for {}", source_url)
                # Still produce one document with raw placeholder
                doc = CountyDocument(
                    fips=fips,
                    county_name=cname,
                    state_abbr=sabbr,
                    category=category,
                    source_url=source_url,
                    content_type=ContentType.pdf,
                    raw_content="[OCR produced no text]",
                    metadata={
                        "page_count": 0,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "is_ocr": True,
                        "source_url": source_url,
                        "fips": fips,
                        "category": category.value,
                    },
                )
                return [doc]
        else:
            full_text = self.extract_text(pdf_bytes)

        if not full_text.strip():
            logger.warning("No text extracted from {}", source_url)
            doc = CountyDocument(
                fips=fips,
                county_name=cname,
                state_abbr=sabbr,
                category=category,
                source_url=source_url,
                content_type=ContentType.pdf,
                raw_content="",
                metadata={
                    "page_count": 0,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "is_ocr": is_scan,
                    "source_url": source_url,
                    "fips": fips,
                    "category": category.value,
                },
            )
            return [doc]

        chunks = self.chunk_text(full_text)
        if not chunks:
            chunks = [full_text[: self.chunk_size]]

        documents: list[CountyDocument] = []
        for i, chunk_content in enumerate(chunks):
            meta: dict[str, Any] = {
                "page_count": 0,  # Could be set if we track pages per chunk
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_ocr": is_scan,
                "source_url": source_url,
                "fips": fips,
                "category": category.value,
            }
            doc = CountyDocument(
                fips=fips,
                county_name=cname,
                state_abbr=sabbr,
                category=category,
                source_url=source_url,
                content_type=ContentType.pdf,
                raw_content=chunk_content,
                processed_content=chunk_content.strip(),
                metadata=meta,
            )
            documents.append(doc)

        logger.info("Processed PDF into {} chunks (is_ocr={})", len(documents), is_scan)
        return documents
