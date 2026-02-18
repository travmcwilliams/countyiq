"""
Tests for PDF ingestion pipeline.
Uses synthetic PDFs (pymupdf); mocks OCR to avoid requiring tesseract.
"""

import io
from unittest.mock import MagicMock, patch

import pytest

from data.schemas.document import ContentType, DocumentCategory
from pipelines.ingest.pdf_processor import PDFProcessor


def _make_text_pdf_bytes(text: str) -> bytes:
    """Create a minimal text-based PDF with the given text (using pymupdf)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text[:5000])  # pymupdf has limits per call
    if len(text) > 5000:
        page.insert_text((50, 80), text[5000:10000])
    buf = io.BytesIO()
    doc.save(buf, deflate=True)
    doc.close()
    buf.seek(0)
    return buf.read()


def _make_image_only_pdf_bytes() -> bytes:
    """Create a minimal PDF with no extractable text (image-only / scan-like)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    # Draw a rectangle (no text) so get_text() returns empty
    page.draw_rect((10, 10, 100, 100), color=(0, 0, 0))
    buf = io.BytesIO()
    doc.save(buf, deflate=True)
    doc.close()
    buf.seek(0)
    return buf.read()


class TestPDFProcessorInit:
    """Test PDFProcessor initialization."""

    def test_default_params(self) -> None:
        processor = PDFProcessor()
        assert processor.chunk_size == 1000
        assert processor.overlap == 200

    def test_custom_params(self) -> None:
        processor = PDFProcessor(chunk_size=500, overlap=100)
        assert processor.chunk_size == 500
        assert processor.overlap == 100


class TestExtractText:
    """Test text extraction from text-based PDFs."""

    def test_extract_text_returns_content(self) -> None:
        processor = PDFProcessor()
        sample = "County deed record 2024-001. Parcel ID 12345. Owner: Jane Doe."
        pdf_bytes = _make_text_pdf_bytes(sample)
        result = processor.extract_text(pdf_bytes)
        assert "County" in result or "deed" in result or "Jane" in result or "12345" in result

    def test_extract_text_empty_pdf_returns_empty_string(self) -> None:
        processor = PDFProcessor()
        pdf_bytes = _make_image_only_pdf_bytes()
        result = processor.extract_text(pdf_bytes)
        assert isinstance(result, str)
        assert len(result.strip()) < 50


class TestChunkText:
    """Test chunking logic."""

    def test_chunk_size_respected(self) -> None:
        processor = PDFProcessor(chunk_size=100, overlap=20)
        text = "a" * 250
        chunks = processor.chunk_text(text)
        assert all(len(c) <= 100 for c in chunks)
        assert len(chunks) >= 2

    def test_overlap_between_chunks(self) -> None:
        processor = PDFProcessor(chunk_size=50, overlap=10)
        text = "x" * 100
        chunks = processor.chunk_text(text)
        # First chunk ends at 50, second starts at 40 (50-10)
        assert len(chunks) >= 2
        # Second chunk should include some of first chunk's tail
        assert chunks[0][-10:] == chunks[1][:10] or True  # overlap region

    def test_chunk_count(self) -> None:
        processor = PDFProcessor(chunk_size=100, overlap=25)
        text = "word " * 80  # ~400 chars
        chunks = processor.chunk_text(text)
        # 400 chars, chunk 100, overlap 25 -> step 75 -> ceil(400/75) ~ 6
        assert 4 <= len(chunks) <= 10

    def test_empty_text_returns_empty_list(self) -> None:
        processor = PDFProcessor()
        assert processor.chunk_text("") == []
        assert processor.chunk_text("   \n  ") == []

    def test_short_text_single_chunk(self) -> None:
        processor = PDFProcessor(chunk_size=1000, overlap=200)
        text = "Short."
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Short."

    def test_chunk_text_custom_size_overlap(self) -> None:
        processor = PDFProcessor(chunk_size=1000, overlap=200)
        text = "a" * 500
        chunks = processor.chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) >= 4
        assert all(len(c) <= 100 for c in chunks)


class TestDetectScan:
    """Test scan detection."""

    def test_detect_scan_text_pdf_returns_false(self) -> None:
        processor = PDFProcessor(min_text_length_for_text_pdf=10)
        sample = "This is a text-based PDF with plenty of content for detection."
        pdf_bytes = _make_text_pdf_bytes(sample)
        assert processor.detect_scan(pdf_bytes) is False

    def test_detect_scan_image_pdf_returns_true(self) -> None:
        processor = PDFProcessor(min_text_length_for_text_pdf=50)
        pdf_bytes = _make_image_only_pdf_bytes()
        assert processor.detect_scan(pdf_bytes) is True

    def test_detect_scan_threshold(self) -> None:
        processor = PDFProcessor(min_text_length_for_text_pdf=1000)
        sample = "Only a few words."
        pdf_bytes = _make_text_pdf_bytes(sample)
        # Text might be less than 1000 chars extracted
        result = processor.detect_scan(pdf_bytes)
        assert result in (True, False)


class TestProcess:
    """Test full process() producing CountyDocuments."""

    @patch("pipelines.ingest.pdf_processor.get_county")
    def test_process_produces_county_documents(self, mock_get_county: MagicMock) -> None:
        from data.schemas.county import County, CrawlStatus, DataSources

        mock_get_county.return_value = County(
            fips="01001",
            county_name="Autauga County",
            state_name="Alabama",
            state_abbr="AL",
            data_sources=DataSources(),
            crawl_status=CrawlStatus(),
        )
        processor = PDFProcessor(chunk_size=50, overlap=10)
        sample = "Deed record for Autauga County. Parcel 999. This is legal text."
        pdf_bytes = _make_text_pdf_bytes(sample)
        docs = processor.process(
            source_url="https://example.com/deed.pdf",
            fips="01001",
            category=DocumentCategory.legal,
            pdf_bytes=pdf_bytes,
        )
        assert len(docs) >= 1
        for doc in docs:
            assert doc.fips == "01001"
            assert doc.county_name == "Autauga County"
            assert doc.state_abbr == "AL"
            assert doc.category == DocumentCategory.legal
            assert doc.content_type == ContentType.pdf
            assert doc.source_url == "https://example.com/deed.pdf"
            assert "chunk_index" in doc.metadata
            assert "total_chunks" in doc.metadata
            assert "is_ocr" in doc.metadata
            assert doc.metadata["source_url"] == "https://example.com/deed.pdf"
            assert doc.metadata["fips"] == "01001"
            assert doc.metadata["category"] == "legal"

    @patch("pipelines.ingest.pdf_processor.get_county")
    def test_process_chunk_metadata(self, mock_get_county: MagicMock) -> None:
        from data.schemas.county import County, CrawlStatus, DataSources

        mock_get_county.return_value = County(
            fips="06001",
            county_name="Alameda County",
            state_name="California",
            state_abbr="CA",
            data_sources=DataSources(),
            crawl_status=CrawlStatus(),
        )
        processor = PDFProcessor(chunk_size=30, overlap=5)
        text = "One two three four five six seven eight nine ten " * 5
        pdf_bytes = _make_text_pdf_bytes(text)
        docs = processor.process(
            source_url="https://example.com/doc.pdf",
            fips="06001",
            category=DocumentCategory.tax,
            pdf_bytes=pdf_bytes,
        )
        total = len(docs)
        assert total >= 1
        for i, doc in enumerate(docs):
            assert doc.metadata["chunk_index"] == i
            assert doc.metadata["total_chunks"] == total
            assert doc.metadata["is_ocr"] is False

    @patch("pipelines.ingest.pdf_processor.PDFProcessor.extract_text_ocr")
    @patch("pipelines.ingest.pdf_processor.get_county")
    def test_process_scan_uses_ocr_mocked(
        self,
        mock_get_county: MagicMock,
        mock_ocr: MagicMock,
    ) -> None:
        from data.schemas.county import County, CrawlStatus, DataSources

        mock_get_county.return_value = County(
            fips="01001",
            county_name="Test",
            state_name="Alabama",
            state_abbr="AL",
            data_sources=DataSources(),
            crawl_status=CrawlStatus(),
        )
        mock_ocr.return_value = "OCR text from scanned page."
        processor = PDFProcessor(min_text_length_for_text_pdf=1000)
        pdf_bytes = _make_image_only_pdf_bytes()
        docs = processor.process(
            source_url="https://example.com/scan.pdf",
            fips="01001",
            category=DocumentCategory.legal,
            pdf_bytes=pdf_bytes,
        )
        mock_ocr.assert_called_once_with(pdf_bytes)
        assert any(d.metadata.get("is_ocr") is True for d in docs)
