"""
Tests for structured data ingestion (CSV and API).
All external calls mocked; no network or file I/O.
"""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.schemas.document import ContentType, DocumentCategory
from pipelines.ingest.structured_processor import StructuredProcessor


def _mock_county():
    from data.schemas.county import County, CrawlStatus, DataSources
    return County(
        fips="01001",
        county_name="Autauga County",
        state_name="Alabama",
        state_abbr="AL",
        data_sources=DataSources(),
        crawl_status=CrawlStatus(),
    )


class TestStructuredProcessorInit:
    def test_init(self) -> None:
        p = StructuredProcessor()
        assert p is not None


class TestDecodeCsvBytes:
    def test_utf8_decode(self) -> None:
        p = StructuredProcessor()
        out = p._decode_csv_bytes("col1,col2\na,b".encode("utf-8"))
        assert "col1" in out and "a" in out

    def test_latin1_fallback(self) -> None:
        p = StructuredProcessor()
        # Bytes that are invalid UTF-8 but valid latin-1
        raw = "name,value\nrésumé,1".encode("latin-1")
        out = p._decode_csv_bytes(raw)
        assert "name" in out
        assert "value" in out or "1" in out


class TestInferSchema:
    def test_infer_schema_string_columns(self) -> None:
        p = StructuredProcessor()
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        schema = p.infer_schema(df)
        assert schema["a"] == "string"
        assert schema["b"] == "string"

    def test_infer_schema_mixed_types(self) -> None:
        p = StructuredProcessor()
        df = pd.DataFrame({
            "id": [1, 2],
            "score": [1.5, 2.5],
            "name": ["a", "b"],
        })
        schema = p.infer_schema(df)
        assert schema["id"] == "integer"
        assert schema["score"] == "float"
        assert schema["name"] == "string"


class TestNormalizeRecord:
    def test_normalize_strips_whitespace(self) -> None:
        p = StructuredProcessor()
        out = p.normalize_record(
            {" name ": "  Jane  ", "value": " 42 "},
            "01001",
            "Autauga County",
            "al",
        )
        assert out["county_name"] == "Autauga County"
        assert out["state_abbr"] == "AL"
        # Keys are stripped, values are stripped
        assert out.get("name") == "Jane"
        assert out.get("value") == "42"

    def test_normalize_missing_values(self) -> None:
        p = StructuredProcessor()
        out = p.normalize_record(
            {"a": None, "b": float("nan")},
            "01001",
            "Test",
            "AL",
        )
        assert out["fips"] == "01001"
        assert out["county_name"] == "Test"

    def test_normalize_type_conversion(self) -> None:
        p = StructuredProcessor()
        out = p.normalize_record(
            {"x": 1, "y": 2.5, "z": "  text  "},
            "06001",
            "Alameda",
            "CA",
        )
        assert out.get("x") == 1
        assert out.get("y") == 2.5
        assert out.get("z") == "text"


class TestProcessCsv:
    @patch("pipelines.ingest.structured_processor.get_county")
    def test_process_csv_produces_documents(self, mock_get_county: MagicMock) -> None:
        mock_get_county.return_value = _mock_county()
        p = StructuredProcessor()
        csv_bytes = b"parcel_id,owner,value\nP001,Alice,100000\nP002,Bob,200000"
        docs = p.process_csv(
            source_url="https://example.com/parcels.csv",
            fips="01001",
            category=DocumentCategory.property,
            csv_bytes=csv_bytes,
        )
        assert len(docs) == 2
        assert docs[0].category == DocumentCategory.property
        assert docs[0].content_type == ContentType.structured
        assert "parcel_id" in docs[0].raw_content or "P001" in docs[0].raw_content
        assert docs[0].metadata["row_index"] == 0
        assert docs[0].metadata["category"] == "property"

    @patch("pipelines.ingest.structured_processor.get_county")
    def test_process_csv_duplicate_rows_skipped(self, mock_get_county: MagicMock) -> None:
        mock_get_county.return_value = _mock_county()
        p = StructuredProcessor()
        csv_bytes = b"id,name\n1,Alice\n1,Alice"
        docs = p.process_csv(
            source_url="https://example.com/dup.csv",
            fips="01001",
            category=DocumentCategory.tax,
            csv_bytes=csv_bytes,
        )
        assert len(docs) == 1

    @patch("pipelines.ingest.structured_processor.get_county")
    def test_process_csv_column_map(self, mock_get_county: MagicMock) -> None:
        mock_get_county.return_value = _mock_county()
        p = StructuredProcessor()
        csv_bytes = b"ParcelID,OwnerName\nP1,Jane"
        docs = p.process_csv(
            source_url="https://example.com/assess.csv",
            fips="01001",
            category=DocumentCategory.property,
            csv_bytes=csv_bytes,
            column_map={"ParcelID": "parcel_id", "OwnerName": "owner_name"},
        )
        assert len(docs) == 1
        assert "parcel_id" in docs[0].metadata or "parcel_id" in docs[0].processed_content

    @patch("pipelines.ingest.structured_processor.get_county")
    def test_process_csv_empty_returns_empty_list(self, mock_get_county: MagicMock) -> None:
        mock_get_county.return_value = _mock_county()
        p = StructuredProcessor()
        csv_bytes = b"a,b\n"
        docs = p.process_csv(
            source_url="https://example.com/empty.csv",
            fips="01001",
            category=DocumentCategory.permits,
            csv_bytes=csv_bytes,
        )
        assert len(docs) == 0


class TestProcessApi:
    @patch("pipelines.ingest.structured_processor.get_county")
    def test_process_api_list_of_dicts(self, mock_get_county: MagicMock) -> None:
        mock_get_county.return_value = _mock_county()
        p = StructuredProcessor()
        response = [
            {"NAME": "Autauga County", "P1_001N": 58805},
            {"NAME": "Other", "P1_001N": 100},
        ]
        docs = p.process_api(
            source_url="https://api.census.gov/...",
            fips="01001",
            category=DocumentCategory.demographics,
            response_json=response,
        )
        assert len(docs) == 2
        assert docs[0].category == DocumentCategory.demographics
        assert docs[0].content_type == ContentType.structured
        assert docs[0].metadata["row_index"] == 0

    @patch("pipelines.ingest.structured_processor.get_county")
    def test_process_api_record_path(self, mock_get_county: MagicMock) -> None:
        mock_get_county.return_value = _mock_county()
        p = StructuredProcessor()
        response = {"data": {"rows": [{"id": 1}, {"id": 2}]}}
        docs = p.process_api(
            source_url="https://api.example.com/data",
            fips="01001",
            category=DocumentCategory.permits,
            response_json=response,
            record_path="data.rows",
        )
        assert len(docs) == 2
        assert docs[0].metadata["row_index"] == 0

    @patch("pipelines.ingest.structured_processor.get_county")
    def test_process_api_duplicate_records_skipped(self, mock_get_county: MagicMock) -> None:
        mock_get_county.return_value = _mock_county()
        p = StructuredProcessor()
        response = [{"x": 1}, {"x": 1}]
        docs = p.process_api(
            source_url="https://example.com",
            fips="01001",
            category=DocumentCategory.tax,
            response_json=response,
        )
        assert len(docs) == 1

    @patch("pipelines.ingest.structured_processor.get_county")
    def test_process_api_single_dict(self, mock_get_county: MagicMock) -> None:
        mock_get_county.return_value = _mock_county()
        p = StructuredProcessor()
        response = {"NAME": "Autauga", "pop": 58805}
        docs = p.process_api(
            source_url="https://example.com",
            fips="01001",
            category=DocumentCategory.demographics,
            response_json=response,
        )
        assert len(docs) == 1
        assert docs[0].metadata["category"] == "demographics"


class TestCountySchemaVersionAndRecordCount:
    def test_county_has_schema_version_and_record_count(self) -> None:
        from data.schemas.county import County, CrawlStatus, DataSources
        c = County(
            fips="01001",
            county_name="Autauga",
            state_name="Alabama",
            state_abbr="AL",
            data_sources=DataSources(),
            crawl_status=CrawlStatus(),
        )
        assert c.schema_version == "1.0"
        assert c.record_count == {}
        c2 = County(
            schema_version="2.0",
            fips="06001",
            county_name="Alameda",
            state_name="California",
            state_abbr="CA",
            record_count={"demographics": 5},
            data_sources=DataSources(),
            crawl_status=CrawlStatus(),
        )
        assert c2.schema_version == "2.0"
        assert c2.record_count == {"demographics": 5}
