"""
Structured data ingestion pipeline for CountyIQ.
Normalizes CSV and API JSON into CountyDocuments for RAG.
"""

import io
import json
from typing import Any

import pandas as pd
from loguru import logger

from data.schemas.document import ContentType, CountyDocument, DocumentCategory
from data.schemas.registry_loader import get_county


# DP-100: Data pipeline - Structured ingestion transforms heterogeneous sources into unified schema
class StructuredProcessor:
    """
    Process CSV and API JSON into CountyDocument instances.
    Each row/record becomes one document with content_type=structured.
    """

    def __init__(self) -> None:
        """Initialize structured processor."""

    def _decode_csv_bytes(self, csv_bytes: bytes) -> str:
        """Decode CSV bytes; fallback to latin-1 on UnicodeDecodeError."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return csv_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        logger.warning("CSV decode failed with common encodings, using latin-1 with errors=replace")
        return csv_bytes.decode("latin-1", errors="replace")

    # DP-100: Feature engineering - Schema inference for structured data
    def infer_schema(self, df: pd.DataFrame) -> dict[str, str]:
        """
        Return column name -> inferred type mapping.

        Args:
            df: DataFrame to infer schema from.

        Returns:
            Dict mapping column name to one of: string, integer, float, boolean, datetime.
        """
        out: dict[str, str] = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                out[str(col)] = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                out[str(col)] = "float"
            elif pd.api.types.is_bool_dtype(dtype):
                out[str(col)] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                out[str(col)] = "datetime"
            else:
                out[str(col)] = "string"
        logger.debug("Inferred schema for {} columns", len(out))
        return out

    def normalize_record(
        self,
        record: dict[str, Any],
        fips: str,
        county_name: str,
        state_abbr: str,
    ) -> dict[str, Any]:
        """
        Standardize field names, strip whitespace, convert types.

        Args:
            record: Raw record dict.
            fips: County FIPS code.
            county_name: County name.
            state_abbr: State abbreviation.

        Returns:
            Normalized record with standard fields and cleaned values.
        """
        normalized: dict[str, Any] = {
            "fips": str(fips).strip().zfill(5),
            "county_name": county_name,
            "state_abbr": state_abbr.strip().upper(),
        }
        for k, v in record.items():
            key = str(k).strip()
            if not key:
                continue
            if v is None:
                normalized[key] = None
                continue
            try:
                if isinstance(v, float) and pd.isna(v):
                    normalized[key] = None
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(v, str):
                normalized[key] = v.strip()
            elif isinstance(v, (int, float, bool)):
                normalized[key] = v
            else:
                normalized[key] = str(v).strip() if v is not None else None
        return normalized

    def process_csv(
        self,
        source_url: str,
        fips: str,
        category: DocumentCategory,
        csv_bytes: bytes,
        column_map: dict[str, str] | None = None,
    ) -> list[CountyDocument]:
        """
        Process CSV bytes into CountyDocuments (one per row).

        Args:
            source_url: URL or path of the CSV.
            fips: 5-digit county FIPS code.
            category: Document category.
            csv_bytes: Raw CSV file bytes.
            column_map: Optional rename map (original_name -> standard_name).

        Returns:
            List of CountyDocument instances; duplicates removed by row content hash.
        """
        fips = str(fips).strip().zfill(5)
        county = get_county(fips)
        county_name = county.county_name if county else "Unknown"
        state_abbr = county.state_abbr if county else ""

        text = self._decode_csv_bytes(csv_bytes)
        try:
            df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        except Exception as e:
            logger.error("CSV parse failed: {}", e)
            return []

        if column_map:
            df = df.rename(columns=column_map)

        schema = self.infer_schema(df.astype(object).replace("", None))
        seen: set[str] = set()
        documents: list[CountyDocument] = []

        for row_index, row in df.iterrows():
            record = row.to_dict()
            normalized = self.normalize_record(record, fips, county_name, state_abbr)
            raw_json = json.dumps(record, default=str)
            content_hash = str(hash(raw_json))
            if content_hash in seen:
                logger.debug("Skipping duplicate row at index {}", row_index)
                continue
            seen.add(content_hash)

            meta: dict[str, Any] = {
                **normalized,
                "schema": schema,
                "row_index": int(row_index),
                "source_url": source_url,
                "category": category.value,
            }

            doc = CountyDocument(
                fips=fips,
                county_name=county_name,
                state_abbr=state_abbr,
                category=category,
                source_url=source_url,
                content_type=ContentType.structured,
                raw_content=raw_json,
                processed_content=json.dumps(normalized, default=str),
                metadata=meta,
            )
            documents.append(doc)

        logger.info("Processed CSV into {} documents ({} duplicates skipped)", len(documents), len(df) - len(documents))
        return documents

    def process_api(
        self,
        source_url: str,
        fips: str,
        category: DocumentCategory,
        response_json: dict[str, Any] | list[Any],
        record_path: str | None = None,
    ) -> list[CountyDocument]:
        """
        Process API JSON response into CountyDocuments.

        Args:
            source_url: API endpoint URL.
            fips: 5-digit county FIPS code.
            category: Document category.
            response_json: Parsed JSON (dict or list).
            record_path: Optional JSON path to records, e.g. "results" or "data.rows".

        Returns:
            List of CountyDocument instances.
        """
        fips = str(fips).strip().zfill(5)
        county = get_county(fips)
        county_name = county.county_name if county else "Unknown"
        state_abbr = county.state_abbr if county else ""

        if record_path:
            obj: Any = response_json
            for part in record_path.split("."):
                obj = obj.get(part) if isinstance(obj, dict) else (obj[int(part)] if isinstance(obj, list) else None)
                if obj is None:
                    logger.warning("record_path {} not found in response", record_path)
                    return []
            records = obj if isinstance(obj, list) else [obj]
        else:
            if isinstance(response_json, list):
                records = response_json
            elif isinstance(response_json, dict):
                records = [response_json]
            else:
                records = []

        seen: set[str] = set()
        documents: list[CountyDocument] = []

        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                rec = {"value": rec}
            normalized = self.normalize_record(rec, fips, county_name, state_abbr)
            raw_json = json.dumps(rec, default=str)
            content_hash = str(hash(raw_json))
            if content_hash in seen:
                logger.debug("Skipping duplicate record at index {}", i)
                continue
            seen.add(content_hash)

            meta: dict[str, Any] = {
                **normalized,
                "row_index": i,
                "source_url": source_url,
                "category": category.value,
            }

            doc = CountyDocument(
                fips=fips,
                county_name=county_name,
                state_abbr=state_abbr,
                category=category,
                source_url=source_url,
                content_type=ContentType.structured,
                raw_content=raw_json,
                processed_content=json.dumps(normalized, default=str),
                metadata=meta,
            )
            documents.append(doc)

        logger.info("Processed API response into {} documents", len(documents))
        return documents
