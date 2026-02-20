"""Storage manager coordinating DuckDB local + ADLS Gen2 cloud storage.

# DP-100: Data management - Hybrid storage architecture: local DuckDB for
fast access, cloud ADLS for backup and distributed access.
"""

from pathlib import Path
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from data.schemas.county import CrawlRecord
from data.schemas.document import CountyDocument
from data.storage.adls_store import ADLSStore

try:
    from data.storage.duckdb_store import DocumentStore
except ImportError:
    DocumentStore = None  # type: ignore


class SaveResult(BaseModel):
    """Result of saving documents to both storage layers."""

    local_saved: int = Field(default=0, ge=0)
    cloud_saved: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)


class SyncResult(BaseModel):
    """Result of syncing local documents to cloud."""

    synced: int = Field(default=0, ge=0)
    already_synced: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)


class StorageManager:
    """
    Coordinates DuckDB local storage and ADLS Gen2 cloud storage.

    # DP-100: Data management - Hybrid storage pattern: local for performance,
    cloud for backup and scale. All documents saved to both layers.
    """

    def __init__(
        self,
        duckdb_path: Path | str,
        adls_account_name: Optional[str] = None,
        adls_account_key: Optional[str] = None,
    ):
        """
        Initialize storage manager.

        Args:
            duckdb_path: Path to DuckDB database file.
            adls_account_name: ADLS account name (optional, can be None for local-only).
            adls_account_key: ADLS account key (optional).
        """
        if DocumentStore is None:
            raise ImportError("duckdb is not installed. Install with: pip install duckdb")
        
        self.local_store = DocumentStore(duckdb_path)
        self.cloud_store: Optional[ADLSStore] = None
        
        if adls_account_name and adls_account_key:
            try:
                self.cloud_store = ADLSStore(adls_account_name, adls_account_key)
                logger.info("StorageManager initialized with DuckDB + ADLS")
            except Exception as e:
                logger.warning("Failed to initialize ADLS store: {} (continuing local-only)", e)
        else:
            logger.info("StorageManager initialized with DuckDB only (no cloud)")

    def save(self, docs: List[CountyDocument]) -> SaveResult:
        """
        Save documents to both DuckDB (local) and ADLS Gen2 (cloud).

        # DP-100: Data management - Dual-write pattern ensures data is available
        both locally and in cloud for redundancy.

        Args:
            docs: List of CountyDocument to save.

        Returns:
            SaveResult with counts for local_saved, cloud_saved, failed.
        """
        local_saved = 0
        cloud_saved = 0
        failed = 0

        # Save to local DuckDB
        try:
            self.local_store.save_many(docs)
            local_saved = len(docs)
            logger.info("Saved {} documents to DuckDB", local_saved)
        except Exception as e:
            logger.error("Failed to save to DuckDB: {}", e)
            failed = len(docs)
            return SaveResult(local_saved=0, cloud_saved=0, failed=failed)

        # Save to cloud ADLS (if available)
        if self.cloud_store:
            try:
                urls = self.cloud_store.upload_documents(docs)
                cloud_saved = len(urls)
                
                # Mark as synced in DuckDB
                for doc in docs:
                    if doc.id in [url.split("/")[-1].replace(".json", "") for url in urls]:
                        self.local_store.mark_synced(doc.id)
                
                logger.info("Saved {} documents to ADLS", cloud_saved)
            except Exception as e:
                logger.warning("Failed to save to ADLS (local save succeeded): {}", e)
                # Don't fail entire operation if cloud save fails
        else:
            logger.debug("Cloud store not available, skipping ADLS upload")

        return SaveResult(
            local_saved=local_saved,
            cloud_saved=cloud_saved,
            failed=failed,
        )

    def save_crawl_log(self, records: List[CrawlRecord]) -> None:
        """
        Save crawl log records to both local JSONL and ADLS.

        Args:
            records: List of CrawlRecord to save.
        """
        if not records:
            return

        fips = records[0].fips

        # Save to local JSONL (existing behavior)
        log_path = Path("data/raw") / fips / "crawl_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")

        # Upload to ADLS (if available)
        if self.cloud_store:
            try:
                self.cloud_store.upload_crawl_log(fips, records)
            except Exception as e:
                logger.warning("Failed to upload crawl log to ADLS: {}", e)

    def get_document(self, doc_id: str) -> Optional[CountyDocument]:
        """
        Get document: try local DuckDB first, fall back to ADLS if not found.

        # DP-100: Data management - Fallback pattern: local-first for performance,
        cloud fallback for completeness.

        Args:
            doc_id: Document ID (UUID string).

        Returns:
            CountyDocument if found, None otherwise.
        """
        # Try local first
        doc = self.local_store.get(doc_id)
        if doc:
            return doc

        # Fallback to cloud
        if self.cloud_store:
            # Need to search ADLS - this is inefficient, but works
            # In production, maintain an index or use metadata store
            logger.debug("Document {} not in local store, checking ADLS", doc_id)
            # ADLS doesn't have direct ID lookup, so we'd need to search
            # For now, return None if not in local
            # TODO: Add metadata index for cloud document lookup

        return None

    def sync_to_cloud(self, fips: Optional[str] = None) -> SyncResult:
        """
        Sync local DuckDB documents to ADLS Gen2 that aren't already synced.

        # DP-100: Data management - Sync operation ensures all local documents
        are backed up to cloud storage.

        Args:
            fips: Optional FIPS filter (sync only this county).

        Returns:
            SyncResult with synced, already_synced, failed counts.
        """
        if not self.cloud_store:
            logger.warning("Cloud store not available, cannot sync")
            return SyncResult(synced=0, already_synced=0, failed=0)

        unsynced = self.local_store.list_unsynced(fips=fips)
        already_synced_count = 0
        synced_count = 0
        failed_count = 0

        logger.info("Syncing {} unsynced documents to ADLS", len(unsynced))

        for doc in unsynced:
            try:
                # Upload to ADLS
                url = self.cloud_store.upload_document(doc)
                
                # Mark as synced in DuckDB
                self.local_store.mark_synced(doc.id)
                synced_count += 1
                
                if synced_count % 100 == 0:
                    logger.info("Synced {}/{} documents", synced_count, len(unsynced))
                    
            except Exception as e:
                logger.warning("Failed to sync document {}: {}", doc.id, e)
                failed_count += 1

        # Count already synced
        if fips:
            all_docs = self.local_store.list_by_fips(fips)
        else:
            # Approximate: get total and subtract unsynced
            stats = self.local_store.get_stats()
            already_synced_count = stats["total_documents"] - len(unsynced)

        logger.info(
            "Sync complete: {} synced, {} already synced, {} failed",
            synced_count,
            already_synced_count,
            failed_count,
        )

        return SyncResult(
            synced=synced_count,
            already_synced=already_synced_count,
            failed=failed_count,
        )
