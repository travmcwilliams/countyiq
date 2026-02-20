"""Storage layer: DuckDB local + ADLS Gen2 cloud storage.

# DP-100: Data storage - Hybrid storage architecture for CountyIQ documents.
"""

from data.storage.adls_store import ADLSStore, StorageStats

# Lazy import for duckdb_store to handle missing duckdb gracefully
try:
    from data.storage.duckdb_store import DocumentStore
except ImportError:
    DocumentStore = None  # type: ignore

from data.storage.storage_manager import SaveResult, StorageManager, SyncResult

__all__ = [
    "ADLSStore",
    "DocumentStore",
    "StorageManager",
    "StorageStats",
    "SaveResult",
    "SyncResult",
]
