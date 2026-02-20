"""
Tests for storage layer: DuckDB + ADLS Gen2.

# DP-100: Data storage - All tests use in-memory DuckDB (:memory:) and
mocked ADLSStore; no real F: drive or Azure calls.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from data.schemas.county import CrawlRecord
from data.schemas.document import ContentType, CountyDocument, DocumentCategory
from data.storage.duckdb_store import DocumentStore
from data.storage.storage_manager import SaveResult, StorageManager, SyncResult


# In-memory DuckDB path for all tests
MEMORY_DB = ":memory:"


def make_doc(
    fips: str = "01001",
    category: DocumentCategory = DocumentCategory.property,
) -> CountyDocument:
    """Create a CountyDocument for testing."""
    return CountyDocument(
        id=uuid4(),
        fips=fips,
        county_name="Test County",
        state_abbr="AL",
        category=category,
        source_url="https://example.com/test",
        content_type=ContentType.html,
        raw_content="<p>test</p>",
        processed_content="test",
    )


# ---- DocumentStore (in-memory DuckDB only) ----


def test_duckdb_store_save_and_get() -> None:
    """Save and get document using in-memory DuckDB."""
    store = DocumentStore(MEMORY_DB)
    try:
        doc = make_doc()
        store.save(doc)
        out = store.get(doc.id)
        assert out is not None
        assert out.id == doc.id
        assert out.fips == doc.fips
    finally:
        store.close()


def test_duckdb_store_save_many() -> None:
    """Save multiple documents to in-memory DuckDB."""
    store = DocumentStore(MEMORY_DB)
    try:
        docs = [make_doc() for _ in range(3)]
        store.save_many(docs)
        for doc in docs:
            assert store.get(doc.id) is not None
    finally:
        store.close()


def test_duckdb_store_insert_or_replace_idempotent() -> None:
    """Re-saving same document (INSERT OR REPLACE) does not raise duplicate key."""
    store = DocumentStore(MEMORY_DB)
    try:
        doc = make_doc()
        store.save(doc)
        store.save(doc)
        store.save(doc)
        out = store.get(doc.id)
        assert out is not None
    finally:
        store.close()


def test_duckdb_store_list_unsynced() -> None:
    """list_unsynced returns only documents with synced_to_cloud = FALSE."""
    store = DocumentStore(MEMORY_DB)
    try:
        a, b, c = make_doc(), make_doc(), make_doc()
        store.save(a)
        store.save(b)
        store.save(c)
        unsynced = store.list_unsynced()
        assert len(unsynced) == 3
        store.mark_synced(a.id)
        unsynced = store.list_unsynced()
        assert len(unsynced) == 2
    finally:
        store.close()


def test_duckdb_store_mark_synced() -> None:
    """mark_synced updates synced_to_cloud to TRUE."""
    store = DocumentStore(MEMORY_DB)
    try:
        doc = make_doc()
        store.save(doc)
        assert len(store.list_unsynced()) == 1
        store.mark_synced(doc.id)
        assert len(store.list_unsynced()) == 0
    finally:
        store.close()


def test_duckdb_store_list_by_fips() -> None:
    """list_by_fips filters by FIPS and optional category."""
    store = DocumentStore(MEMORY_DB)
    try:
        store.save(make_doc(fips="01001"))
        store.save(make_doc(fips="01001", category=DocumentCategory.legal))
        store.save(make_doc(fips="01003"))
        by_fips = store.list_by_fips("01001")
        assert len(by_fips) == 2
        by_fips_cat = store.list_by_fips("01001", category="legal")
        assert len(by_fips_cat) == 1
    finally:
        store.close()


def test_duckdb_store_get_stats() -> None:
    """get_stats returns total and unsynced counts."""
    store = DocumentStore(MEMORY_DB)
    try:
        store.save(make_doc())
        store.save(make_doc())
        stats = store.get_stats()
        assert stats["total_documents"] == 2
        assert stats["unsynced_count"] == 2
        store.mark_synced(store.list_unsynced()[0].id)
        stats = store.get_stats()
        assert stats["unsynced_count"] == 1
    finally:
        store.close()


# ---- StorageManager with mocked ADLSStore ----


@patch("data.storage.storage_manager.ADLSStore")
def test_storage_manager_save_local_only(mock_adls_class: MagicMock) -> None:
    """Without cloud credentials, save only writes to DuckDB."""
    mock_adls_class.return_value = None  # not used; manager has no cloud
    manager = StorageManager(duckdb_path=MEMORY_DB)
    try:
        docs = [make_doc()]
        result = manager.save(docs)
        assert result.local_saved == 1
        assert result.cloud_saved == 0
    finally:
        manager.close()


@patch("data.storage.storage_manager.ADLSStore")
def test_storage_manager_save_to_both(mock_adls_class: MagicMock) -> None:
    """With mocked ADLS, save writes to DuckDB and calls upload_documents."""
    mock_adls = MagicMock()
    mock_adls.upload_documents.return_value = ["https://fake.com/1.json"]
    mock_adls_class.return_value = mock_adls
    manager = StorageManager(
        duckdb_path=MEMORY_DB,
        adls_account_name="test",
        adls_account_key="key",
    )
    try:
        docs = [make_doc()]
        result = manager.save(docs)
        assert result.local_saved == 1
        assert result.cloud_saved == 1
        mock_adls.upload_documents.assert_called_once()
    finally:
        manager.close()


@patch("data.storage.storage_manager.ADLSStore")
def test_storage_manager_get_document(mock_adls_class: MagicMock) -> None:
    """get_document returns from local DuckDB."""
    mock_adls_class.return_value = None
    manager = StorageManager(duckdb_path=MEMORY_DB)
    try:
        doc = make_doc()
        manager.local_store.save(doc)
        out = manager.get_document(str(doc.id))
        assert out is not None
        assert out.id == doc.id
    finally:
        manager.close()


@patch("data.storage.storage_manager.ADLSStore")
def test_storage_manager_sync_to_cloud(mock_adls_class: MagicMock) -> None:
    """
    Sync: insert unsynced docs into in-memory DuckDB, then sync; mock ADLS.
    """
    mock_adls = MagicMock()
    mock_adls.upload_document.return_value = "https://fake.com/doc.json"
    mock_adls_class.return_value = mock_adls

    manager = StorageManager(
        duckdb_path=MEMORY_DB,
        adls_account_name="test",
        adls_account_key="key",
    )
    manager.cloud_store = mock_adls
    try:
        # Manually insert documents as unsynced (via local_store only)
        docs = [make_doc() for _ in range(3)]
        for d in docs:
            manager.local_store.save(d)
        assert len(manager.local_store.list_unsynced()) == 3

        result = manager.sync_to_cloud()
        assert result.synced == 3
        assert result.failed == 0
        assert mock_adls.upload_document.call_count == 3
        assert len(manager.local_store.list_unsynced()) == 0
    finally:
        manager.close()


@patch("data.storage.storage_manager.ADLSStore")
def test_storage_manager_sync_by_fips(mock_adls_class: MagicMock) -> None:
    """Sync only documents for a given FIPS."""
    mock_adls = MagicMock()
    mock_adls.upload_document.return_value = "https://fake.com/doc.json"
    mock_adls_class.return_value = mock_adls

    manager = StorageManager(
        duckdb_path=MEMORY_DB,
        adls_account_name="test",
        adls_account_key="key",
    )
    manager.cloud_store = mock_adls
    try:
        manager.local_store.save(make_doc(fips="01001"))
        manager.local_store.save(make_doc(fips="01003"))
        assert len(manager.local_store.list_unsynced()) == 2

        result = manager.sync_to_cloud(fips="01001")
        assert result.synced == 1
        assert result.failed == 0
        assert mock_adls.upload_document.call_count == 1
        unsynced = manager.local_store.list_unsynced()
        assert len(unsynced) == 1
        assert unsynced[0].fips == "01003"
    finally:
        manager.close()


@patch("data.storage.storage_manager.ADLSStore")
def test_storage_manager_save_crawl_log(mock_adls_class: MagicMock) -> None:
    """save_crawl_log calls ADLS upload_crawl_log when cloud_store is set."""
    mock_adls = MagicMock()
    mock_adls.upload_crawl_log.return_value = "https://fake.com/log.jsonl"
    mock_adls_class.return_value = mock_adls

    manager = StorageManager(
        duckdb_path=MEMORY_DB,
        adls_account_name="test",
        adls_account_key="key",
    )
    manager.cloud_store = mock_adls
    try:
        records = [
            CrawlRecord(fips="01001", category="property", success=True, record_count=5),
        ]
        manager.save_crawl_log(records)
        mock_adls.upload_crawl_log.assert_called_once_with("01001", records)
    finally:
        manager.close()


# ---- ADLSStore (fully mocked at DataLakeServiceClient level) ----


@patch("data.storage.adls_store.DataLakeServiceClient")
def test_adls_store_upload_document(mock_dls_client: MagicMock) -> None:
    """ADLSStore.upload_document returns URL when client is mocked."""
    mock_fs = MagicMock()
    mock_file = MagicMock()
    mock_dls_client.return_value.get_file_system_client.return_value = mock_fs
    mock_fs.get_file_client.return_value = mock_file

    from data.storage.adls_store import ADLSStore

    store = ADLSStore("testaccount", "testkey")
    doc = make_doc()
    url = store.upload_document(doc)
    assert "testaccount" in url
    assert doc.fips in url
    mock_file.upload_data.assert_called_once()


@patch("data.storage.adls_store.DataLakeServiceClient")
def test_adls_store_upload_documents_batch(mock_dls_client: MagicMock) -> None:
    """ADLSStore.upload_documents uploads each doc and returns URLs."""
    mock_fs = MagicMock()
    mock_file = MagicMock()
    mock_dls_client.return_value.get_file_system_client.return_value = mock_fs
    mock_fs.get_file_client.return_value = mock_file

    from data.storage.adls_store import ADLSStore

    store = ADLSStore("testaccount", "testkey")
    docs = [make_doc(), make_doc()]
    urls = store.upload_documents(docs)
    assert len(urls) == 2
    assert mock_file.upload_data.call_count == 2
