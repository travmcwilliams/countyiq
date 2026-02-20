"""Tests for storage layer: DuckDB + ADLS Gen2.

# DP-100: Data storage - Comprehensive testing of hybrid storage architecture.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from data.schemas.document import ContentType, CountyDocument, DocumentCategory
from data.storage.adls_store import ADLSStore, StorageStats
from data.storage.duckdb_store import DocumentStore
from data.storage.storage_manager import SaveResult, StorageManager, SyncResult


def create_test_document(
    fips: str = "01001",
    category: DocumentCategory = DocumentCategory.property,
) -> CountyDocument:
    """Create a test CountyDocument."""
    return CountyDocument(
        fips=fips,
        county_name="Test County",
        state_abbr="AL",
        category=category,
        source_url="https://example.com/test",
        content_type=ContentType.html,
        raw_content="<html>Test content</html>",
        processed_content="Test content",
    )


# DuckDB Store Tests
def test_duckdb_store_save_and_get() -> None:
    """Test saving and retrieving documents from DuckDB."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        store = DocumentStore(db_path)
        
        doc = create_test_document()
        store.save(doc)
        
        retrieved = store.get(doc.id)
        assert retrieved is not None
        assert retrieved.fips == doc.fips
        assert retrieved.category == doc.category


def test_duckdb_store_list_by_fips() -> None:
    """Test listing documents by FIPS."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        store = DocumentStore(db_path)
        
        doc1 = create_test_document(fips="01001")
        doc2 = create_test_document(fips="01001")
        doc3 = create_test_document(fips="01003")
        
        store.save_many([doc1, doc2, doc3])
        
        docs_01001 = store.list_by_fips("01001")
        assert len(docs_01001) == 2
        
        docs_01003 = store.list_by_fips("01003")
        assert len(docs_01003) == 1


def test_duckdb_store_list_unsynced() -> None:
    """Test listing unsynced documents."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        store = DocumentStore(db_path)
        
        doc = create_test_document()
        store.save(doc)
        
        unsynced = store.list_unsynced()
        assert len(unsynced) == 1
        
        store.mark_synced(doc.id)
        unsynced_after = store.list_unsynced()
        assert len(unsynced_after) == 0


def test_duckdb_store_mark_synced() -> None:
    """Test marking documents as synced."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        store = DocumentStore(db_path)
        
        doc = create_test_document()
        store.save(doc)
        
        assert len(store.list_unsynced()) == 1
        store.mark_synced(doc.id)
        assert len(store.list_unsynced()) == 0


# ADLS Store Tests (mocked)
def test_adls_store_upload_document(mocker) -> None:
    """Test uploading a document to ADLS (mocked)."""
    mock_service_client = MagicMock()
    mock_file_system = MagicMock()
    mock_file_client = MagicMock()
    
    mock_service_client.get_file_system_client.return_value = mock_file_system
    mock_file_system.get_file_client.return_value = mock_file_client
    
    with patch("data.storage.adls_store.DataLakeServiceClient", return_value=mock_service_client):
        store = ADLSStore("testaccount", "testkey")
        doc = create_test_document()
        
        url = store.upload_document(doc)
        
        assert url.startswith("https://testaccount.dfs.core.windows.net")
        mock_file_client.upload_data.assert_called_once()


def test_adls_store_upload_documents_batch(mocker) -> None:
    """Test batch uploading documents to ADLS."""
    mock_service_client = MagicMock()
    mock_file_system = MagicMock()
    mock_file_client = MagicMock()
    
    mock_service_client.get_file_system_client.return_value = mock_file_system
    mock_file_system.get_file_client.return_value = mock_file_client
    
    with patch("data.storage.adls_store.DataLakeServiceClient", return_value=mock_service_client):
        store = ADLSStore("testaccount", "testkey")
        docs = [create_test_document() for _ in range(3)]
        
        urls = store.upload_documents(docs)
        
        assert len(urls) == 3
        assert mock_file_client.upload_data.call_count == 3


def test_adls_store_download_document(mocker) -> None:
    """Test downloading a document from ADLS."""
    mock_service_client = MagicMock()
    mock_file_system = MagicMock()
    mock_file_client = MagicMock()
    mock_download = MagicMock()
    
    doc = create_test_document()
    doc_json = doc.model_dump_json()
    
    mock_service_client.get_file_system_client.return_value = mock_file_system
    mock_file_system.get_file_client.return_value = mock_file_client
    mock_file_client.download_file.return_value = mock_download
    mock_download.readall.return_value = doc_json.encode("utf-8")
    
    with patch("data.storage.adls_store.DataLakeServiceClient", return_value=mock_service_client):
        store = ADLSStore("testaccount", "testkey")
        
        retrieved = store.download_document(doc.fips, doc.category.value, str(doc.id))
        
        assert retrieved is not None
        assert retrieved.fips == doc.fips


def test_adls_store_download_not_found(mocker) -> None:
    """Test downloading non-existent document returns None."""
    mock_service_client = MagicMock()
    mock_file_system = MagicMock()
    mock_file_client = MagicMock()
    
    mock_service_client.get_file_system_client.return_value = mock_file_system
    mock_file_system.get_file_client.return_value = mock_file_client
    mock_file_client.download_file.side_effect = Exception("Not found")
    
    with patch("data.storage.adls_store.DataLakeServiceClient", return_value=mock_service_client):
        store = ADLSStore("testaccount", "testkey")
        
        retrieved = store.download_document("01001", "property", str(uuid4()))
        assert retrieved is None


# StorageManager Tests
def test_storage_manager_save_to_both(mocker) -> None:
    """Test StorageManager saves to both DuckDB and ADLS."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        
        mock_adls = MagicMock()
        mock_adls.upload_documents.return_value = ["url1", "url2"]
        
        with patch("data.storage.storage_manager.ADLSStore", return_value=mock_adls):
            manager = StorageManager(
                duckdb_path=db_path,
                adls_account_name="testaccount",
                adls_account_key="testkey",
            )
            
            docs = [create_test_document() for _ in range(2)]
            result = manager.save(docs)
            
            assert result.local_saved == 2
            assert result.cloud_saved == 2
            assert result.failed == 0
            
            # Verify DuckDB has documents
            local_docs = manager.local_store.list_by_fips("01001")
            assert len(local_docs) == 2


def test_storage_manager_save_local_only() -> None:
    """Test StorageManager saves locally when ADLS not configured."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        
        manager = StorageManager(duckdb_path=db_path)
        
        docs = [create_test_document()]
        result = manager.save(docs)
        
        assert result.local_saved == 1
        assert result.cloud_saved == 0  # No cloud store


def test_storage_manager_get_document_local() -> None:
    """Test getting document from local DuckDB."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        
        manager = StorageManager(duckdb_path=db_path)
        
        doc = create_test_document()
        manager.save([doc])
        
        retrieved = manager.get_document(str(doc.id))
        assert retrieved is not None
        assert retrieved.id == doc.id


def test_storage_manager_sync_to_cloud(mocker) -> None:
    """Test syncing unsynced documents to cloud."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        
        mock_adls = MagicMock()
        mock_adls.upload_document.return_value = "https://test.com/doc.json"
        
        with patch("data.storage.storage_manager.ADLSStore", return_value=mock_adls):
            manager = StorageManager(
                duckdb_path=db_path,
                adls_account_name="testaccount",
                adls_account_key="testkey",
            )
            
            # Save documents (they'll be unsynced)
            docs = [create_test_document() for _ in range(3)]
            manager.save(docs)
            
            # Sync to cloud
            result = manager.sync_to_cloud()
            
            assert result.synced == 3
            assert result.failed == 0
            
            # Verify all are marked as synced
            unsynced = manager.local_store.list_unsynced()
            assert len(unsynced) == 0


def test_storage_manager_sync_by_fips(mocker) -> None:
    """Test syncing documents filtered by FIPS."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        
        mock_adls = MagicMock()
        mock_adls.upload_document.return_value = "https://test.com/doc.json"
        
        with patch("data.storage.storage_manager.ADLSStore", return_value=mock_adls):
            manager = StorageManager(
                duckdb_path=db_path,
                adls_account_name="testaccount",
                adls_account_key="testkey",
            )
            
            # Save documents for different counties
            doc1 = create_test_document(fips="01001")
            doc2 = create_test_document(fips="01003")
            manager.save([doc1, doc2])
            
            # Sync only 01001
            result = manager.sync_to_cloud(fips="01001")
            
            assert result.synced == 1
            assert manager.local_store.get(doc1.id) is not None
            # doc2 should still be unsynced
            unsynced = manager.local_store.list_unsynced()
            assert len(unsynced) == 1


def test_storage_manager_save_crawl_log(mocker) -> None:
    """Test saving crawl log records."""
    from data.schemas.county import CrawlRecord
    
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        
        mock_adls = MagicMock()
        
        with patch("data.storage.storage_manager.ADLSStore", return_value=mock_adls):
            manager = StorageManager(
                duckdb_path=db_path,
                adls_account_name="testaccount",
                adls_account_key="testkey",
            )
            
            records = [
                CrawlRecord(fips="01001", category="property", success=True, record_count=5),
            ]
            
            manager.save_crawl_log(records)
            
            # Verify ADLS upload was called
            mock_adls.upload_crawl_log.assert_called_once_with("01001", records)


def test_storage_stats_model() -> None:
    """Test StorageStats Pydantic model."""
    stats = StorageStats(
        total_files=100,
        total_size_gb=1.5,
        files_by_category={"property": 50, "legal": 50},
    )
    assert stats.total_files == 100
    assert stats.total_size_gb == 1.5
