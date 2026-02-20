"""Azure Data Lake Storage Gen2 store for CountyIQ documents.

# DP-100: Data storage - Cloud storage layer using Azure Data Lake Storage Gen2
for scalable, hierarchical document storage.
"""

import json
from datetime import datetime
from typing import List, Optional

from azure.core.exceptions import AzureError
from azure.storage.filedatalake import DataLakeServiceClient
from loguru import logger
from pydantic import BaseModel, Field

from data.schemas.county import CrawlRecord
from data.schemas.document import CountyDocument


class StorageStats(BaseModel):
    """Storage statistics for ADLS Gen2 account."""

    total_files: int = Field(default=0, ge=0)
    total_size_gb: float = Field(default=0.0, ge=0.0)
    files_by_category: dict[str, int] = Field(default_factory=dict)
    last_upload: Optional[datetime] = Field(default=None)


class ADLSStore:
    """
    Azure Data Lake Storage Gen2 store for CountyIQ documents.

    # DP-100: Data storage - Cloud storage using ADLS Gen2 hierarchical namespace
    for organized document storage by FIPS/category.
    """

    def __init__(self, account_name: str, account_key: str):
        """
        Initialize ADLS Gen2 store.

        Args:
            account_name: Storage account name (e.g. 'countyiqdata').
            account_key: Storage account key for authentication.
        """
        self.account_name = account_name
        self.account_key = account_key
        account_url = f"https://{account_name}.dfs.core.windows.net"
        
        try:
            self.service_client = DataLakeServiceClient(
                account_url=account_url,
                credential=account_key,
            )
            logger.info("Connected to ADLS Gen2: {}", account_name)
        except Exception as e:
            logger.error("Failed to connect to ADLS Gen2: {}", e)
            raise

    def upload_document(self, doc: CountyDocument) -> str:
        """
        Upload a single document to ADLS Gen2.

        Path: raw/{fips}/{category}/{doc_id}.json

        # DP-100: Data storage - Uploading documents to cloud storage for
        distributed access and backup.

        Args:
            doc: CountyDocument to upload.

        Returns:
            ADLS file path URL.
        """
        container = "raw"
        file_path = f"{doc.fips}/{doc.category}/{doc.id}.json"  # category is already a string
        
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            file_client = file_system_client.get_file_client(file_path)
            
            # Serialize document to JSON
            doc_json = doc.model_dump_json(indent=2)
            
            # Upload file
            file_client.upload_data(
                data=doc_json.encode("utf-8"),
                overwrite=True,
                content_settings={"ContentType": "application/json"},
            )
            
            url = f"https://{self.account_name}.dfs.core.windows.net/{container}/{file_path}"
            logger.debug("Uploaded document {} to ADLS: {}", doc.id, url)
            return url
            
        except AzureError as e:
            logger.error("Failed to upload document {}: {}", doc.id, e)
            raise

    def upload_documents(self, docs: List[CountyDocument]) -> List[str]:
        """
        Batch upload multiple documents to ADLS Gen2.

        # DP-100: Data storage - Batch operations for efficient cloud uploads.

        Args:
            docs: List of CountyDocument to upload.

        Returns:
            List of ADLS file path URLs.
        """
        urls: List[str] = []
        for doc in docs:
            try:
                url = self.upload_document(doc)
                urls.append(url)
            except Exception as e:
                logger.warning("Failed to upload document {} in batch: {}", doc.id, e)
        logger.info("Uploaded {}/{} documents to ADLS", len(urls), len(docs))
        return urls

    def upload_pdf(self, fips: str, category: str, filename: str, pdf_bytes: bytes) -> str:
        """
        Upload raw PDF file to ADLS Gen2.

        Path: raw/{fips}/{category}/pdfs/{filename}

        Args:
            fips: County FIPS code.
            category: Document category.
            filename: PDF filename.
            pdf_bytes: PDF file bytes.

        Returns:
            ADLS file path URL.
        """
        container = "raw"
        file_path = f"{fips}/{category}/pdfs/{filename}"
        
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            file_client = file_system_client.get_file_client(file_path)
            
            file_client.upload_data(
                data=pdf_bytes,
                overwrite=True,
                content_settings={"ContentType": "application/pdf"},
            )
            
            url = f"https://{self.account_name}.dfs.core.windows.net/{container}/{file_path}"
            logger.info("Uploaded PDF {} to ADLS: {}", filename, url)
            return url
            
        except AzureError as e:
            logger.error("Failed to upload PDF {}: {}", filename, e)
            raise

    def upload_crawl_log(self, fips: str, records: List[CrawlRecord]) -> str:
        """
        Upload crawl log records to ADLS Gen2.

        Path: crawl-logs/{fips}/log.jsonl

        Args:
            fips: County FIPS code.
            records: List of CrawlRecord to upload.

        Returns:
            ADLS file path URL.
        """
        container = "crawl-logs"
        file_path = f"{fips}/log.jsonl"
        
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            file_client = file_system_client.get_file_client(file_path)
            
            # Append records as JSONL
            jsonl_lines = [r.model_dump_json() + "\n" for r in records]
            jsonl_data = "".join(jsonl_lines).encode("utf-8")
            
            # Check if file exists - if so, append; otherwise create
            try:
                existing_data = file_client.download_file().readall()
                jsonl_data = existing_data + jsonl_data
            except AzureError:
                # File doesn't exist yet, create new
                pass
            
            file_client.upload_data(
                data=jsonl_data,
                overwrite=True,
                content_settings={"ContentType": "application/x-ndjson"},
            )
            
            url = f"https://{self.account_name}.dfs.core.windows.net/{container}/{file_path}"
            logger.info("Uploaded {} crawl log records for FIPS {} to ADLS", len(records), fips)
            return url
            
        except AzureError as e:
            logger.error("Failed to upload crawl log for FIPS {}: {}", fips, e)
            raise

    def download_document(self, fips: str, category: str, doc_id: str) -> Optional[CountyDocument]:
        """
        Download a document from ADLS Gen2.

        Args:
            fips: County FIPS code.
            category: Document category.
            doc_id: Document ID (UUID string).

        Returns:
            CountyDocument if found, None otherwise.
        """
        container = "raw"
        file_path = f"{fips}/{category}/{doc_id}.json"
        
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            file_client = file_system_client.get_file_client(file_path)
            
            file_data = file_client.download_file().readall()
            doc_dict = json.loads(file_data.decode("utf-8"))
            
            doc = CountyDocument.model_validate(doc_dict)
            logger.debug("Downloaded document {} from ADLS", doc_id)
            return doc
            
        except AzureError as e:
            logger.debug("Document {} not found in ADLS: {}", doc_id, e)
            return None

    def list_documents(self, fips: str, category: Optional[str] = None) -> List[str]:
        """
        List document IDs in ADLS Gen2 for a county (and optionally category).

        Args:
            fips: County FIPS code.
            category: Optional category filter.

        Returns:
            List of document IDs (UUID strings).
        """
        container = "raw"
        path_prefix = f"{fips}/"
        if category:
            path_prefix += f"{category}/"
        
        doc_ids: List[str] = []
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            paths = file_system_client.get_paths(path=path_prefix, recursive=True)
            
            for path in paths:
                if path.is_directory:
                    continue
                # Extract doc_id from path: {fips}/{category}/{doc_id}.json
                parts = path.name.split("/")
                if len(parts) >= 3 and parts[-1].endswith(".json"):
                    doc_id = parts[-1].replace(".json", "")
                    doc_ids.append(doc_id)
            
            logger.debug("Listed {} documents for FIPS {} category {}", len(doc_ids), fips, category)
            return doc_ids
            
        except AzureError as e:
            logger.error("Failed to list documents for FIPS {}: {}", fips, e)
            return []

    def get_storage_stats(self) -> StorageStats:
        """
        Get storage statistics for ADLS Gen2 account.

        # DP-100: Data monitoring - Storage statistics for capacity planning.

        Returns:
            StorageStats with file counts and sizes.
        """
        total_files = 0
        total_size_bytes = 0
        files_by_category: dict[str, int] = {}
        last_upload: Optional[datetime] = None
        
        containers = ["raw", "processed", "embeddings", "crawl-logs"]
        
        try:
            for container_name in containers:
                file_system_client = self.service_client.get_file_system_client(container_name)
                try:
                    paths = file_system_client.get_paths(recursive=True)
                    for path in paths:
                        if path.is_directory:
                            continue
                        total_files += 1
                        if path.content_length:
                            total_size_bytes += path.content_length
                        
                        # Extract category from path
                        parts = path.name.split("/")
                        if len(parts) >= 2:
                            category = parts[1]
                            files_by_category[category] = files_by_category.get(category, 0) + 1
                        
                        # Track latest upload time
                        if path.last_modified:
                            if last_upload is None or path.last_modified > last_upload:
                                last_upload = path.last_modified
                except AzureError:
                    # Container might not exist
                    continue
            
            total_size_gb = total_size_bytes / (1024 ** 3)
            
            return StorageStats(
                total_files=total_files,
                total_size_gb=round(total_size_gb, 2),
                files_by_category=files_by_category,
                last_upload=last_upload,
            )
            
        except Exception as e:
            logger.error("Failed to get storage stats: {}", e)
            return StorageStats()
