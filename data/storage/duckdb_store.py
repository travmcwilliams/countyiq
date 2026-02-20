"""DuckDB local document store for CountyIQ.

# DP-100: Data storage - Local DuckDB database for fast querying and
offline access to crawled documents.
"""

import json
from pathlib import Path
from typing import List, Optional
from uuid import UUID

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore

from loguru import logger

from data.schemas.document import CountyDocument


class DocumentStore:
    """
    DuckDB-based local document store.

    # DP-100: Data storage - DuckDB provides fast local querying and
    analytics on crawled documents without cloud dependency.
    """

    def __init__(self, db_path: Path | str):
        """
        Initialize DuckDB document store.

        Args:
            db_path: Path to DuckDB database file (e.g. F:/countyiq/db/countyiq.duckdb).
        """
        if duckdb is None:
            raise ImportError("duckdb is not installed. Install with: pip install duckdb")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = duckdb.connect(str(self.db_path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create documents table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR PRIMARY KEY,
                fips VARCHAR(5) NOT NULL,
                county_name VARCHAR NOT NULL,
                state_abbr VARCHAR(2) NOT NULL,
                category VARCHAR NOT NULL,
                source_url VARCHAR,
                content_type VARCHAR NOT NULL,
                raw_content TEXT,
                processed_content TEXT,
                embedding JSON,
                metadata JSON,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                synced_to_cloud BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create indexes for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fips ON documents(fips)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON documents(category)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_synced ON documents(synced_to_cloud)")
        
        logger.debug("DuckDB schema ensured at {}", self.db_path)

    def save(self, doc: CountyDocument) -> None:
        """
        Save a document to DuckDB.

        Args:
            doc: CountyDocument to save.
        """
        embedding_json = json.dumps(doc.embedding) if doc.embedding else None
        metadata_json = json.dumps(doc.metadata) if doc.metadata else None
        
        self.conn.execute("""
            INSERT OR REPLACE INTO documents (
                id, fips, county_name, state_abbr, category, source_url,
                content_type, raw_content, processed_content, embedding,
                metadata, created_at, updated_at, synced_to_cloud
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(doc.id),
            doc.fips,
            doc.county_name,
            doc.state_abbr,
            doc.category.value,
            doc.source_url,
            doc.content_type.value,
            doc.raw_content,
            doc.processed_content,
            embedding_json,
            metadata_json,
            doc.created_at,
            doc.updated_at,
            False,  # Not synced yet
        ))
        logger.debug("Saved document {} to DuckDB", doc.id)

    def save_many(self, docs: List[CountyDocument]) -> None:
        """
        Batch save multiple documents to DuckDB.

        Args:
            docs: List of CountyDocument to save.
        """
        for doc in docs:
            self.save(doc)
        logger.info("Saved {} documents to DuckDB", len(docs))

    def get(self, doc_id: str | UUID) -> Optional[CountyDocument]:
        """
        Get a document by ID from DuckDB.

        Args:
            doc_id: Document ID (UUID string or UUID).

        Returns:
            CountyDocument if found, None otherwise.
        """
        doc_id_str = str(doc_id)
        result = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id_str,),
        ).fetchone()
        
        if not result:
            return None
        
        # Reconstruct CountyDocument from row
        return self._row_to_document(result)

    def list_by_fips(self, fips: str, category: Optional[str] = None) -> List[CountyDocument]:
        """
        List documents for a county (and optionally category).

        Args:
            fips: County FIPS code.
            category: Optional category filter.

        Returns:
            List of CountyDocument.
        """
        if category:
            result = self.conn.execute(
                "SELECT * FROM documents WHERE fips = ? AND category = ?",
                (fips, category),
            ).fetchall()
        else:
            result = self.conn.execute(
                "SELECT * FROM documents WHERE fips = ?",
                (fips,),
            ).fetchall()
        
        return [self._row_to_document(row) for row in result]

    def list_unsynced(self, fips: Optional[str] = None) -> List[CountyDocument]:
        """
        List documents not yet synced to cloud.

        Args:
            fips: Optional FIPS filter.

        Returns:
            List of unsynced CountyDocument.
        """
        if fips:
            result_rows = self.conn.execute(
                "SELECT * FROM documents WHERE synced_to_cloud = FALSE AND fips = ?",
                (fips,),
            ).fetchall()
        else:
            result_rows = self.conn.execute(
                "SELECT * FROM documents WHERE synced_to_cloud = FALSE",
            ).fetchall()
        
        return [self._row_to_document(row) for row in result_rows]

    def mark_synced(self, doc_id: str | UUID) -> None:
        """
        Mark a document as synced to cloud.

        Args:
            doc_id: Document ID.
        """
        doc_id_str = str(doc_id)
        self.conn.execute(
            "UPDATE documents SET synced_to_cloud = TRUE WHERE id = ?",
            (doc_id_str,),
        )

    def _row_to_document(self, row) -> CountyDocument:
        """Convert DuckDB row to CountyDocument."""
        from data.schemas.document import ContentType, DocumentCategory
        
        embedding = json.loads(row[9]) if row[9] else None
        metadata = json.loads(row[10]) if row[10] else {}
        
        return CountyDocument(
            id=UUID(row[0]),
            fips=row[1],
            county_name=row[2],
            state_abbr=row[3],
            category=DocumentCategory(row[4]),
            source_url=row[5],
            content_type=ContentType(row[6]),
            raw_content=row[7] or "",
            processed_content=row[8],
            embedding=embedding,
            metadata=metadata,
            created_at=row[11],
            updated_at=row[12],
        )

    def get_stats(self) -> dict:
        """Get document statistics from DuckDB."""
        total = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        by_category = self.conn.execute(
            "SELECT category, COUNT(*) FROM documents GROUP BY category"
        ).fetchall()
        unsynced = self.conn.execute(
            "SELECT COUNT(*) FROM documents WHERE synced_to_cloud = FALSE"
        ).fetchone()[0]
        
        return {
            "total_documents": total,
            "by_category": {cat: count for cat, count in by_category},
            "unsynced_count": unsynced,
        }
