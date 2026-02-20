"""Initialize DuckDB database for CountyIQ.

# DP-100: Data management - Database initialization script creates schema
and ensures proper directory structure for hybrid storage.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from data.storage.duckdb_store import DocumentStore


def main() -> None:
    """Initialize DuckDB database at F:/countyiq/db/countyiq.duckdb."""
    db_path = Path("F:/countyiq/db/countyiq.duckdb")
    
    # Create parent directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Created directory: {}", db_path.parent)
    
    # Initialize DocumentStore - this will create the schema
    try:
        store = DocumentStore(db_path)
        logger.info("DuckDB database initialized at: {}", db_path)
        
        # Verify schema was created
        result = store.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        logger.info("Schema verified - documents table exists (current count: {})", result[0])
        
        logger.success("DuckDB initialization complete!")
        logger.info("Database path: {}", db_path.absolute())
        
    except ImportError as e:
        logger.error("Failed to import duckdb: {}", e)
        logger.error("Install with: pip install duckdb")
        raise
    except Exception as e:
        logger.error("Failed to initialize DuckDB: {}", e)
        raise


if __name__ == "__main__":
    main()
