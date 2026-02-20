"""CLI to sync local DuckDB documents to ADLS Gen2.

# DP-100: Data management - Sync script ensures all local documents are
backed up to cloud storage.
"""

import argparse
import os
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from data.storage.storage_manager import StorageManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync local DuckDB documents to ADLS Gen2 cloud storage."
    )
    parser.add_argument(
        "--fips",
        type=str,
        help="Sync only documents for this county FIPS (e.g. 01001).",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Sync only documents for this category (e.g. property).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sync all unsynced documents (default).",
    )
    parser.add_argument(
        "--duckdb-path",
        type=str,
        default=None,
        help="Path to DuckDB database (default: from DUCKDB_PATH env or F:/countyiq/db/countyiq.duckdb).",
    )
    args = parser.parse_args()

    # Get ADLS credentials from environment
    adls_account_name = os.getenv("ADLS_ACCOUNT_NAME")
    adls_account_key = os.getenv("ADLS_ACCOUNT_KEY")
    
    if not adls_account_name or not adls_account_key:
        logger.error("ADLS_ACCOUNT_NAME and ADLS_ACCOUNT_KEY must be set in environment")
        sys.exit(1)

    # Get DuckDB path
    duckdb_path = args.duckdb_path or os.getenv("DUCKDB_PATH", "F:/countyiq/db/countyiq.duckdb")

    # Initialize StorageManager
    try:
        manager = StorageManager(
            duckdb_path=duckdb_path,
            adls_account_name=adls_account_name,
            adls_account_key=adls_account_key,
        )
    except Exception as e:
        logger.error("Failed to initialize StorageManager: {}", e)
        sys.exit(1)

    # Determine sync scope
    fips_filter = args.fips if args.fips else None
    
    if args.category and not args.fips:
        logger.warning("--category requires --fips (category filtering not yet implemented for sync)")
        if not args.fips:
            logger.error("--category requires --fips")
            sys.exit(1)

    # Run sync
    logger.info("Starting sync to ADLS Gen2...")
    if fips_filter:
        logger.info("Syncing documents for FIPS: {}", fips_filter)
    else:
        logger.info("Syncing all unsynced documents")

    try:
        result = manager.sync_to_cloud(fips=fips_filter)
        
        print("\nSync Summary:")
        print(f"  Synced:        {result.synced}")
        print(f"  Already synced: {result.already_synced}")
        print(f"  Failed:        {result.failed}")
        print(f"  Total:         {result.synced + result.already_synced + result.failed}")
        
        if result.failed > 0:
            logger.warning("Some documents failed to sync - check logs")
            sys.exit(1)
        else:
            logger.success("Sync completed successfully")
            
    except Exception as e:
        logger.error("Sync failed: {}", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
