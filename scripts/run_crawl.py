"""CLI to run county crawl pipeline (single county, state, or all).

# DP-100: Pipeline orchestration - Command-line interface for scheduled crawl jobs.
"""

import argparse
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from pipelines.ingest.crawl_orchestrator import CrawlOrchestrator
from pipelines.ingest.county_registry import load_counties, get_counties_by_state


def _parse_categories(s: str | None) -> list[str] | None:
    if s is None:
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CountyIQ county crawl pipeline (county, state, or all)."
    )
    parser.add_argument(
        "--state",
        type=str,
        help="Crawl all counties in this state (e.g. AL, CA).",
    )
    parser.add_argument(
        "--fips",
        type=str,
        help="Crawl a single county by FIPS code (e.g. 01001).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Crawl all counties in the registry.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="property,legal,demographics",
        help="Comma-separated categories (default: property,legal,demographics).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Max parallel workers for --all (default: 10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be crawled without running crawls.",
    )
    args = parser.parse_args()

    categories = _parse_categories(args.categories)

    if sum([bool(args.state), bool(args.fips), args.all]) != 1:
        parser.error("Exactly one of --state, --fips, or --all is required.")

    orchestrator = CrawlOrchestrator(max_workers=args.max_workers)

    if args.dry_run:
        if args.fips:
            from pipelines.ingest.county_registry import get_county_by_fips
            c = get_county_by_fips(args.fips)
            if c:
                print(f"Dry run: would crawl 1 county: {c.county_name} ({c.fips})")
            else:
                print(f"Dry run: unknown FIPS {args.fips}")
        elif args.state:
            counties = get_counties_by_state(args.state)
            print(f"Dry run: would crawl {len(counties)} counties in {args.state}:")
            for c in counties[:20]:
                print(f"  {c.fips} {c.county_name}")
            if len(counties) > 20:
                print(f"  ... and {len(counties) - 20} more")
        else:
            counties = load_counties()
            print(f"Dry run: would crawl all {len(counties)} counties (max_workers={args.max_workers})")
            for c in counties[:15]:
                print(f"  {c.fips} {c.county_name} ({c.state_abbr})")
            if len(counties) > 15:
                print(f"  ... and {len(counties) - 15} more")
        print(f"Categories: {categories or 'default'}")
        return

    # Live run
    if args.fips:
        summary = orchestrator.run_county(args.fips, categories)
        _print_table([summary])
        return

    if args.state:
        summaries = orchestrator.run_state(args.state, categories)
        _print_table(summaries)
        return

    result = orchestrator.run_all(categories=categories, max_workers=args.max_workers)
    _print_table(result.summaries)
    print("\nOrchestration summary:")
    print(f"  Total counties: {result.total_counties}")
    print(f"  Succeeded:      {result.succeeded}")
    print(f"  Failed:         {result.failed}")
    print(f"  Skipped:        {result.skipped}")
    print(f"  Total documents: {result.total_documents}")
    print(f"  Duration:       {result.duration_seconds}s")


def _print_table(summaries: list) -> None:
    """Print a simple progress table of crawl summaries."""
    if not summaries:
        return
    rows = [
        ("FIPS", "County", "Categories", "Docs", "Duration", "Errors"),
    ]
    for s in summaries:
        rows.append((
            s.fips,
            s.county_name[:20] if s.county_name else "-",
            f"{len(s.categories_succeeded)}/{len(s.categories_attempted)}",
            str(s.documents_crawled),
            f"{s.duration_seconds}s",
            str(len(s.errors)) if s.errors else "-",
        ))
    col_widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        line = "  ".join(str(x).ljust(col_widths[j]) for j, x in enumerate(row))
        print(line)
        if i == 0:
            print("  " + "-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))


if __name__ == "__main__":
    main()
