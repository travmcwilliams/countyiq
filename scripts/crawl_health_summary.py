"""Print crawl health report for CI (e.g. GitHub Actions)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.pipeline_monitor import PipelineMonitor

def main() -> None:
    monitor = PipelineMonitor()
    report = monitor.get_crawl_health()
    print("Crawl health report")
    print("------------------")
    print("Total counties:", report.total_counties)
    print("Crawled counties:", report.crawled_counties)
    print("Pending counties:", report.pending_counties)
    print("Failed counties:", report.failed_counties)
    print("Total documents:", report.total_documents)
    print("Avg RAG confidence:", round(report.avg_confidence, 4))
    print("Last crawl time:", report.last_crawl_time or "N/A")

if __name__ == "__main__":
    main()
