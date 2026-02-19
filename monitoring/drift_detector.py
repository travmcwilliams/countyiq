"""Drift detection for crawl volume and content.

# DP-100: Data drift and monitoring - Detecting distribution shift in crawl outputs.
"""

import json
from pathlib import Path
from typing import List, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

# Default location for crawl logs (JSONL: one JSON object per line)
DEFAULT_CRAWL_LOG_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


class DriftResult(BaseModel):
    """Result of a single drift check."""

    fips: str = Field(..., min_length=5, max_length=5)
    category: str = Field(...)
    drift_detected: bool = Field(default=False)
    drift_type: Literal["volume", "content", "none"] = Field(default="none")
    current_value: float = Field(default=0.0)
    expected_value: float = Field(default=0.0)
    severity: Literal["low", "medium", "high"] = Field(default="low")


def _severity(ratio: float, threshold: float) -> Literal["low", "medium", "high"]:
    """Map deviation ratio to severity."""
    if ratio < threshold:
        return "low"
    if ratio < threshold * 1.5:
        return "medium"
    return "high"


class DriftDetector:
    """
    Detects volume and content drift from crawl log files.

    # DP-100: Data drift - Compares current crawl metrics to historical baseline;
    flags possible site changes or blocks (volume) or format changes (content).
    """

    def __init__(
        self,
        crawl_log_dir: Optional[Path] = None,
        volume_drop_threshold: float = 0.20,
        content_change_threshold: float = 0.30,
    ):
        self.crawl_log_dir = crawl_log_dir or DEFAULT_CRAWL_LOG_DIR
        self.volume_drop_threshold = volume_drop_threshold
        self.content_change_threshold = content_change_threshold

    def _read_log_entries(
        self,
        fips: str,
        category: str,
        window_days: int = 7,
    ) -> List[dict]:
        """
        Read crawl log entries for a county/category from JSONL files.

        Looks for files under {crawl_log_dir}/{fips}/{category}/crawl_log.jsonl
        or similar, and parses lines as JSON. Each line should have: timestamp,
        document_count, and optionally content_length or doc_lengths.
        """
        # DP-100: Data asset - Reading structured log for drift baseline.
        entries: List[dict] = []
        category_dir = self.crawl_log_dir / fips / category
        if not category_dir.exists():
            return entries
        log_file = category_dir / "crawl_log.jsonl"
        if not log_file.exists():
            # Fallback: infer from crawl.json if present (one "run" = one doc count)
            crawl_file = category_dir / "crawl.json"
            if crawl_file.exists():
                try:
                    with open(crawl_file, encoding="utf-8") as f:
                        data = json.load(f)
                    count = len(data) if isinstance(data, list) else 1
                    total_len = sum(len(str(d).encode()) for d in (data if isinstance(data, list) else [data]))
                    entries.append({
                        "document_count": count,
                        "content_length_avg": total_len / max(count, 1),
                        "timestamp": crawl_file.stat().st_mtime,
                    })
                except Exception as e:
                    logger.debug("Could not read crawl.json for drift: {}", e)
            return entries
        try:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning("Could not read crawl log {}: {}", log_file, e)
        return entries

    def check_volume_drift(
        self,
        fips: str,
        category: str,
        window_days: int = 7,
    ) -> DriftResult:
        """
        Compare current crawl document count to historical average.

        Flags if volume drops by more than volume_drop_threshold (default 20%),
        e.g. possible site change or block.
        """
        # DP-100: Data drift - Volume drift detection for monitoring pipeline health.
        entries = self._read_log_entries(fips, category, window_days)
        if not entries:
            return DriftResult(
                fips=fips,
                category=category,
                drift_detected=False,
                drift_type="none",
                current_value=0.0,
                expected_value=0.0,
                severity="low",
            )
        counts = [e.get("document_count", 0) for e in entries]
        current = counts[-1] if counts else 0
        historical_avg = sum(counts[:-1]) / max(len(counts) - 1, 1) if len(counts) > 1 else current
        if historical_avg <= 0:
            return DriftResult(
                fips=fips,
                category=category,
                drift_detected=False,
                drift_type="none",
                current_value=float(current),
                expected_value=historical_avg,
                severity="low",
            )
        ratio = 1.0 - (current / historical_avg)
        drift_detected = ratio >= self.volume_drop_threshold
        return DriftResult(
            fips=fips,
            category=category,
            drift_detected=drift_detected,
            drift_type="volume" if drift_detected else "none",
            current_value=float(current),
            expected_value=round(historical_avg, 2),
            severity=_severity(ratio, self.volume_drop_threshold) if drift_detected else "low",
        )

    def check_content_drift(self, fips: str, category: str) -> DriftResult:
        """
        Compare current average document length to historical average.

        Flags if average length changes by more than content_change_threshold (default 30%),
        e.g. possible format change.
        """
        # DP-100: Data drift - Content/schema drift via document size distribution.
        entries = self._read_log_entries(fips, category, window_days=7)
        if not entries:
            return DriftResult(
                fips=fips,
                category=category,
                drift_detected=False,
                drift_type="none",
                current_value=0.0,
                expected_value=0.0,
                severity="low",
            )
        lengths = []
        for e in entries:
            avg_len = e.get("content_length_avg") or e.get("avg_content_length")
            if avg_len is not None:
                lengths.append(float(avg_len))
            elif "document_count" in e and "total_content_length" in e:
                n = e["document_count"] or 1
                lengths.append((e["total_content_length"] or 0) / n)
        if not lengths:
            return DriftResult(
                fips=fips,
                category=category,
                drift_detected=False,
                drift_type="none",
                current_value=0.0,
                expected_value=0.0,
                severity="low",
            )
        current = lengths[-1]
        historical_avg = sum(lengths[:-1]) / max(len(lengths) - 1, 1) if len(lengths) > 1 else current
        if historical_avg <= 0:
            return DriftResult(
                fips=fips,
                category=category,
                drift_detected=False,
                drift_type="none",
                current_value=current,
                expected_value=historical_avg,
                severity="low",
            )
        ratio = abs(current - historical_avg) / historical_avg
        drift_detected = ratio >= self.content_change_threshold
        return DriftResult(
            fips=fips,
            category=category,
            drift_detected=drift_detected,
            drift_type="content" if drift_detected else "none",
            current_value=round(current, 2),
            expected_value=round(historical_avg, 2),
            severity=_severity(ratio, self.content_change_threshold) if drift_detected else "low",
        )
