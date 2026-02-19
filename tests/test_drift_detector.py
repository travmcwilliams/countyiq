"""Tests for DriftDetector.

# DP-100: Data drift and monitoring - Unit tests for volume and content drift.
"""

import json
import tempfile
from pathlib import Path
import pytest

from monitoring.drift_detector import DriftDetector, DriftResult


def test_drift_result_model() -> None:
    """DriftResult accepts valid fields."""
    r = DriftResult(
        fips="01001",
        category="property",
        drift_detected=True,
        drift_type="volume",
        current_value=5.0,
        expected_value=10.0,
        severity="medium",
    )
    assert r.fips == "01001"
    assert r.severity == "medium"


def test_volume_drift_no_entries() -> None:
    """check_volume_drift returns no drift when no log entries."""
    with tempfile.TemporaryDirectory() as tmp:
        det = DriftDetector(crawl_log_dir=Path(tmp))
        result = det.check_volume_drift("01001", "property")
    assert result.drift_detected is False
    assert result.drift_type == "none"
    assert result.current_value == 0.0


def test_volume_drift_drop_detected() -> None:
    """Volume drop > 20% is detected as drift."""
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "01001" / "property"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "crawl_log.jsonl"
        # Write all lines at once: historical 10,10 then current 5 -> 50% drop
        lines = [
            json.dumps({"document_count": 10, "timestamp": 0}),
            json.dumps({"document_count": 10, "timestamp": 0}),
            json.dumps({"document_count": 5, "timestamp": 1}),
        ]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        det = DriftDetector(crawl_log_dir=Path(tmp), volume_drop_threshold=0.20)
        result = det.check_volume_drift("01001", "property")
    assert result.drift_detected is True
    assert result.drift_type == "volume"
    assert result.current_value == 5.0
    assert result.expected_value == 10.0  # historical_avg of [10,10] = 10


def test_volume_drift_no_drop() -> None:
    """No drift when volume is stable."""
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "01001" / "property"
        log_dir.mkdir(parents=True)
        (log_dir / "crawl_log.jsonl").write_text(
            "\n".join([
                json.dumps({"document_count": 10}),
                json.dumps({"document_count": 10}),
                json.dumps({"document_count": 10}),
            ]) + "\n",
            encoding="utf-8",
        )
        det = DriftDetector(crawl_log_dir=Path(tmp), volume_drop_threshold=0.20)
        result = det.check_volume_drift("01001", "property")
    assert result.drift_detected is False


def test_content_drift_no_entries() -> None:
    """check_content_drift returns no drift when no entries."""
    with tempfile.TemporaryDirectory() as tmp:
        det = DriftDetector(crawl_log_dir=Path(tmp))
        result = det.check_content_drift("01001", "property")
    assert result.drift_detected is False
    assert result.drift_type == "none"


def test_content_drift_change_detected() -> None:
    """Content length change > 30% is detected."""
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "01001" / "property"
        log_dir.mkdir(parents=True)
        (log_dir / "crawl_log.jsonl").write_text(
            "\n".join([
                json.dumps({"content_length_avg": 1000}),
                json.dumps({"content_length_avg": 1000}),
                json.dumps({"content_length_avg": 500}),  # 50% drop
            ]) + "\n",
            encoding="utf-8",
        )
        det = DriftDetector(crawl_log_dir=Path(tmp), content_change_threshold=0.30)
        result = det.check_content_drift("01001", "property")
    assert result.drift_detected is True
    assert result.drift_type == "content"
    assert result.current_value == 500
    assert result.expected_value == 1000.0


def test_severity_levels() -> None:
    """Severity is low/medium/high based on deviation."""
    r_low = DriftResult(fips="01001", category="property", severity="low")
    r_med = DriftResult(fips="01001", category="property", severity="medium")
    r_high = DriftResult(fips="01001", category="property", severity="high")
    assert r_low.severity == "low"
    assert r_med.severity == "medium"
    assert r_high.severity == "high"


def test_drift_fallback_from_crawl_json() -> None:
    """Drift detector can infer one entry from crawl.json when no JSONL."""
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "01001" / "property"
        log_dir.mkdir(parents=True)
        (log_dir / "crawl.json").write_text(
            json.dumps([{"a": 1}, {"b": 2}]),
            encoding="utf-8",
        )
        det = DriftDetector(crawl_log_dir=Path(tmp))
        result = det.check_volume_drift("01001", "property")
    assert result.current_value == 2  # len of list
    assert result.drift_detected is False
