#!/usr/bin/env python3
"""Standalone test runner for baseline_classifier fixture tests.

Runs without pytest — uses only stdlib unittest + the project's venv.
This avoids the missing pytest dependency in the API venv.

Usage:
    cd /path/to/VoxLens
    PYTHONPATH=apps/api:scripts apps/api/.venv/bin/python scripts/run_baseline_tests.py
"""

import sys
import tempfile
import json
from pathlib import Path
from typing import Any

# Ensure scripts/ is on path for baseline_classifier import
sys.path.insert(0, str(Path(__file__).parent))

from baseline_classifier import (
    BreakpointFinding,
    RunBreakpoints,
    CitationCheckResult,
    classify_run,
    _normalize_asr_key,
)


# ---- Test helpers (same as test_baseline_classifier.py fixtures) ----

def _write_run_artifacts(
    tmp_path: Path,
    run_id: str = "run-test0001",
    record_data: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    report_data: dict[str, Any] | None = None,
) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if record_data is None:
        record_data = {
            "runId": run_id,
            "status": "completed",
            "request": {
                "query": "test query", "need": "test", "lang": "zh",
                "platforms": ["bilibili", "youtube"], "useLiveProviders": True,
            },
            "createdAt": "2026-08-03T15:00:00",
            "updatedAt": "2026-08-03T15:05:00",
            "queuedAt": "2026-08-03T15:00:00",
            "startedAt": "2026-08-03T15:00:01",
            "completedAt": "2026-08-03T15:05:00",
            "progress": 100,
            "eventCount": 10,
            "report": None,
            "error": "",
        }

    (run_dir / "record.json").write_text(
        json.dumps(record_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if events is not None:
        lines = [json.dumps(e, ensure_ascii=False) for e in events]
        (run_dir / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if report_data is not None:
        (run_dir / "report.json").write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return run_dir


def _minimal_report(
    sources: list[dict[str, Any]] | None = None,
    run_logs: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    agent_trace: list[dict[str, Any]] | None = None,
    sections: list[dict[str, Any]] | None = None,
    takeaways: list[dict[str, Any]] | None = None,
    insights: list[dict[str, Any]] | None = None,
    query: str = "test query",
) -> dict[str, Any]:
    return {
        "runId": "run-test0001",
        "productName": "VoxLens Studio",
        "slogan": "Evidence, not answers.",
        "title": "Test Report",
        "query": query,
        "need": "test",
        "lang": "zh",
        "status": "completed",
        "confidence": "medium",
        "generatedAt": "2026-08-03T15:05:00",
        "totalVideos": len(sources or []),
        "totalComments": 0,
        "platforms": [],
        "outline": [],
        "takeaways": takeaways or [],
        "insights": insights or [],
        "coverage": {
            "totalSources": len(sources or []), "totalComments": 0, "platformCount": 0,
            "citedSources": 0, "withComments": 0, "withTranscripts": 0,
            "providerRuns": 0, "successfulProviderRuns": 0, "failedProviderRuns": 0,
        },
        "quality": {
            "coverage": {"score": 80, "label": "", "rationale": "", "sourceIds": []},
            "citationAccuracy": {"score": 100, "label": "", "rationale": "", "sourceIds": []},
            "evidenceStrength": {"score": 70, "label": "", "rationale": "", "sourceIds": []},
            "conclusionRisk": {"score": 30, "label": "", "rationale": "", "sourceIds": []},
            "overall": 70, "warnings": [],
        },
        "sections": sections or [],
        "sources": sources or [],
        "warnings": warnings or [],
        "methodology": [],
        "runLogs": run_logs or [],
        "agentTrace": agent_trace or [],
        "isDemoFallback": False,
    }


# ---- Test runner ----

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def assert_eq(self, actual, expected, msg: str = ""):
        if actual != expected:
            raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")

    def assert_true(self, condition, msg: str = ""):
        if not condition:
            raise AssertionError(f"{msg}: expected True, got False")

    def run(self, name: str, fn):
        try:
            fn(self)
            self.passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            self.failed += 1
            self.errors.append(f"{name}: {e}")
            print(f"  ✗ {name}: {e}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for e in self.errors:
                print(f"  - {e}")
        return self.failed == 0


# ---- Tests ----

def test_normalize_asr_key(t: TestRunner):
    """Normalized key should match across record.error, event, and agentTrace formats."""
    # Same exception text from different sources should produce the same key
    key1 = _normalize_asr_key("record.error", "ASRProviderUnavailable: faster-whisper runtime is not bundled")
    key2 = _normalize_asr_key("ASRPipeline", "ASRProviderUnavailable: faster-whisper runtime is not bundled")
    t.assert_eq(key1, key2, "Different step names with same exception → same key")


def test_structural_gap_douyin(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = _minimal_report(
            sources=[{"id": 1, "platform": "douyin", "title": "V",
                      "transcriptPreview": "", "fullTranscript": "", "transcriptText": "",
                      "sourceType": "video", "url": "https://example.com/1"}],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)
        structural = result.findings_for("structural_gap")
        t.assert_eq(len(structural), 1, "douyin without transcript → 1 structural_gap")
        t.assert_eq(structural[0].confidence, "high")
        t.assert_eq(structural[0].platform, "douyin")


def test_structural_gap_douyin_with_transcript(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = _minimal_report(
            sources=[{"id": 1, "platform": "douyin", "title": "V",
                      "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"}],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)
        structural = result.findings_for("structural_gap")
        t.assert_eq(len(structural), 0, "douyin WITH transcript → 0 structural_gap")


def test_structural_gap_bilibili_not_flagged(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = _minimal_report(
            sources=[{"id": 1, "platform": "bilibili", "title": "V",
                      "transcriptPreview": "", "sourceType": "video", "url": "https://example.com/1"}],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)
        structural = result.findings_for("structural_gap")
        t.assert_eq(len(structural), 0, "bilibili → 0 structural_gap")


def test_asr_unavailable_in_agenttrace(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = _minimal_report(
            agent_trace=[{"name": "ASRPipeline", "role": "ASR", "status": "ok",
                          "message": "ASRProviderUnavailable: faster-whisper runtime is not bundled",
                          "elapsedSec": 0.1, "metrics": {}}],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)
        adapter = result.findings_for("adapter_not_wired")
        t.assert_eq(len(adapter), 1, "ASRProviderUnavailable in agentTrace → 1 finding")
        t.assert_eq(adapter[0].confidence, "high")


def test_asr_skipped_without_exception_not_flagged(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = _minimal_report(
            agent_trace=[{"name": "ASRPipeline", "role": "ASR", "status": "skipped",
                          "message": "No URL-backed source qualified.",
                          "elapsedSec": 0.0, "metrics": {"candidates": 0}}],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)
        adapter = result.findings_for("adapter_not_wired")
        t.assert_eq(len(adapter), 0, "ASR skipped without exception → 0 adapter_not_wired")


def test_asr_unavailable_in_events(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        events = [{"type": "agent_step", "sequence": 5, "name": "ASRPipeline",
                   "status": "ok",
                   "message": "ASRProviderUnavailable: faster-whisper runtime is not bundled"}]
        run_dir = _write_run_artifacts(tmp_path, events=events)
        result = classify_run(run_dir)
        adapter = result.findings_for("adapter_not_wired")
        t.assert_eq(len(adapter), 1, "ASRProviderUnavailable in events → exactly 1 finding")


def test_asr_no_duplicate_events_and_trace(t: TestRunner):
    """Same ASRProviderUnavailable in events AND agentTrace → exactly 1 finding."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        events = [{"type": "agent_step", "sequence": 5, "name": "ASRPipeline",
                   "status": "ok",
                   "message": "ASRProviderUnavailable: faster-whisper runtime is not bundled"}]
        report = _minimal_report(
            agent_trace=[{"name": "ASRPipeline", "role": "ASR", "status": "ok",
                          "message": "ASRProviderUnavailable: faster-whisper runtime is not bundled",
                          "elapsedSec": 0.1, "metrics": {}}],
        )
        run_dir = _write_run_artifacts(tmp_path, events=events, report_data=report)
        result = classify_run(run_dir)
        adapter = result.findings_for("adapter_not_wired")
        t.assert_eq(len(adapter), 1, "Same exception in events + agentTrace → deduplicated to 1")


def test_asr_no_duplicate_record_error_and_agenttrace(t: TestRunner):
    """Same ASRProviderUnavailable in record.error AND agentTrace → exactly 1 finding."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        record_data = {
            "runId": "run-test0001", "status": "failed",
            "request": {"query": "test", "need": "test", "lang": "zh",
                        "platforms": ["douyin"], "useLiveProviders": True},
            "createdAt": "2026-08-03T15:00:00", "updatedAt": "2026-08-03T15:01:00",
            "queuedAt": "2026-08-03T15:00:00", "startedAt": "2026-08-03T15:00:01",
            "completedAt": "2026-08-03T15:01:00", "progress": 30, "eventCount": 3,
            "report": None,
            "error": "ASRProviderUnavailable: faster-whisper runtime is not bundled",
        }
        report = _minimal_report(
            agent_trace=[{"name": "ASRPipeline", "role": "ASR", "status": "ok",
                          "message": "ASRProviderUnavailable: faster-whisper runtime is not bundled",
                          "elapsedSec": 0.1, "metrics": {}}],
        )
        run_dir = _write_run_artifacts(tmp_path, record_data=record_data, report_data=report)
        result = classify_run(run_dir)
        adapter = result.findings_for("adapter_not_wired")
        t.assert_eq(len(adapter), 1, "Same exception in record.error + agentTrace → deduplicated to 1")


def test_stage_partial_without_runtime_keyword_not_runtime(t: TestRunner):
    """stage status=partial without runtime keyword → NOT runtime_volatility."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        events = [{"type": "stage", "sequence": 3, "status": "partial",
                   "message": "Coverage below threshold for this section",
                   "platform": "bilibili"}]
        run_dir = _write_run_artifacts(tmp_path, events=events)
        result = classify_run(run_dir)
        runtime = result.findings_for("runtime_volatility")
        stage_runtime = [r for r in runtime if "stage event" in r.evidence]
        t.assert_eq(len(stage_runtime), 0, "partial without runtime keyword → 0 runtime_volatility")


def test_stage_error_always_runtime(t: TestRunner):
    """stage status=error → always runtime_volatility (explicit failure)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        events = [{"type": "stage", "sequence": 3, "status": "error",
                   "message": "Platform crawler failed",
                   "platform": "douyin"}]
        run_dir = _write_run_artifacts(tmp_path, events=events)
        result = classify_run(run_dir)
        runtime = result.findings_for("runtime_volatility")
        t.assert_eq(len(runtime), 1, "stage error → 1 runtime_volatility")
        t.assert_eq(runtime[0].platform, "douyin")


def test_failed_runlog_detected(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = _minimal_report(
            run_logs=[{"provider": "crawler", "platform": "douyin", "ok": False,
                       "count": 0, "elapsedSec": 5.0, "note": "Connection timeout", "query": "test"}],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)
        runtime = result.findings_for("runtime_volatility")
        t.assert_eq(len(runtime), 1, "Failed RunLog → 1 runtime_volatility")
        t.assert_eq(runtime[0].platform, "douyin")


def test_record_error_non_asr(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        record_data = {
            "runId": "run-test0001", "status": "failed",
            "request": {"query": "test", "need": "test", "lang": "zh",
                        "platforms": ["bilibili"], "useLiveProviders": True},
            "createdAt": "2026-08-03T15:00:00", "updatedAt": "2026-08-03T15:01:00",
            "queuedAt": "2026-08-03T15:00:00", "startedAt": "2026-08-03T15:00:01",
            "completedAt": "2026-08-03T15:01:00", "progress": 50, "eventCount": 5,
            "report": None, "error": "ConnectionRefusedError: Unable to connect",
        }
        run_dir = _write_run_artifacts(tmp_path, record_data=record_data)
        result = classify_run(run_dir)
        runtime = result.findings_for("runtime_volatility")
        t.assert_true(len(runtime) >= 1, "ConnectionRefusedError has runtime keyword → runtime_volatility")


def test_record_error_unknown_goes_to_review(t: TestRunner):
    """Unknown error without runtime keywords → needs_manual_review, not runtime_volatility."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        record_data = {
            "runId": "run-test0001", "status": "failed",
            "request": {"query": "test", "need": "test", "lang": "zh",
                        "platforms": ["bilibili"], "useLiveProviders": True},
            "createdAt": "2026-08-03T15:00:00", "updatedAt": "2026-08-03T15:01:00",
            "queuedAt": "2026-08-03T15:00:00", "startedAt": "2026-08-03T15:00:01",
            "completedAt": "2026-08-03T15:01:00", "progress": 50, "eventCount": 3,
            "report": None, "error": "Unknown error occurred",
        }
        run_dir = _write_run_artifacts(tmp_path, record_data=record_data)
        result = classify_run(run_dir)
        runtime = result.findings_for("runtime_volatility")
        review = result.findings_for("needs_manual_review")
        t.assert_eq(len(runtime), 0, "Unknown error → 0 runtime_volatility")
        t.assert_true(len(review) >= 1, "Unknown error → needs_manual_review")
        t.assert_eq(review[0].confidence, "low")


def test_asr_error_in_record_not_runtime(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        record_data = {
            "runId": "run-test0001", "status": "failed",
            "request": {"query": "test", "need": "test", "lang": "zh",
                        "platforms": ["douyin"], "useLiveProviders": True},
            "createdAt": "2026-08-03T15:00:00", "updatedAt": "2026-08-03T15:01:00",
            "queuedAt": "2026-08-03T15:00:00", "startedAt": "2026-08-03T15:00:01",
            "completedAt": "2026-08-03T15:01:00", "progress": 30, "eventCount": 3,
            "report": None, "error": "ASRProviderUnavailable: faster-whisper runtime is not bundled",
        }
        run_dir = _write_run_artifacts(tmp_path, record_data=record_data)
        result = classify_run(run_dir)
        adapter = result.findings_for("adapter_not_wired")
        t.assert_true(len(adapter) >= 1, "ASR error in record → adapter_not_wired")
        asr_runtime = [r for r in result.findings_for("runtime_volatility")
                       if "ASRProviderUnavailable" in r.evidence]
        t.assert_eq(len(asr_runtime), 0, "ASR error should NOT produce runtime_volatility")


def test_unresolved_citation(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = _minimal_report(
            sources=[{"id": 1, "platform": "bilibili", "title": "S1",
                      "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"}],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        checks = (CitationCheckResult("cit-001", 999, False, False, "run-test0001", "test"),)
        result = classify_run(run_dir, checks)
        synth = result.findings_for("synthesis_issue")
        t.assert_eq(len(synth), 1, "Unresolved citation → 1 synthesis_issue")
        t.assert_eq(synth[0].confidence, "high")


def test_resolved_citation_not_flagged(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = _minimal_report(
            sources=[{"id": 1, "platform": "bilibili", "title": "S1",
                      "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"}],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        checks = (CitationCheckResult("cit-001", 1, True, True, "run-test0001", "test"),)
        result = classify_run(run_dir, checks)
        synth = result.findings_for("synthesis_issue")
        t.assert_eq(len(synth), 0, "Resolved citation → 0 synthesis_issue")


def test_failed_run_no_evidence(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        record_data = {
            "runId": "run-test0001", "status": "failed",
            "request": {"query": "test", "need": "test", "lang": "zh",
                        "platforms": ["bilibili"], "useLiveProviders": True},
            "createdAt": "2026-08-03T15:00:00", "updatedAt": "2026-08-03T15:01:00",
            "queuedAt": "2026-08-03T15:00:00", "startedAt": "2026-08-03T15:00:01",
            "completedAt": "2026-08-03T15:01:00", "progress": 50, "eventCount": 3,
            "report": None, "error": "Unknown error occurred",
        }
        run_dir = _write_run_artifacts(tmp_path, record_data=record_data)
        result = classify_run(run_dir)
        review = result.findings_for("needs_manual_review")
        t.assert_true(len(review) >= 1, "Failed run with no evidence → needs_manual_review")
        t.assert_eq(review[0].confidence, "low")


def test_citation_from_section_bullets(t: TestRunner):
    from baseline_citation_checker import _collect_citations
    from app.models import ResearchReport
    report_data = _minimal_report(
        sources=[{"id": 1, "platform": "bilibili", "title": "S1",
                  "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"}],
        sections=[{"id": "s1", "title": "S", "bullets": [{"text": "claim", "citations": [1]}]}],
    )
    report = ResearchReport.model_validate(report_data)
    citations = _collect_citations(report)
    t.assert_eq(len(citations), 1, "bullets[].citations → 1 citation")
    t.assert_eq(citations[0]["sourceId"], 1)


def test_citation_from_section_sourceIds(t: TestRunner):
    from baseline_citation_checker import _collect_citations
    from app.models import ResearchReport
    report_data = _minimal_report(
        sources=[{"id": 1, "platform": "bilibili", "title": "S1",
                  "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
                 {"id": 2, "platform": "youtube", "title": "S2",
                  "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/2"}],
        sections=[{"id": "s1", "title": "S", "sourceIds": [1, 2]}],
    )
    report = ResearchReport.model_validate(report_data)
    citations = _collect_citations(report)
    t.assert_eq(len(citations), 2, "sourceIds → 2 citations")


def test_citation_from_takeaways(t: TestRunner):
    from baseline_citation_checker import _collect_citations
    from app.models import ResearchReport
    report_data = _minimal_report(
        sources=[{"id": 1, "platform": "bilibili", "title": "S1",
                  "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"}],
        takeaways=[{"text": "insight", "citations": [1]}],
    )
    report = ResearchReport.model_validate(report_data)
    citations = _collect_citations(report)
    t.assert_eq(len(citations), 1, "takeaways[].citations → 1 citation")


def test_citation_from_quote(t: TestRunner):
    from baseline_citation_checker import _collect_citations
    from app.models import ResearchReport
    report_data = _minimal_report(
        sources=[{"id": 1, "platform": "bilibili", "title": "S1",
                  "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"}],
        sections=[{"id": "s1", "title": "S",
                   "quote": {"sourceId": 1, "quote": "q", "author": "a", "thumbnail": ""}}],
    )
    report = ResearchReport.model_validate(report_data)
    citations = _collect_citations(report)
    t.assert_eq(len(citations), 1, "quote.sourceId → 1 citation")


def test_citation_from_table_evidence(t: TestRunner):
    from baseline_citation_checker import _collect_citations
    from app.models import ResearchReport
    report_data = _minimal_report(
        sources=[{"id": 1, "platform": "bilibili", "title": "S1",
                  "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
                 {"id": 2, "platform": "youtube", "title": "S2",
                  "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/2"}],
        sections=[{"id": "s1", "title": "C", "table": [{"name": "P", "evidence": [1, 2]}]}],
    )
    report = ResearchReport.model_validate(report_data)
    citations = _collect_citations(report)
    t.assert_eq(len(citations), 2, "table[].evidence → 2 citations")


def test_multiple_findings_structural_and_runtime(t: TestRunner):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = _minimal_report(
            sources=[{"id": 1, "platform": "douyin", "title": "V1",
                      "transcriptPreview": "", "sourceType": "video", "url": "https://example.com/1"},
                     {"id": 2, "platform": "bilibili", "title": "V2",
                      "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/2"}],
            run_logs=[{"provider": "crawler", "platform": "douyin", "ok": False,
                       "count": 0, "elapsedSec": 10.0, "note": "Rate limited", "query": "test"}],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)
        structural = result.findings_for("structural_gap")
        runtime = result.findings_for("runtime_volatility")
        t.assert_eq(len(structural), 1, "douyin → 1 structural_gap")
        t.assert_eq(structural[0].platform, "douyin")
        douyin_runtime = [r for r in runtime if r.platform == "douyin"]
        t.assert_eq(len(douyin_runtime), 1, "douyin runtime failure → 1 finding")


# ---- Main ----

def main():
    runner = TestRunner()

    tests = [
        ("test_normalize_asr_key", test_normalize_asr_key),
        ("test_structural_gap_douyin", test_structural_gap_douyin),
        ("test_structural_gap_douyin_with_transcript", test_structural_gap_douyin_with_transcript),
        ("test_structural_gap_bilibili_not_flagged", test_structural_gap_bilibili_not_flagged),
        ("test_asr_unavailable_in_agenttrace", test_asr_unavailable_in_agenttrace),
        ("test_asr_skipped_without_exception_not_flagged", test_asr_skipped_without_exception_not_flagged),
        ("test_asr_unavailable_in_events", test_asr_unavailable_in_events),
        ("test_asr_no_duplicate_events_and_trace", test_asr_no_duplicate_events_and_trace),
        ("test_asr_no_duplicate_record_error_and_agenttrace", test_asr_no_duplicate_record_error_and_agenttrace),
        ("test_stage_partial_without_runtime_keyword_not_runtime", test_stage_partial_without_runtime_keyword_not_runtime),
        ("test_stage_error_always_runtime", test_stage_error_always_runtime),
        ("test_failed_runlog_detected", test_failed_runlog_detected),
        ("test_record_error_non_asr", test_record_error_non_asr),
        ("test_record_error_unknown_goes_to_review", test_record_error_unknown_goes_to_review),
        ("test_asr_error_in_record_not_runtime", test_asr_error_in_record_not_runtime),
        ("test_unresolved_citation", test_unresolved_citation),
        ("test_resolved_citation_not_flagged", test_resolved_citation_not_flagged),
        ("test_failed_run_no_evidence", test_failed_run_no_evidence),
        ("test_citation_from_section_bullets", test_citation_from_section_bullets),
        ("test_citation_from_section_sourceIds", test_citation_from_section_sourceIds),
        ("test_citation_from_takeaways", test_citation_from_takeaways),
        ("test_citation_from_quote", test_citation_from_quote),
        ("test_citation_from_table_evidence", test_citation_from_table_evidence),
        ("test_multiple_findings_structural_and_runtime", test_multiple_findings_structural_and_runtime),
    ]

    print(f"Running {len(tests)} fixture tests...\n")
    for name, fn in tests:
        runner.run(name, fn)

    success = runner.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
