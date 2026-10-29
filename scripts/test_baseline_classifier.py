"""Fixture tests for baseline_classifier breakpoint classification.

Tests use synthetic run artifacts to verify that each breakpoint category
is correctly detected and classified with the right confidence level.

Run: cd /path/to/VoxLens && PYTHONPATH=apps/api:scripts python -m pytest scripts/test_baseline_classifier.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from baseline_classifier import (
    BreakpointFinding,
    RunBreakpoints,
    CitationCheckResult,
    classify_run,
)


def _write_run_artifacts(
    tmp_path: Path,
    run_id: str = "run-test0001",
    record_data: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    report_data: dict[str, Any] | None = None,
) -> Path:
    """Write synthetic run artifacts to a temporary directory."""
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if record_data is None:
        record_data = {
            "runId": run_id,
            "status": "completed",
            "request": {
                "query": "test query",
                "need": "test",
                "lang": "zh",
                "platforms": ["bilibili", "youtube"],
                "useLiveProviders": True,
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
    """Build a minimal valid report dict for testing."""
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
            "totalSources": len(sources or []),
            "totalComments": 0,
            "platformCount": 0,
            "citedSources": 0,
            "withComments": 0,
            "withTranscripts": 0,
            "providerRuns": 0,
            "successfulProviderRuns": 0,
            "failedProviderRuns": 0,
        },
        "quality": {
            "coverage": {"score": 80, "label": "", "rationale": "", "sourceIds": []},
            "citationAccuracy": {"score": 100, "label": "", "rationale": "", "sourceIds": []},
            "evidenceStrength": {"score": 70, "label": "", "rationale": "", "sourceIds": []},
            "conclusionRisk": {"score": 30, "label": "", "rationale": "", "sourceIds": []},
            "overall": 70,
            "warnings": [],
        },
        "sections": sections or [],
        "sources": sources or [],
        "warnings": warnings or [],
        "methodology": [],
        "runLogs": run_logs or [],
        "agentTrace": agent_trace or [],
        "isDemoFallback": False,
    }


# ============================================================
# Test 1: structural_gap detection
# Rule: Platform in _PLATFORMS_WITHOUT_TRANSCRIPT + no source has transcript
# ============================================================
class TestStructuralGap:
    def test_douyin_no_transcript_detected(self, tmp_path: Path):
        """douyin sources without transcript content → structural_gap, high confidence."""
        report = _minimal_report(
            sources=[
                {"id": 1, "platform": "douyin", "title": "Test Video",
                 "transcriptPreview": "", "fullTranscript": "", "transcriptText": "",
                 "sourceType": "video", "url": "https://example.com/1"},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)

        structural = result.findings_for("structural_gap")
        assert len(structural) == 1
        assert structural[0].confidence == "high"
        assert structural[0].platform == "douyin"
        assert "douyin" in structural[0].evidence

    def test_douyin_with_transcript_not_flagged(self, tmp_path: Path):
        """douyin source WITH transcript content → not a structural gap."""
        report = _minimal_report(
            sources=[
                {"id": 1, "platform": "douyin", "title": "Test Video",
                 "transcriptPreview": "some transcript text",
                 "fullTranscript": "", "transcriptText": "",
                 "sourceType": "video", "url": "https://example.com/1"},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)

        structural = result.findings_for("structural_gap")
        assert len(structural) == 0

    def test_bilibili_not_flagged_as_structural(self, tmp_path: Path):
        """bilibili has transcript capability → never a structural gap."""
        report = _minimal_report(
            sources=[
                {"id": 1, "platform": "bilibili", "title": "Test Video",
                 "transcriptPreview": "", "fullTranscript": "", "transcriptText": "",
                 "sourceType": "video", "url": "https://example.com/1"},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)

        structural = result.findings_for("structural_gap")
        assert len(structural) == 0


# ============================================================
# Test 2: adapter_not_wired detection
# Rule: Only ASRProviderUnavailable exception → high confidence
#       ASR step status=skipped alone → NOT adapter_not_wired
# ============================================================
class TestAdapterNotWired:
    def test_asr_provider_unavailable_in_agenttrace(self, tmp_path: Path):
        """ASRProviderUnavailable in agentTrace → adapter_not_wired, high confidence."""
        report = _minimal_report(
            agent_trace=[
                {"name": "ASRPipeline", "role": "ASR planning",
                 "status": "ok", "message": "ASRProviderUnavailable: faster-whisper runtime is not bundled",
                 "elapsedSec": 0.1, "metrics": {}},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)

        adapter = result.findings_for("adapter_not_wired")
        assert len(adapter) == 1
        assert adapter[0].confidence == "high"
        assert "ASRProviderUnavailable" in adapter[0].evidence

    def test_asr_skipped_without_exception_not_flagged(self, tmp_path: Path):
        """ASR step with status=skipped but no ASRProviderUnavailable → NOT adapter_not_wired."""
        report = _minimal_report(
            agent_trace=[
                {"name": "ASRPipeline", "role": "ASR planning",
                 "status": "skipped", "message": "No URL-backed source qualified for deferred ASR execution.",
                 "elapsedSec": 0.0, "metrics": {"candidates": 0}},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)

        adapter = result.findings_for("adapter_not_wired")
        assert len(adapter) == 0

    def test_asr_provider_unavailable_in_events(self, tmp_path: Path):
        """ASRProviderUnavailable in events.jsonl → adapter_not_wired, no duplicates."""
        events = [
            {"type": "agent_step", "sequence": 5,
             "name": "ASRPipeline", "status": "ok",
             "message": "ASRProviderUnavailable: faster-whisper runtime is not bundled"},
        ]
        run_dir = _write_run_artifacts(tmp_path, events=events)
        result = classify_run(run_dir)

        adapter = result.findings_for("adapter_not_wired")
        # Should produce exactly 1 finding, not duplicates
        assert len(adapter) == 1
        assert adapter[0].confidence == "high"

    def test_normalize_asr_key_matches_across_sources(self, tmp_path: Path):
        """Normalized key should match across record.error, event, and agentTrace."""
        from baseline_classifier import _normalize_asr_key

        key1 = _normalize_asr_key("record.error", "ASRProviderUnavailable: faster-whisper runtime is not bundled")
        key2 = _normalize_asr_key("ASRPipeline", "ASRProviderUnavailable: faster-whisper runtime is not bundled")
        assert key1 == key2, "Different step names with same exception should produce same key"


# ============================================================
# Test 3: runtime_volatility detection
# Rule: RunLog.ok=False, record.error (non-ASR), error events, stage error/partial
# ============================================================
class TestRuntimeVolatility:
    def test_failed_runlog_detected(self, tmp_path: Path):
        """RunLog with ok=False → runtime_volatility, high confidence, per-platform."""
        report = _minimal_report(
            run_logs=[
                {"provider": "crawler", "platform": "douyin", "ok": False,
                 "count": 0, "elapsedSec": 5.0, "note": "Connection timeout", "query": "test"},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)

        runtime = result.findings_for("runtime_volatility")
        assert len(runtime) == 1
        assert runtime[0].confidence == "high"
        assert runtime[0].platform == "douyin"
        assert "timeout" in runtime[0].evidence.lower() or "Connection" in runtime[0].evidence

    def test_record_error_detected(self, tmp_path: Path):
        """record.error with non-ASR error → runtime_volatility."""
        record_data = {
            "runId": "run-test0001",
            "status": "failed",
            "request": {
                "query": "test", "need": "test", "lang": "zh",
                "platforms": ["bilibili"], "useLiveProviders": True,
            },
            "createdAt": "2026-08-03T15:00:00",
            "updatedAt": "2026-08-03T15:01:00",
            "queuedAt": "2026-08-03T15:00:00",
            "startedAt": "2026-08-03T15:00:01",
            "completedAt": "2026-08-03T15:01:00",
            "progress": 50,
            "eventCount": 5,
            "report": None,
            "error": "ConnectionRefusedError: Unable to connect to bilibili",
        }
        run_dir = _write_run_artifacts(tmp_path, record_data=record_data)
        result = classify_run(run_dir)

        runtime = result.findings_for("runtime_volatility")
        assert len(runtime) >= 1
        assert runtime[0].confidence == "high"

    def test_asr_error_in_record_not_counted_as_runtime(self, tmp_path: Path):
        """record.error containing ASRProviderUnavailable → NOT runtime_volatility."""
        record_data = {
            "runId": "run-test0001",
            "status": "failed",
            "request": {
                "query": "test", "need": "test", "lang": "zh",
                "platforms": ["douyin"], "useLiveProviders": True,
            },
            "createdAt": "2026-08-03T15:00:00",
            "updatedAt": "2026-08-03T15:01:00",
            "queuedAt": "2026-08-03T15:00:00",
            "startedAt": "2026-08-03T15:00:01",
            "completedAt": "2026-08-03T15:01:00",
            "progress": 30,
            "eventCount": 3,
            "report": None,
            "error": "ASRProviderUnavailable: faster-whisper runtime is not bundled",
        }
        run_dir = _write_run_artifacts(tmp_path, record_data=record_data)
        result = classify_run(run_dir)

        # Should be adapter_not_wired (now scanned from record.error), not runtime_volatility
        adapter = result.findings_for("adapter_not_wired")
        assert len(adapter) >= 1, "ASR error in record.error should produce adapter_not_wired finding"

        runtime = result.findings_for("runtime_volatility")
        # The ASR error in record.error should NOT produce a runtime_volatility finding
        asr_runtime = [r for r in runtime if "ASRProviderUnavailable" in r.evidence]
        assert len(asr_runtime) == 0


# ============================================================
# Test 4: synthesis_issue detection
# Rule: Citation ID cannot resolve to source → synthesis_issue, high confidence
# ============================================================
class TestSynthesisIssue:
    def test_unresolved_citation_detected(self, tmp_path: Path):
        """Citation with sourceId not in sources → synthesis_issue."""
        report = _minimal_report(
            sources=[
                {"id": 1, "platform": "bilibili", "title": "Source 1",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        citation_checks = (
            CitationCheckResult(
                citation_id="cit-001",
                source_id=999,  # not in sources
                resolved=False,
                text_match=False,
                run_id="run-test0001",
                query="test query",
            ),
        )
        result = classify_run(run_dir, citation_checks)

        synth = result.findings_for("synthesis_issue")
        assert len(synth) == 1
        assert synth[0].confidence == "high"
        assert "cit-001" in synth[0].evidence

    def test_resolved_citation_not_flagged(self, tmp_path: Path):
        """Citation that resolves correctly → no synthesis_issue."""
        report = _minimal_report(
            sources=[
                {"id": 1, "platform": "bilibili", "title": "Source 1",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        citation_checks = (
            CitationCheckResult(
                citation_id="cit-001",
                source_id=1,  # exists in sources
                resolved=True,
                text_match=True,
                run_id="run-test0001",
                query="test query",
            ),
        )
        result = classify_run(run_dir, citation_checks)

        synth = result.findings_for("synthesis_issue")
        assert len(synth) == 0


class TestCitationCheckerRealFields:
    """Test citation collection using actual ResearchReport model field paths."""

    def test_citations_from_section_bullets(self, tmp_path: Path):
        """sections[].bullets[].citations should be collected."""
        from baseline_citation_checker import _collect_citations
        from app.models import ResearchReport

        report_data = _minimal_report(
            sources=[
                {"id": 1, "platform": "bilibili", "title": "Source 1",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
            ],
            sections=[{
                "id": "sec-1",
                "title": "Test Section",
                "kind": "narrative",
                "body": "body text",
                "bullets": [
                    {"text": "important claim", "citations": [1]},
                    {"text": "another claim", "citations": [1]},
                ],
            }],
        )
        report = ResearchReport.model_validate(report_data)
        citations = _collect_citations(report)

        assert len(citations) == 2
        assert all(c["sourceId"] == 1 for c in citations)

    def test_citations_from_section_sourceIds(self, tmp_path: Path):
        """sections[].sourceIds should be collected."""
        from baseline_citation_checker import _collect_citations
        from app.models import ResearchReport

        report_data = _minimal_report(
            sources=[
                {"id": 1, "platform": "bilibili", "title": "Source 1",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
                {"id": 2, "platform": "youtube", "title": "Source 2",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/2"},
            ],
            sections=[{
                "id": "sec-1",
                "title": "Test Section",
                "sourceIds": [1, 2],
            }],
        )
        report = ResearchReport.model_validate(report_data)
        citations = _collect_citations(report)

        assert len(citations) == 2
        source_ids = {c["sourceId"] for c in citations}
        assert source_ids == {1, 2}

    def test_citations_from_takeaways(self, tmp_path: Path):
        """takeaways[].citations should be collected."""
        from baseline_citation_checker import _collect_citations
        from app.models import ResearchReport

        report_data = _minimal_report(
            sources=[
                {"id": 1, "platform": "bilibili", "title": "Source 1",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
            ],
            takeaways=[
                {"text": "key insight", "citations": [1]},
            ],
        )
        report = ResearchReport.model_validate(report_data)
        citations = _collect_citations(report)

        assert len(citations) == 1
        assert citations[0]["sourceId"] == 1

    def test_citations_from_quote(self, tmp_path: Path):
        """sections[].quote.sourceId should be collected."""
        from baseline_citation_checker import _collect_citations
        from app.models import ResearchReport

        report_data = _minimal_report(
            sources=[
                {"id": 1, "platform": "bilibili", "title": "Source 1",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
            ],
            sections=[{
                "id": "sec-1",
                "title": "Test Section",
                "quote": {"sourceId": 1, "quote": "this is a quote",
                          "author": "user", "thumbnail": ""},
            }],
        )
        report = ResearchReport.model_validate(report_data)
        citations = _collect_citations(report)

        assert len(citations) == 1
        assert citations[0]["sourceId"] == 1

    def test_citations_from_table_evidence(self, tmp_path: Path):
        """sections[].table[].evidence should be collected."""
        from baseline_citation_checker import _collect_citations
        from app.models import ResearchReport

        report_data = _minimal_report(
            sources=[
                {"id": 1, "platform": "bilibili", "title": "Source 1",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
                {"id": 2, "platform": "youtube", "title": "Source 2",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/2"},
            ],
            sections=[{
                "id": "sec-1",
                "title": "Comparison",
                "table": [{
                    "name": "Product A",
                    "evidence": [1, 2],
                }],
            }],
        )
        report = ResearchReport.model_validate(report_data)
        citations = _collect_citations(report)

        assert len(citations) == 2
        source_ids = {c["sourceId"] for c in citations}
        assert source_ids == {1, 2}


# ============================================================
# Test 5: needs_manual_review fallback
# Rule: Failed run with no specific evidence → needs_manual_review
# ============================================================
class TestNeedsManualReview:
    def test_failed_run_no_evidence(self, tmp_path: Path):
        """Failed run with no recognizable evidence → needs_manual_review, low confidence."""
        record_data = {
            "runId": "run-test0001",
            "status": "failed",
            "request": {
                "query": "test", "need": "test", "lang": "zh",
                "platforms": ["bilibili"], "useLiveProviders": True,
            },
            "createdAt": "2026-08-03T15:00:00",
            "updatedAt": "2026-08-03T15:01:00",
            "queuedAt": "2026-08-03T15:00:00",
            "startedAt": "2026-08-03T15:00:01",
            "completedAt": "2026-08-03T15:01:00",
            "progress": 50,
            "eventCount": 3,
            "report": None,
            "error": "Unknown error occurred",
        }
        run_dir = _write_run_artifacts(tmp_path, record_data=record_data)
        result = classify_run(run_dir)

        review = result.findings_for("needs_manual_review")
        assert len(review) >= 1
        assert review[0].confidence == "low"

    def test_successful_run_no_findings(self, tmp_path: Path):
        """Successful run with no issues → needs_manual_review (informational)."""
        report = _minimal_report(
            sources=[
                {"id": 1, "platform": "bilibili", "title": "Source 1",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/1"},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)

        review = result.findings_for("needs_manual_review")
        assert len(review) == 1
        assert review[0].confidence == "low"


# ============================================================
# Test 6: Multiple findings in one run
# ============================================================
class TestMultipleFindings:
    def test_structural_gap_and_runtime_volatility(self, tmp_path: Path):
        """A run can have both structural_gap and runtime_volatility findings."""
        report = _minimal_report(
            sources=[
                {"id": 1, "platform": "douyin", "title": "Video 1",
                 "transcriptPreview": "", "sourceType": "video", "url": "https://example.com/1"},
                {"id": 2, "platform": "bilibili", "title": "Video 2",
                 "transcriptPreview": "text", "sourceType": "video", "url": "https://example.com/2"},
            ],
            run_logs=[
                {"provider": "crawler", "platform": "douyin", "ok": False,
                 "count": 0, "elapsedSec": 10.0, "note": "Rate limited", "query": "test"},
            ],
        )
        run_dir = _write_run_artifacts(tmp_path, report_data=report)
        result = classify_run(run_dir)

        structural = result.findings_for("structural_gap")
        runtime = result.findings_for("runtime_volatility")

        assert len(structural) == 1
        assert structural[0].platform == "douyin"

        assert len(runtime) >= 1
        douyin_runtime = [r for r in runtime if r.platform == "douyin"]
        assert len(douyin_runtime) == 1
