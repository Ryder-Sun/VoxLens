"""Tests for t3 (harness split + async LLM transport) and t5 (citation anchoring).

Covers:
- Async LLM transport module (llm_transport.py)
- Harness operator split into granular resumable operators
- Citation ref model and sanitizer extensions
- Segment-level evidence unit creation with timestamps
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.agents.budget import ResearchBudget
from app.agents.coordinator import (
    complete_research_pipeline,
    create_acquisition_batch,
    prepare_evidence_batch,
    _parse_timestamp_to_ms,
)
from app.agents.planner import build_research_plan
from app.harness import FileHarnessStore
from app.agents.artifacts import AcquisitionBatch
from app.models import (
    Artifact,
    CitationRef,
    CitationText,
    Comment,
    EvidenceUnit,
    RunLog,
    Source,
    TranscriptSegment,
    ResearchRequest,
)
from app.services.llm_sanitizers import (
    result_from_payload,
    _sanitize_citation_refs,
    valid_ids,
)
from app.services.llm_transport import (
    LLMTransportError,
    gather_independent_calls,
    post_with_retry,
)


class TimestampParsingTests(unittest.TestCase):
    """Test _parse_timestamp_to_ms handles all timestamp formats."""

    def test_parse_mm_ss_format(self):
        self.assertEqual(_parse_timestamp_to_ms("12:30"), 750000)

    def test_parse_hh_mm_ss_format(self):
        self.assertEqual(_parse_timestamp_to_ms("01:12:30"), 4350000)

    def test_parse_numeric_seconds(self):
        self.assertEqual(_parse_timestamp_to_ms(45.5), 45500)

    def test_parse_numeric_ms(self):
        self.assertEqual(_parse_timestamp_to_ms(750000), 750000)

    def test_parse_none(self):
        self.assertIsNone(_parse_timestamp_to_ms(None))

    def test_parse_empty_string(self):
        self.assertIsNone(_parse_timestamp_to_ms(""))

    def test_parse_invalid_string(self):
        self.assertIsNone(_parse_timestamp_to_ms("invalid"))


class CitationRefModelTests(unittest.TestCase):
    """Test CitationRef model and CitationText.citationRefs field."""

    def test_citation_ref_with_timestamp(self):
        ref = CitationRef(
            sourceId=1,
            platform="bilibili",
            platformObjectId="BV1xx",
            startMs=750000,
            endMs=780000,
        )
        result = ref.format_ref()
        self.assertIn("BILIBILI:BV1xx@", result)
        self.assertIn("12:30", result)

    def test_citation_ref_with_comment(self):
        ref = CitationRef(
            sourceId=1,
            platform="douyin",
            platformObjectId="123456",
            commentId="abc123",
        )
        result = ref.format_ref()
        self.assertIn("DOUYIN:123456/comment/abc123", result)

    def test_citation_ref_source_only(self):
        ref = CitationRef(sourceId=1, platform="youtube", platformObjectId="vid123")
        result = ref.format_ref()
        self.assertEqual(result, "[YOUTUBE:vid123]")

    def test_citation_text_has_citation_refs(self):
        ct = CitationText(text="test", citations=[1], citationRefs=[
            CitationRef(sourceId=1, platform="bilibili", platformObjectId="BV1xx", startMs=1000, endMs=2000)
        ])
        self.assertEqual(len(ct.citationRefs), 1)
        self.assertEqual(ct.citationRefs[0].sourceId, 1)

    def test_citation_text_default_empty_refs(self):
        ct = CitationText(text="test")
        self.assertEqual(ct.citationRefs, [])


class CitationRefSanitizerTests(unittest.TestCase):
    """Test _sanitize_citation_refs validates and filters citation refs."""

    def test_sanitizes_valid_refs(self):
        items = [
            {"sourceId": 1, "platform": "bilibili", "platformObjectId": "BV1xx", "startMs": 1000, "endMs": 2000},
            {"sourceId": 2, "platform": "youtube", "platformObjectId": "vid2"},
        ]
        result = _sanitize_citation_refs(items, {1, 2})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["sourceId"], 1)
        self.assertEqual(result[0]["startMs"], 1000)

    def test_filters_invalid_source_ids(self):
        items = [
            {"sourceId": 1, "platform": "bilibili"},
            {"sourceId": 99, "platform": "youtube"},
        ]
        result = _sanitize_citation_refs(items, {1, 2})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sourceId"], 1)

    def test_handles_non_list_input(self):
        result = _sanitize_citation_refs("not a list", {1})
        self.assertEqual(result, [])

    def test_limits_to_12_refs(self):
        items = [{"sourceId": 1, "platform": "bilibili"} for _ in range(20)]
        result = _sanitize_citation_refs(items, {1})
        self.assertEqual(len(result), 12)


class ResultFromPayloadWithCitationRefsTests(unittest.TestCase):
    """Test that result_from_payload correctly passes through citationRefs."""

    def test_takeaways_with_citation_refs(self):
        payload = {
            "takeaways": [
                {
                    "text": "Good battery life",
                    "citations": [1],
                    "citationRefs": [
                        {"sourceId": 1, "platform": "bilibili", "platformObjectId": "BV1xx", "startMs": 1000, "endMs": 5000}
                    ],
                }
            ],
            "sections": [],
            "warnings": [],
        }
        result = result_from_payload(payload, {1})
        self.assertEqual(len(result.takeaways), 1)
        self.assertEqual(len(result.takeaways[0].citationRefs), 1)
        self.assertEqual(result.takeaways[0].citationRefs[0].sourceId, 1)

    def test_sections_with_citation_refs(self):
        payload = {
            "takeaways": [],
            "sections": [
                {
                    "id": "test-section",
                    "title": "Test",
                    "kind": "narrative",
                    "body": "Test body",
                    "sourceIds": [1],
                    "citationRefs": [
                        {"sourceId": 1, "platform": "youtube", "platformObjectId": "vid1", "startMs": 2000}
                    ],
                }
            ],
            "warnings": [],
        }
        result = result_from_payload(payload, {1})
        self.assertEqual(len(result.sections), 1)
        self.assertEqual(len(result.sections[0].citationRefs), 1)


class SegmentLevelEvidenceTests(unittest.TestCase):
    """Test that prepare_evidence_batch creates segment-level EvidenceUnits with timestamps."""

    def _make_batch_with_segments(self) -> AcquisitionBatch:
        source = Source(
            id=1,
            platform="bilibili",
            title="Phone review",
            summary="Great phone with good battery",
            url="https://example.com/1",
            transcriptSegments=[
                TranscriptSegment(text="The battery lasts 12 hours of continuous use.", start="10:30", end="11:00"),
                TranscriptSegment(text="Camera quality is excellent in low light.", start="15:00", end="15:30"),
            ],
            comments=[
                Comment(text="I agree about the battery", author="user1"),
            ],
        )
        artifact = Artifact(
            id="artifact-1",
            source_id=1,
            content_hash="abc123",
            captured_at="2026-01-01T00:00:00",
            collector_version="1.0",
        )
        return AcquisitionBatch(
            sources=(source,),
            raw_artifacts=(artifact,),
            collection_logs=(RunLog(provider="mock", platform="bilibili", ok=True, count=1, query="test"),),
        )

    def test_creates_segment_level_units_with_timestamps(self):
        batch = self._make_batch_with_segments()
        evidence_batch, _ = prepare_evidence_batch(batch)
        segment_units = [u for u in evidence_batch.evidence_units if u.modality == "speech"]
        self.assertGreaterEqual(len(segment_units), 2)
        for unit in segment_units:
            self.assertIsNotNone(unit.start_ms)
            self.assertIsNotNone(unit.end_ms)
            self.assertEqual(unit.extraction_method, "transcript-segment")

    def test_creates_comment_level_units(self):
        batch = self._make_batch_with_segments()
        evidence_batch, _ = prepare_evidence_batch(batch)
        comment_units = [u for u in evidence_batch.evidence_units if u.modality == "comment"]
        self.assertGreaterEqual(len(comment_units), 1)
        self.assertTrue(comment_units[0].comment_id)

    def test_segment_timestamps_correctly_parsed(self):
        batch = self._make_batch_with_segments()
        evidence_batch, _ = prepare_evidence_batch(batch)
        speech_units = [u for u in evidence_batch.evidence_units if u.modality == "speech"]
        if speech_units:
            first = speech_units[0]
            # "10:30" should be 630000 ms
            self.assertEqual(first.start_ms, 630000)
            self.assertEqual(first.end_ms, 660000)

    def test_fallback_coarse_unit_when_no_segments(self):
        source = Source(
            id=1,
            platform="youtube",
            title="Review",
            summary="A summary",
            url="https://example.com/1",
        )
        artifact = Artifact(
            id="artifact-1",
            source_id=1,
            content_hash="abc123",
            captured_at="2026-01-01T00:00:00",
            collector_version="1.0",
        )
        batch = AcquisitionBatch(
            sources=(source,),
            raw_artifacts=(artifact,),
            collection_logs=(),
        )
        evidence_batch, _ = prepare_evidence_batch(batch)
        self.assertGreaterEqual(len(evidence_batch.evidence_units), 1)
        unit = evidence_batch.evidence_units[0]
        self.assertEqual(unit.extraction_method, "legacy-source-adapter")


class AsyncLLMTransportTests(unittest.IsolatedAsyncioTestCase):
    """Test async LLM transport module."""

    async def test_post_with_retry_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "test"}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("app.services.llm_transport.post_chat_completion_async", return_value=mock_response):
            result = await post_with_retry(
                {"model": "test", "messages": []},
                api_key="test-key",
                max_retries=0,
            )
            self.assertEqual(result, mock_response)

    async def test_gather_independent_calls_concurrent(self):
        call_order: list[str] = []

        async def mock_call(payload, **kwargs):
            idx = payload.get("idx", 0); call_order.append(f"start-{idx}")
            await asyncio.sleep(0.01)
            call_order.append(f"end-{idx}")
            return f"result-{idx}"

        payloads = [{"idx": 0}, {"idx": 1}, {"idx": 2}]
        with patch("app.services.llm_transport.post_with_retry", side_effect=mock_call):
            results = await gather_independent_calls(payloads, max_concurrency=3)

        self.assertEqual(len(results), 3)
        # All should have started before any finished (concurrent execution)
        starts = [e for e in call_order if e.startswith("start")]
        ends = [e for e in call_order if e.startswith("end")]
        self.assertEqual(len(starts), 3)
        self.assertEqual(len(ends), 3)

    async def test_post_with_retry_raises_on_persistent_failure(self):
        import httpx
        with patch("app.services.llm_transport.post_chat_completion_async", side_effect=httpx.ConnectError("network error")):
            with self.assertRaises(LLMTransportError):
                await post_with_retry(
                    {"model": "test"},
                    api_key="test-key",
                    max_retries=2,
                    backoff_base=0.01,
                )


class HarnessGranularOperatorTests(unittest.IsolatedAsyncioTestCase):
    """Test that the harness split creates granular resumable operators."""

    async def test_harness_creates_four_resumable_operators(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            store = FileHarnessStore(Path(tmp.name))
            request = ResearchRequest(
                need="Compare Phone A battery",
                query="Phone A battery review",
                platforms=["bilibili", "youtube"],
                useLiveProviders=True,
                lang="en",
            )
            plan, _ = build_research_plan(request)
            acquisition = create_acquisition_batch(
                [
                    Source(
                        id=1, platform="bilibili", title="Review A", summary="Battery is stable.",
                        url="https://example.com/1", provider="mock",
                        metrics={"evidence_quality_label": "Moderate"},
                        quality={"label": "Moderate"}, collectedAt="2026-01-01T00:00:00",
                    ),
                    Source(
                        id=2, platform="youtube", title="Review B", summary="Battery degrades.",
                        url="https://example.com/2", provider="mock",
                        metrics={"evidence_quality_label": "Moderate"},
                        quality={"label": "Moderate"}, collectedAt="2026-01-01T00:00:00",
                    ),
                ],
                [
                    RunLog(provider="mock", platform="bilibili", ok=True, count=1, query="battery"),
                    RunLog(provider="mock", platform="youtube", ok=True, count=1, query="battery"),
                ],
            )
            with patch.dict(os.environ, {"VOXLENS_ENABLE_LLM": "false"}, clear=False):
                result = await complete_research_pipeline(
                    request=request,
                    plan=plan,
                    acquisition_batch=acquisition,
                    budget=ResearchBudget(),
                    run_id="run-granular-test",
                    harness_store=store,
                )

            manifest = store.read_manifest("run-granular-test")
            expected_ops = {"coverage_gate", "evidence_structuring", "report_synthesis", "claim_grounding"}
            self.assertEqual(set(manifest.operators.keys()), expected_ops)
            for name in expected_ops:
                self.assertEqual(manifest.operators[name].status, "completed")

            self.assertIsNotNone(result.verified_report)
            self.assertGreater(len(result.evidence_batch.evidence_units), 0)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
