from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from app.agents.artifacts import PlanSpec
from app.agents.budget import ResearchBudget
from app.agents.coordinator import complete_research_pipeline, create_acquisition_batch
from app.agents.grounding_verifier import ClaimVerification
from app.agents.planner import build_research_plan
from app.harness import (
    FileHarnessStore,
    HarnessExecutionError,
    HarnessGraphError,
    HarnessOperator,
    HarnessReplayError,
    HarnessStoreError,
    OperatorBlocked,
    OperatorRecord,
    OperatorSkipped,
    ResearchHarness,
)
from app.harness.serialization import jsonable, payload_hash
from app.models import Artifact, Claim, ClaimEvidence, EvidenceUnit, ResearchRequest, RunLog, Source
from app.services.research_stream import _chunks, _initial_stages, _stream_report


class FileHarnessStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = FileHarnessStore(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_store_writes_manifest_checkpoint_and_events(self) -> None:
        manifest = self.store.initialize("run-1", {"query": "battery review"})
        checkpoint = self.store.write_checkpoint(
            "run-1",
            "evidence",
            version="1",
            input_hash="input-1",
            payload={"count": 2},
        )
        event = self.store.append_event(
            "run-1",
            {"type": "operator_completed", "operator": "evidence"},
        )

        self.assertEqual(manifest.schemaVersion, 1)
        self.assertEqual(self.store.read_checkpoint("run-1", "evidence"), checkpoint)
        self.assertEqual(checkpoint.payload, {"count": 2})
        self.assertEqual(event["sequence"], 1)
        self.assertEqual(list(self.store.iter_events("run-1"))[-1]["operator"], "evidence")
        self.assertFalse(list((self.root / "run-1").glob("*.tmp-*")))

    def test_event_cannot_override_store_owned_fields(self) -> None:
        first = self.store.append_event("run-1", {
            "type": "operator_completed",
            "schemaVersion": 2,
            "sequence": 99,
            "writtenAt": "spoofed",
        })
        second = self.store.append_event("run-1", {"type": "operator_started"})

        self.assertEqual(first["schemaVersion"], 1)
        self.assertEqual(first["sequence"], 1)
        self.assertNotEqual(first["writtenAt"], "spoofed")
        self.assertEqual(second["sequence"], 2)

    def test_corrupt_checkpoint_fails_closed(self) -> None:
        self.store.initialize("run-1")
        checkpoint_path = self.root / "run-1" / "checkpoints" / "evidence.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("{broken", encoding="utf-8")

        with self.assertRaises(HarnessStoreError):
            self.store.read_checkpoint("run-1", "evidence")

    def test_operator_name_must_be_path_safe(self) -> None:
        self.store.initialize("run-1")

        with self.assertRaises(ValueError):
            self.store.write_checkpoint(
                "run-1",
                "../evidence",
                version="1",
                input_hash="input-1",
                payload={},
            )

    def test_unknown_manifest_schema_fails_closed(self) -> None:
        self.store.initialize("run-1")
        manifest_path = self.root / "run-1" / "harness.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["schemaVersion"] = 2
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaises(HarnessStoreError):
            self.store.read_manifest("run-1")

    def test_run_id_must_be_path_safe_without_normalization(self) -> None:
        for run_id in ("a/b", "!!!", ""):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                self.store.initialize(run_id)

        self.assertFalse((self.root / "ab").exists())

    def test_set_serialization_is_deterministic(self) -> None:
        self.assertEqual(jsonable({"b", "a"}), ["a", "b"])

    def test_set_serialization_inside_pydantic_model_is_deterministic(self) -> None:
        class Payload(BaseModel):
            values: set[str]

        self.assertEqual(jsonable(Payload(values={"b", "a"})), {"values": ["a", "b"]})

    def test_checkpoint_normalizes_structured_payload_before_hashing_and_writing(self) -> None:
        class Payload(BaseModel):
            values: set[str]

        self.store.write_checkpoint(
            "run-1",
            "evidence",
            version="1",
            input_hash="input-1",
            payload=Payload(values={"b", "a"}),
        )

        checkpoint = self.store.read_checkpoint("run-1", "evidence")
        self.assertEqual(checkpoint.payload, {"values": ["a", "b"]})

    def test_first_checkpoint_fsyncs_new_directory_and_run_directory(self) -> None:
        self.store.initialize("run-1")

        with patch.object(self.store, "_fsync_directory") as fsync_directory:
            self.store.write_checkpoint(
                "run-1",
                "evidence",
                version="1",
                input_hash="input-1",
                payload={"count": 1},
            )

        synced = {call.args[0] for call in fsync_directory.call_args_list}
        self.assertIn(self.root / "run-1" / "checkpoints", synced)
        self.assertIn(self.root / "run-1", synced)

    def test_domain_dataclass_with_read_only_mapping_is_serializable(self) -> None:
        plan = PlanSpec(
            query="battery",
            entities=("Phone A",),
            comparison_targets=(),
            time_range="",
            evidence_intents=("speech",),
            platform_queries={"youtube": "Phone A battery"},
        )

        payload = jsonable(plan)
        self.assertEqual(payload["platform_queries"], {"youtube": "Phone A battery"})

    def test_unsupported_serialization_fails_closed(self) -> None:
        with self.assertRaises(TypeError):
            jsonable(object())

    def test_mapping_keys_must_be_stable_and_collision_free(self) -> None:
        self.assertEqual(jsonable({1: "numeric"}), {"1": "numeric"})

        with self.assertRaises(TypeError):
            jsonable({object(): "unsupported"})
        with self.assertRaises(ValueError):
            jsonable({1: "numeric", "1": "string"})

    def test_operator_record_name_must_be_path_safe(self) -> None:
        self.store.initialize("run-1")

        with self.assertRaises(ValueError):
            self.store.update_operator("run-1", OperatorRecord(name="../evidence"))


class ResearchHarnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = FileHarnessStore(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    async def test_runs_dependencies_in_deterministic_order(self) -> None:
        calls: list[str] = []

        async def plan(_context):
            calls.append("plan")
            return {"query": "battery"}

        def evidence(context):
            calls.append("evidence")
            return {"query": context.get("plan")["query"], "count": 2}

        outputs = await ResearchHarness(self.store, "run-1").run([
            HarnessOperator(name="evidence", dependencies=("plan",), run=evidence),
            HarnessOperator(name="plan", run=plan),
        ])

        self.assertEqual(calls, ["plan", "evidence"])
        self.assertEqual(outputs["evidence"], {"query": "battery", "count": 2})

    async def test_resume_loads_checkpoint_without_invoking_operator(self) -> None:
        calls = 0

        def run_once(_context):
            nonlocal calls
            calls += 1
            return {"query": "battery"}

        operator = HarnessOperator(
            name="plan",
            run=run_once,
            dump=lambda value: value,
            load=lambda payload: payload,
            resumable=True,
        )
        first = await ResearchHarness(self.store, "run-1").run([operator])
        second = await ResearchHarness(self.store, "run-1").run([operator], mode="resume")

        self.assertEqual(first, second)
        self.assertEqual(calls, 1)
        manifest = self.store.read_manifest("run-1")
        self.assertEqual(manifest.operators["plan"].status, "resumed")

    async def test_version_change_invalidates_checkpoint(self) -> None:
        calls: list[str] = []

        def run_v1(_context):
            calls.append("v1")
            return {"version": 1}

        def run_v2(_context):
            calls.append("v2")
            return {"version": 2}

        await ResearchHarness(self.store, "run-1").run([
            HarnessOperator(name="plan", version="1", run=run_v1, load=lambda payload: payload, resumable=True),
        ])
        outputs = await ResearchHarness(self.store, "run-1").run([
            HarnessOperator(name="plan", version="2", run=run_v2, load=lambda payload: payload, resumable=True),
        ])

        self.assertEqual(calls, ["v1", "v2"])
        self.assertEqual(outputs["plan"], {"version": 2})

    async def test_failure_is_recorded_and_stops_execution(self) -> None:
        def fail(_context):
            raise RuntimeError("provider failed")

        with self.assertRaises(HarnessExecutionError):
            await ResearchHarness(self.store, "run-1").run([
                HarnessOperator(name="acquire", run=fail),
                HarnessOperator(name="evidence", dependencies=("acquire",), run=lambda _context: {}),
            ])

        manifest = self.store.read_manifest("run-1")
        self.assertEqual(manifest.operators["acquire"].status, "failed")
        self.assertNotIn("evidence", manifest.operators)

    async def test_dump_failure_is_recorded_as_operator_failure(self) -> None:
        def broken_dump(_value):
            raise ValueError("cannot serialize output")

        with self.assertRaises(HarnessExecutionError):
            await ResearchHarness(self.store, "run-1").run([
                HarnessOperator(name="evidence", run=lambda _context: {"count": 1}, dump=broken_dump),
            ])

        manifest = self.store.read_manifest("run-1")
        self.assertEqual(manifest.operators["evidence"].status, "failed")
        self.assertIn("cannot serialize output", manifest.operators["evidence"].message)

    async def test_blocked_operator_records_policy_state(self) -> None:
        def block(_context):
            raise OperatorBlocked("rights basis missing")

        with self.assertRaises(OperatorBlocked):
            await ResearchHarness(self.store, "run-1").run([
                HarnessOperator(name="permission", run=block),
            ])

        manifest = self.store.read_manifest("run-1")
        self.assertEqual(manifest.operators["permission"].status, "blocked")
        self.assertEqual(manifest.operators["permission"].message, "rights basis missing")

    async def test_skipped_operator_records_stable_output_for_dependents(self) -> None:
        def skip(_context):
            raise OperatorSkipped("not required")

        outputs = await ResearchHarness(self.store, "run-1").run([
            HarnessOperator(name="optional", run=skip),
            HarnessOperator(
                name="consumer",
                dependencies=("optional",),
                run=lambda context: {"optional": context.get("optional")},
            ),
        ])

        manifest = self.store.read_manifest("run-1")
        self.assertEqual(outputs["consumer"], {"optional": None})
        self.assertEqual(manifest.operators["optional"].status, "skipped")
        self.assertEqual(manifest.operators["optional"].outputHash, payload_hash(None))

    async def test_unknown_dependency_and_cycle_fail_before_execution(self) -> None:
        calls: list[str] = []

        with self.assertRaises(HarnessGraphError):
            await ResearchHarness(self.store, "run-1").run([
                HarnessOperator(name="evidence", dependencies=("missing",), run=lambda _context: calls.append("run")),
            ])

        with self.assertRaises(HarnessGraphError):
            await ResearchHarness(self.store, "run-2").run([
                HarnessOperator(name="a", dependencies=("b",), run=lambda _context: calls.append("a")),
                HarnessOperator(name="b", dependencies=("a",), run=lambda _context: calls.append("b")),
            ])

        self.assertEqual(calls, [])

    async def test_invalid_operator_name_fails_before_execution(self) -> None:
        calls: list[str] = []

        with self.assertRaises(ValueError):
            HarnessOperator(name="../plan", run=lambda _context: calls.append("run"))

        self.assertEqual(calls, [])
        self.assertFalse((self.root / "run-1").exists())

    async def test_replay_hashes_dependencies_by_checkpoint_output(self) -> None:
        class StageValue:
            def __init__(self, value: str) -> None:
                self.value = value

        calls: list[str] = []

        source = HarnessOperator(
            name="source",
            run=lambda _context: calls.append("source") or StageValue("battery"),
            dump=lambda value: {"value": value.value},
            load=lambda payload: StageValue(payload["value"]),
            resumable=True,
        )
        summarize = HarnessOperator(
            name="summarize",
            dependencies=("source",),
            run=lambda context: calls.append("summarize") or {"value": context.get("source").value},
            load=lambda payload: payload,
            resumable=True,
        )

        await ResearchHarness(self.store, "run-1").run([source, summarize], mode="execute")
        outputs = await ResearchHarness(self.store, "run-1").run([source, summarize], mode="replay")

        self.assertEqual(calls, ["source", "summarize"])
        self.assertEqual(outputs["summarize"], {"value": "battery"})

    async def test_corrupt_resume_checkpoint_does_not_run_operator(self) -> None:
        calls = 0
        self.store.initialize("run-1")
        checkpoint_path = self.root / "run-1" / "checkpoints" / "plan.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"schemaVersion": 1, "payload": "missing fields"}), encoding="utf-8")

        def run_once(_context):
            nonlocal calls
            calls += 1
            return {}

        with self.assertRaises(HarnessStoreError):
            await ResearchHarness(self.store, "run-1").run([
                HarnessOperator(name="plan", run=run_once, load=lambda payload: payload, resumable=True),
            ])

        self.assertEqual(calls, 0)

    async def test_replay_never_invokes_operator_and_requires_matching_checkpoint(self) -> None:
        calls = 0

        def execute(_context):
            nonlocal calls
            calls += 1
            return {"query": "battery"}

        operator = HarnessOperator(
            name="plan",
            run=execute,
            dump=lambda value: value,
            load=lambda payload: payload,
            resumable=True,
        )
        await ResearchHarness(self.store, "run-1").run([operator], mode="execute")

        def forbidden(_context):
            raise AssertionError("replay invoked an operator")

        replay_operator = HarnessOperator(
            name="plan",
            run=forbidden,
            dump=lambda value: value,
            load=lambda payload: payload,
            resumable=True,
        )
        outputs = await ResearchHarness(self.store, "run-1").run([replay_operator], mode="replay")
        self.assertEqual(outputs["plan"], {"query": "battery"})
        self.assertEqual(calls, 1)

        with self.assertRaises(HarnessReplayError):
            await ResearchHarness(self.store, "missing-run").run([replay_operator], mode="replay")
        self.assertFalse((self.root / "missing-run").exists())


class PipelineSnapshotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = FileHarnessStore(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    async def test_pipeline_snapshot_persists_evidence_domain_objects(self) -> None:
        result = await self._pipeline_result("run-snapshot")

        self.store.write_pipeline_snapshot("run-snapshot", result)
        snapshot = self.store.read_pipeline_snapshot("run-snapshot")
        self.assertIsNotNone(snapshot)

        expected = {
            "artifacts.jsonl": (Artifact, result.acquisition_batch.raw_artifacts),
            "evidence.jsonl": (EvidenceUnit, result.evidence_batch.evidence_units),
            "claims.jsonl": (Claim, result.claim_set.claims),
            "claim-evidence.jsonl": (ClaimEvidence, result.claim_set.claim_evidence_links),
            "verifications.jsonl": (ClaimVerification, result.verified_report.verification_results),
        }
        for filename, (model_type, originals) in expected.items():
            records = snapshot["files"][filename]
            self.assertEqual(len(records), len(originals))
            self.assertTrue(all(record["schemaVersion"] == 1 for record in records))
            self.assertTrue(all(record["payloadHash"] for record in records))
            for record, original in zip(records, originals, strict=True):
                payload = record["payload"]
                if model_type is ClaimVerification:
                    rebuilt = ClaimVerification(
                        **{
                            **payload,
                            "supporting_evidence_ids": tuple(payload["supporting_evidence_ids"]),
                            "contradicting_evidence_ids": tuple(payload["contradicting_evidence_ids"]),
                        }
                    )
                else:
                    rebuilt = model_type.model_validate(payload)
                self.assertEqual(jsonable(rebuilt), jsonable(original))

    async def test_complete_pipeline_uses_harness_and_persists_snapshot(self) -> None:
        result = await self._pipeline_result("run-integrated", harness_store=self.store)

        manifest = self.store.read_manifest("run-integrated")
        # The pipeline is split into granular resumable operators
        for operator_name in ("coverage_gate", "evidence_structuring", "report_synthesis", "claim_grounding"):
            self.assertEqual(manifest.operators[operator_name].status, "completed")
        checkpoint = self.store.read_checkpoint("run-integrated", "claim_grounding")
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint.payload["reportStatus"], result.verified_report.report.status)
        self.assertEqual(
            len(self.store.read_jsonl("run-integrated", "evidence.jsonl")),
            len(result.evidence_batch.evidence_units),
        )

    async def test_pipeline_checkpoint_input_hash_covers_acquired_source_content(self) -> None:
        await self._pipeline_result("run-input", harness_store=self.store)
        first_hash = self.store.read_checkpoint("run-input", "coverage_gate").inputHash
        first_initial_hash = self.store.read_manifest("run-input").metadata["initialHash"]

        await self._pipeline_result(
            "run-input",
            harness_store=self.store,
            second_summary="Battery degrades rapidly under heavy use.",
        )
        second_hash = self.store.read_checkpoint("run-input", "coverage_gate").inputHash
        second_initial_hash = self.store.read_manifest("run-input").metadata["initialHash"]

        self.assertNotEqual(first_hash, second_hash)
        self.assertNotEqual(first_initial_hash, second_initial_hash)

    async def test_pipeline_snapshot_pointer_is_unchanged_on_partial_write_failure(self) -> None:
        result = await self._pipeline_result("run-atomic")
        self.store.write_pipeline_snapshot("run-atomic", result)
        pointer_path = self.root / "run-atomic" / "snapshot.json"
        pointer_before = pointer_path.read_text(encoding="utf-8")
        original_write = self.store._write_jsonl_path

        def fail_on_claims(path, kind, values):
            if path.name == "claims.jsonl":
                raise OSError("disk full")
            return original_write(path, kind, values)

        with patch.object(self.store, "_write_jsonl_path", side_effect=fail_on_claims):
            with self.assertRaises(OSError):
                self.store.write_pipeline_snapshot("run-atomic", result)

        self.assertEqual(pointer_path.read_text(encoding="utf-8"), pointer_before)
        self.assertEqual(
            len(self.store.read_jsonl("run-atomic", "claims.jsonl")),
            len(result.claim_set.claims),
        )

    async def test_pipeline_snapshot_pointer_requires_committed_timestamp(self) -> None:
        result = await self._pipeline_result("run-pointer-schema")
        self.store.write_pipeline_snapshot("run-pointer-schema", result)
        pointer_path = self.root / "run-pointer-schema" / "snapshot.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        pointer.pop("committedAt")
        pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

        with self.assertRaises(HarnessStoreError):
            self.store.read_pipeline_snapshot("run-pointer-schema")

    async def test_pipeline_snapshot_batch_read_stays_on_one_generation(self) -> None:
        first = await self._pipeline_result("run-consistent")
        second = replace(
            first,
            acquisition_batch=replace(first.acquisition_batch, raw_artifacts=()),
            evidence_batch=replace(first.evidence_batch, evidence_units=()),
            claim_set=replace(first.claim_set, claims=(), claim_evidence_links=()),
            verified_report=replace(first.verified_report, verification_results=()),
        )
        self.store.write_pipeline_snapshot("run-consistent", first)
        original_read = self.store._read_jsonl_path
        published_second = False

        def publish_during_read(run_id, filename, path):
            nonlocal published_second
            records = original_read(run_id, filename, path)
            if not published_second:
                published_second = True
                self.store.write_pipeline_snapshot("run-consistent", second)
            return records

        with patch.object(self.store, "_read_jsonl_path", side_effect=publish_during_read):
            snapshot = self.store.read_pipeline_snapshot("run-consistent")

        self.assertIsNotNone(snapshot)
        files = snapshot["files"]
        self.assertEqual(len(files["artifacts.jsonl"]), len(first.acquisition_batch.raw_artifacts))
        self.assertEqual(len(files["evidence.jsonl"]), len(first.evidence_batch.evidence_units))
        self.assertEqual(len(files["claims.jsonl"]), len(first.claim_set.claims))
        self.assertEqual(
            len(files["claim-evidence.jsonl"]),
            len(first.claim_set.claim_evidence_links),
        )
        self.assertEqual(
            len(files["verifications.jsonl"]),
            len(first.verified_report.verification_results),
        )
        active = self.store.read_pipeline_snapshot("run-consistent")
        self.assertEqual(len(active["files"]["artifacts.jsonl"]), 0)

    async def test_pipeline_snapshot_fsyncs_generation_and_pointer_directories(self) -> None:
        result = await self._pipeline_result("run-durable")

        with patch.object(self.store, "_fsync_directory") as fsync_directory:
            self.store.write_pipeline_snapshot("run-durable", result)

        synced = {call.args[0] for call in fsync_directory.call_args_list}
        run_dir = self.root / "run-durable"
        self.assertIn(run_dir, synced)
        self.assertIn(run_dir / "snapshots", synced)

    async def test_report_stream_preserves_public_event_order(self) -> None:
        result = await self._pipeline_result("run-events")
        report = result.verified_report.report.model_copy(deep=True)

        with patch("app.services.research_stream.time.sleep", return_value=None):
            raw_events = list(_stream_report(report, _initial_stages(report.lang)))

        names = [event.splitlines()[0].removeprefix("event: ") for event in raw_events]
        expected = ["outline", "report_patch"]
        for section in report.sections:
            expected.append("section_started")
            expected.extend("section_delta" for _ in _chunks(section.body, 72))
            expected.append("section_complete")
        expected.extend(["stage", "final_report"])
        self.assertEqual(names, expected)

    async def _pipeline_result(
        self,
        run_id: str,
        harness_store=None,
        second_summary: str = "Battery is not stable under heavy use.",
    ):
        request = ResearchRequest(
            need="Compare Phone A battery experience and risks",
            query="Phone A battery review comparison",
            platforms=["bilibili", "youtube"],
            useLiveProviders=True,
            lang="en",
        )
        plan, _ = build_research_plan(request)
        acquisition = create_acquisition_batch(
            [
                _source(1, "bilibili", "Phone A battery review", "Battery is stable in daily use."),
                _source(2, "youtube", "Phone A long-term review", second_summary),
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
                run_id=run_id,
                harness_store=harness_store,
            )
        return result


def _source(source_id: int, platform: str, title: str, summary: str) -> Source:
    return Source(
        id=source_id,
        platform=platform,
        title=title,
        summary=summary,
        url=f"https://example.test/{platform}/{source_id}",
        provider="mock",
        metrics={"evidence_quality_label": "Moderate"},
        quality={"label": "Moderate"},
        collectedAt="2026-07-22T10:00:00",
    )


if __name__ == "__main__":
    unittest.main()
