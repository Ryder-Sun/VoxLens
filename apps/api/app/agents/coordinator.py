from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Coroutine, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar
from uuid import uuid4

from app.agents.artifacts import AcquisitionBatch, ClaimSet, EvidenceBatch, PlanSpec, VerifiedReport
from app.agents.budget import ResearchBudget, StopPolicy
from app.agents.coverage_gate import CoverageGate, CoverageReport
from app.agents.crawler import crawl_sources
from app.agents.evidence import assess_evidence_quality, prepare_evidence
from app.agents.grounding_verifier import ClaimGroundingVerifier, ClaimVerification
from app.agents.llm_orchestrator import LLMCallSpec, LLMOrchestrator
from app.agents.planner import analyze_research_question, build_research_plan
from app.harness import FileHarnessStore, HarnessContext, HarnessOperator, ResearchHarness
from app.models import (
    AgentStep,
    Artifact,
    Claim,
    ClaimEvidence,
    EvidenceQualityLabel,
    EvidenceUnit,
    ResearchPlan,
    ResearchReport,
    ResearchRequest,
    RunLog,
    Source,
)
from app.services.asr_pipeline import plan_asr_candidates
from app.services.llm_synthesis import synthesize_with_llm, synthesize_with_llm_async
from app.services.llm_types import LlmSynthesisResult
from app.services.report_builder import build_report, infer_report_kind
from app.utils import dedupe_sources_key, text_excerpt


_QUALITY_RANK = {"Strong": 4, "Moderate": 3, "Weak": 2, "Insufficient": 1}
_REPORT_CONFIDENCE: dict[str, EvidenceQualityLabel] = {
    "high": "Strong",
    "medium": "Moderate",
    "low": "Weak",
    "insufficient": "Insufficient",
}
_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class PipelineResult:
    """Return the shared post-acquisition pipeline result for sync and stream paths."""

    acquisition_batch: AcquisitionBatch
    evidence_batch: EvidenceBatch
    claim_set: ClaimSet
    verified_report: VerifiedReport
    steps: tuple[AgentStep, ...]


def run_agentic_research(request: ResearchRequest) -> ResearchReport:
    """Run the shared evidence-first pipeline while preserving the API response model."""

    trace: list[AgentStep] = []
    query = (request.query or request.need).strip()
    run_id = f"run-{uuid4().hex[:12]}"
    legacy_plan, planner_step = build_research_plan(request)
    trace.append(planner_step)

    if not request.useLiveProviders:
        report = build_report(
            need=request.need,
            query=query,
            lang=request.lang,
            sources=[],
            run_logs=[],
            is_demo=True,
            generated_at=datetime.now(),
            run_id=run_id,
            research_mode=request.researchMode,
            enable_llm_synthesis=False,
        )
        trace.append(AgentStep(
            name="SocialCrawlerAgent",
            role="Live provider execution",
            status="skipped",
            message="Live providers disabled; returned a query-specific planning report without factual claims.",
        ))
        report.agentTrace = trace
        report.isDemoFallback = True
        report.plan = legacy_plan
        return report

    budget = create_research_budget(request)
    acquisition_batch, crawler_step = acquire_sources(request, legacy_plan, budget)
    trace.append(crawler_step)
    result = _run_async(complete_research_pipeline(
        request=request,
        plan=legacy_plan,
        acquisition_batch=acquisition_batch,
        budget=budget,
        run_id=run_id,
    ))
    trace.extend(result.steps)
    report = result.verified_report.report
    report.plan = legacy_plan
    report.agentTrace = trace
    return report


def create_research_budget(request: ResearchRequest) -> ResearchBudget:
    """Create the single shared budget ledger used by every research stage."""

    return ResearchBudget(
        max_sources=50,
        min_sources_for_synthesis=request.minLiveSources or 3,
        min_platforms_for_synthesis=min(2, len(request.platforms)),
    )


async def complete_research_pipeline(
    *,
    request: ResearchRequest,
    plan: ResearchPlan,
    acquisition_batch: AcquisitionBatch,
    budget: ResearchBudget,
    run_id: str,
    harness_store: FileHarnessStore | None = None,
) -> PipelineResult:
    """Run the post-acquisition pipeline with granular resumable harness operators.

    When harness_store is provided the pipeline is split into individually
    checkpointable operators (coverage, evidence, synthesis, verification) so
    that a crash or timeout can resume from the last completed stage instead of
    restarting from scratch. When no store is provided, falls back to direct
    execution with no checkpointing.
    """

    if harness_store is None:
        return await _complete_research_pipeline(
            request=request,
            plan=plan,
            acquisition_batch=acquisition_batch,
            budget=budget,
            run_id=run_id,
        )

    # --- Granular harness operators with dependencies ---
    # coverage_gate → evidence_structuring → report_synthesis → claim_grounding

    async def _run_coverage(ctx: HarnessContext) -> dict[str, Any]:
        plan_spec = create_plan_spec(request, plan)
        coverage_gate = CoverageGate()
        acquisition = acquisition_batch
        stop_policy = StopPolicy()

        coverage = coverage_gate.evaluate(acquisition, plan_spec)
        steps: list[AgentStep] = []

        if (
            coverage_gate.should_block_synthesis(coverage, budget)
            and stop_policy.should_continue_search(acquisition, budget)
        ):
            retry_plan = _supplemental_plan(plan, coverage, budget)
            if retry_plan.targets:
                retry_batch, retry_step = acquire_sources(request, retry_plan, budget)
                acquisition = merge_acquisition_batches(acquisition, retry_batch, budget.max_sources)
                budget.set_sources_collected(len(acquisition.sources))
                coverage = coverage_gate.evaluate(acquisition, plan_spec)
                steps.append(retry_step.model_copy(update={
                    "name": "SupplementalCrawlerAgent",
                    "role": "Retries uncovered platforms or evidence gaps within the shared research budget.",
                }))

        blocked = coverage_gate.should_block_synthesis(coverage, budget)
        steps.append(_coverage_step(coverage, coverage_gate.identify_gaps(coverage, plan_spec), blocked))
        return {
            "acquisition_batch": acquisition,
            "coverage": coverage,
            "blocked": blocked,
            "steps": steps,
        }

    def _dump_coverage(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "sources": len(result["acquisition_batch"].sources),
            "blocked": result["blocked"],
            "platformsCovered": len(result["coverage"].platforms_covered),
        }

    async def _run_evidence(ctx: HarnessContext) -> dict[str, Any]:
        coverage_result = ctx.get("coverage_gate")
        acquisition = coverage_result["acquisition_batch"]
        evidence_batch, evidence_step = prepare_evidence_batch(acquisition)
        return {
            "acquisition_batch": acquisition,
            "coverage": coverage_result["coverage"],
            "blocked": coverage_result["blocked"],
            "steps": [*coverage_result["steps"], evidence_step],
            "evidence_batch": evidence_batch,
        }

    def _dump_evidence(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "evidenceUnits": len(result["evidence_batch"].evidence_units),
            "sources": len(result["acquisition_batch"].sources),
        }

    async def _run_synthesis(ctx: HarnessContext) -> dict[str, Any]:
        evidence_result = ctx.get("evidence_structuring")
        acquisition = evidence_result["acquisition_batch"]
        evidence_batch = evidence_result["evidence_batch"]
        coverage = evidence_result["coverage"]
        blocked = evidence_result["blocked"]
        steps = list(evidence_result["steps"])

        min_sources = request.minLiveSources or 1
        if len(acquisition.sources) < min_sources or StopPolicy().should_stop_for_insufficient_evidence(acquisition):
            steps.append(AgentStep(
                name="FallbackGuardAgent",
                role="Prevents ungrounded synthesis when live evidence is below the requested threshold.",
                status="partial",
                message=(
                    f"Only {len(acquisition.sources)} live sources collected; requested at least {min_sources}. "
                    "Building an evidence-limited report."
                ),
                metrics={"liveSources": len(acquisition.sources), "minSources": min_sources},
            ))

        report_sources = _sources_for_report(acquisition, evidence_batch)
        report_orchestrator = LLMOrchestrator(
            budget,
            _report_llm_provider(request, report_sources) if not blocked and _llm_available() else None,
        )
        report_started = time.perf_counter()
        report = await _build_report_with_orchestrator(
            request=request,
            sources=report_sources,
            run_logs=[log.model_copy(deep=True) for log in acquisition.collection_logs],
            run_id=run_id,
            orchestrator=report_orchestrator,
        )
        steps.append(AgentStep(
            name="ReportSynthesisAgent",
            role="Builds the structured report after the coverage gate and within the shared LLM budget.",
            status="partial" if blocked else "ok",
            message=(
                "Coverage gaps blocked LLM synthesis; generated the deterministic evidence-limited report."
                if blocked
                else "Generated the report through the budgeted LLM orchestrator with deterministic fallback."
            ),
            elapsedSec=round(time.perf_counter() - report_started, 3),
            metrics={
                "sections": len(report.sections),
                "takeaways": len(report.takeaways),
                **report_orchestrator.remaining_budget(),
            },
        ))
        return {
            "acquisition_batch": acquisition,
            "evidence_batch": evidence_batch,
            "coverage": coverage,
            "blocked": blocked,
            "steps": steps,
            "report": report,
        }

    def _dump_synthesis(result: dict[str, Any]) -> dict[str, Any]:
        report = result["report"]
        return {
            "sections": len(report.sections),
            "takeaways": len(report.takeaways),
            "reportStatus": report.status,
        }

    async def _run_verification(ctx: HarnessContext) -> PipelineResult:
        synth_result = ctx.get("report_synthesis")
        acquisition = synth_result["acquisition_batch"]
        evidence_batch = synth_result["evidence_batch"]
        coverage = synth_result["coverage"]
        blocked = synth_result["blocked"]
        steps = list(synth_result["steps"])
        report = synth_result["report"]

        claim_set = build_claim_set_from_report(report, evidence_batch, coverage, run_id)
        verifications = ClaimGroundingVerifier().verify_claims(
            list(claim_set.claims),
            list(evidence_batch.evidence_units),
            list(claim_set.claim_evidence_links),
        )
        steps.append(_verification_step(verifications))
        _append_architecture_warnings(report, coverage, blocked, verifications)

        return PipelineResult(
            acquisition_batch=acquisition,
            evidence_batch=evidence_batch,
            claim_set=claim_set,
            verified_report=VerifiedReport(
                report=report,
                verification_results=tuple(verifications),
                coverage_gate_passed=not blocked,
                budget_remaining=budget.remaining(),
            ),
            steps=tuple(steps),
        )

    def _dump_verification(result: PipelineResult) -> dict[str, Any]:
        return _pipeline_checkpoint_payload(result)

    operators = [
        HarnessOperator(
            name="coverage_gate",
            version="1",
            run=_run_coverage,
            dump=_dump_coverage,
            resumable=True,
        ),
        HarnessOperator(
            name="evidence_structuring",
            version="1",
            run=_run_evidence,
            dump=_dump_evidence,
            dependencies=("coverage_gate",),
            resumable=True,
        ),
        HarnessOperator(
            name="report_synthesis",
            version="1",
            run=_run_synthesis,
            dump=_dump_synthesis,
            dependencies=("evidence_structuring",),
            resumable=True,
        ),
        HarnessOperator(
            name="claim_grounding",
            version="1",
            run=_run_verification,
            dump=_dump_verification,
            dependencies=("report_synthesis",),
            resumable=True,
        ),
    ]

    outputs = await ResearchHarness(harness_store, run_id).run(
        operators,
        initial_values=_pipeline_harness_inputs(
            request=request,
            plan=plan,
            acquisition_batch=acquisition_batch,
            budget=budget,
            run_id=run_id,
        ),
        mode="resume",
    )
    pipeline_result: PipelineResult = outputs["claim_grounding"]
    harness_store.write_pipeline_snapshot(run_id, pipeline_result)
    return pipeline_result


async def _complete_research_pipeline(
    *,
    request: ResearchRequest,
    plan: ResearchPlan,
    acquisition_batch: AcquisitionBatch,
    budget: ResearchBudget,
    run_id: str,
) -> PipelineResult:
    """Run the common artifact, coverage, evidence, synthesis, and grounding stages."""

    steps: list[AgentStep] = []
    stop_policy = StopPolicy()
    coverage_gate = CoverageGate()
    plan_spec = create_plan_spec(request, plan)

    if request.crawlMedia:
        _, asr_step = plan_asr_candidates(
            [source.model_copy(deep=True) for source in acquisition_batch.sources],
            top_n=_asr_top_n(request),
        )
        steps.append(asr_step)

    coverage = coverage_gate.evaluate(acquisition_batch, plan_spec)
    if (
        coverage_gate.should_block_synthesis(coverage, budget)
        and stop_policy.should_continue_search(acquisition_batch, budget)
    ):
        retry_plan = _supplemental_plan(plan, coverage, budget)
        if retry_plan.targets:
            retry_batch, retry_step = acquire_sources(request, retry_plan, budget)
            acquisition_batch = merge_acquisition_batches(acquisition_batch, retry_batch, budget.max_sources)
            budget.set_sources_collected(len(acquisition_batch.sources))
            coverage = coverage_gate.evaluate(acquisition_batch, plan_spec)
            steps.append(retry_step.model_copy(update={
                "name": "SupplementalCrawlerAgent",
                "role": "Retries uncovered platforms or evidence gaps within the shared research budget.",
            }))

    blocked = coverage_gate.should_block_synthesis(coverage, budget)
    steps.append(_coverage_step(coverage, coverage_gate.identify_gaps(coverage, plan_spec), blocked))

    evidence_batch, evidence_step = prepare_evidence_batch(acquisition_batch)
    steps.append(evidence_step)

    min_sources = request.minLiveSources or 1
    if len(acquisition_batch.sources) < min_sources or stop_policy.should_stop_for_insufficient_evidence(acquisition_batch):
        steps.append(AgentStep(
            name="FallbackGuardAgent",
            role="Prevents ungrounded synthesis when live evidence is below the requested threshold.",
            status="partial",
            message=(
                f"Only {len(acquisition_batch.sources)} live sources collected; requested at least {min_sources}. "
                "Building an evidence-limited report."
            ),
            metrics={"liveSources": len(acquisition_batch.sources), "minSources": min_sources},
        ))

    report_sources = _sources_for_report(acquisition_batch, evidence_batch)
    report_orchestrator = LLMOrchestrator(
        budget,
        _report_llm_provider(request, report_sources) if not blocked and _llm_available() else None,
    )
    report_started = time.perf_counter()
    report = await _build_report_with_orchestrator(
        request=request,
        sources=report_sources,
        run_logs=[log.model_copy(deep=True) for log in acquisition_batch.collection_logs],
        run_id=run_id,
        orchestrator=report_orchestrator,
    )
    steps.append(AgentStep(
        name="ReportSynthesisAgent",
        role="Builds the structured report after the coverage gate and within the shared LLM budget.",
        status="partial" if blocked else "ok",
        message=(
            "Coverage gaps blocked LLM synthesis; generated the deterministic evidence-limited report."
            if blocked
            else "Generated the report through the budgeted LLM orchestrator with deterministic fallback."
        ),
        elapsedSec=round(time.perf_counter() - report_started, 3),
        metrics={
            "sections": len(report.sections),
            "takeaways": len(report.takeaways),
            **report_orchestrator.remaining_budget(),
        },
    ))

    claim_set = build_claim_set_from_report(report, evidence_batch, coverage, run_id)
    verifications = ClaimGroundingVerifier().verify_claims(
        list(claim_set.claims),
        list(evidence_batch.evidence_units),
        list(claim_set.claim_evidence_links),
    )
    steps.append(_verification_step(verifications))
    _append_architecture_warnings(report, coverage, blocked, verifications)

    verified_report = VerifiedReport(
        report=report,
        verification_results=tuple(verifications),
        coverage_gate_passed=not blocked,
        budget_remaining=budget.remaining(),
    )
    return PipelineResult(
        acquisition_batch=acquisition_batch,
        evidence_batch=evidence_batch,
        claim_set=claim_set,
        verified_report=verified_report,
        steps=tuple(steps),
    )


def _pipeline_checkpoint_payload(result: PipelineResult) -> dict[str, Any]:
    """Persist a compact stage summary while domain entities live in JSONL snapshots."""

    return {
        "sources": len(result.acquisition_batch.sources),
        "artifacts": len(result.acquisition_batch.raw_artifacts),
        "evidenceUnits": len(result.evidence_batch.evidence_units),
        "claims": len(result.claim_set.claims),
        "verifications": len(result.verified_report.verification_results),
        "reportStatus": result.verified_report.report.status,
        "coverageGatePassed": result.verified_report.coverage_gate_passed,
        "budgetRemaining": result.verified_report.budget_remaining,
    }


def _pipeline_harness_inputs(
    *,
    request: ResearchRequest,
    plan: ResearchPlan,
    acquisition_batch: AcquisitionBatch,
    budget: ResearchBudget,
    run_id: str,
) -> dict[str, Any]:
    """Capture every logical input read by the post-acquisition operator."""

    return {
        "runId": run_id,
        "request": request,
        "plan": plan,
        "acquisitionBatch": acquisition_batch,
        "budget": {
            "maxSources": budget.max_sources,
            "maxAsrMinutes": budget.max_asr_minutes,
            "maxLlmCalls": budget.max_llm_calls,
            "maxLlmTokens": budget.max_llm_tokens,
            "maxElapsedSeconds": budget.max_elapsed_seconds,
            "minSourcesForSynthesis": budget.min_sources_for_synthesis,
            "minPlatformsForSynthesis": budget.min_platforms_for_synthesis,
            "sourcesCollected": budget.sources_collected,
            "asrMinutesUsed": budget.asr_minutes_used,
            "llmCallsUsed": budget.llm_calls_used,
            "llmTokensUsed": budget.llm_tokens_used,
        },
    }


def create_plan_spec(request: ResearchRequest, plan: ResearchPlan) -> PlanSpec:
    """Convert the legacy planner model into the immutable planning contract."""

    analysis = analyze_research_question((request.query or request.need).strip(), request.lang)
    platform_queries = {
        str(target.platform): target.query
        for target in plan.targets
        if target.role != "fallback"
    }
    return PlanSpec(
        query=analysis.canonical_query,
        entities=analysis.entities,
        comparison_targets=analysis.comparison_objects,
        time_range=analysis.time_range,
        evidence_intents=analysis.evidence_types,
        platform_queries=platform_queries,
    )


def acquire_sources(
    request: ResearchRequest,
    plan: ResearchPlan,
    budget: ResearchBudget,
) -> tuple[AcquisitionBatch, AgentStep]:
    """Run the crawler once and freeze deep-copied acquisition snapshots."""

    sources, logs, step = crawl_sources(request, plan)
    batch = create_acquisition_batch(sources[:budget.max_sources], logs)
    budget.set_sources_collected(len(batch.sources))
    return batch, step


def create_acquisition_batch(sources: Iterable[Source], logs: Iterable[RunLog]) -> AcquisitionBatch:
    """Create an immutable envelope containing defensive source and log snapshots."""

    source_snapshots: list[Source] = []
    artifacts: list[Artifact] = []
    captured_at = datetime.now().isoformat(timespec="seconds")
    for index, source in enumerate(sources, start=1):
        snapshot = source.model_copy(deep=True)
        snapshot.id = index
        source_snapshots.append(snapshot)
        payload = snapshot.model_dump(mode="json")
        payload.update({
            "fullTranscript": snapshot.fullTranscript,
            "transcriptText": snapshot.transcriptText,
            "transcriptSegments": [segment.model_dump(mode="json") for segment in snapshot.transcriptSegments],
        })
        content_hash = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        artifacts.append(Artifact(
            id=f"raw-{index}-{content_hash[:12]}",
            source_id=index,
            artifact_type="source_snapshot",
            content_hash=content_hash,
            captured_at=snapshot.collectedAt or captured_at,
            collector_version="voxlens-agent-v1",
            collector_name=snapshot.provider or "legacy-crawler",
            storage_uri=snapshot.url,
            mime_type="application/json",
            raw_data=payload,
            full_transcript=snapshot.fullTranscript or snapshot.transcriptText,
            transcript_segments=tuple(snapshot.transcriptSegments),
        ))
    return AcquisitionBatch(
        sources=tuple(source_snapshots),
        raw_artifacts=tuple(artifacts),
        collection_logs=tuple(log.model_copy(deep=True) for log in logs),
    )


def merge_acquisition_batches(
    first: AcquisitionBatch,
    second: AcquisitionBatch,
    max_sources: int,
) -> AcquisitionBatch:
    """Merge retry results without mutating either acquisition artifact."""

    merged: list[Source] = []
    seen: set[tuple[str, str, str]] = set()
    for source in (*first.sources, *second.sources):
        key = dedupe_sources_key(str(source.platform), source.url, source.title)
        if key in seen:
            continue
        seen.add(key)
        merged.append(source.model_copy(deep=True))
        if len(merged) >= max_sources:
            break
    return create_acquisition_batch(merged, (*first.collection_logs, *second.collection_logs))


def prepare_evidence_batch(batch: AcquisitionBatch) -> tuple[EvidenceBatch, AgentStep]:
    """Prepare immutable evidence units from copied acquisition sources.

    When transcript segments are available, creates segment-level evidence units
    with timestamp ranges (start_ms/end_ms) for precise citation anchoring.
    Also creates comment-level evidence units with comment_id for comment citations.
    Falls back to a single coarse unit per source when no segments are available.
    """

    mutable_sources = [source.model_copy(deep=True) for source in batch.sources]
    ranked_sources, step = prepare_evidence(mutable_sources)
    artifacts_by_source = {artifact.source_id: artifact for artifact in batch.raw_artifacts}
    evidence_units: list[EvidenceUnit] = []
    assessments = []
    source_scores: dict[int, EvidenceQualityLabel] = {}

    for source in ranked_sources:
        artifact = artifacts_by_source[source.id]
        segments = source.transcriptSegments or list(artifact.transcript_segments)
        created_segment_units = False

        # Create segment-level evidence units with timestamps
        for seg_idx, segment in enumerate(segments[:40]):
            seg_text = (segment.text or "").strip()
            if not seg_text or len(seg_text) < 8:
                continue
            start_ms = _parse_timestamp_to_ms(segment.start)
            end_ms = _parse_timestamp_to_ms(segment.end)
            if start_ms is not None and end_ms is not None and end_ms < start_ms:
                end_ms = start_ms
            unit = EvidenceUnit(
                id=f"evidence-{source.id}-seg-{seg_idx}",
                text=seg_text[:1600],
                modality="speech",
                source_id=source.id,
                artifact_id=artifact.id,
                normalized_text=seg_text.strip().lower()[:1600],
                start_ms=start_ms,
                end_ms=end_ms,
                extraction_method="transcript-segment",
                content_hash=artifact.content_hash,
                language="",
            )
            evidence_units.append(unit)
            created_segment_units = True

        # Create comment-level evidence units
        for cmt_idx, comment in enumerate(source.comments[:8]):
            cmt_text = (comment.text or "").strip()
            if not cmt_text:
                continue
            cmt_id = getattr(comment, "id", "") or f"c{cmt_idx}"
            unit = EvidenceUnit(
                id=f"evidence-{source.id}-cmt-{cmt_idx}",
                text=cmt_text[:800],
                modality="comment",
                source_id=source.id,
                artifact_id=artifact.id,
                normalized_text=cmt_text.strip().lower()[:800],
                comment_id=cmt_id,
                author=comment.author,
                extraction_method="comment",
                content_hash=artifact.content_hash,
            )
            evidence_units.append(unit)

        # Fallback: single coarse unit if no segments or comments
        if not created_segment_units and not any(u.source_id == source.id for u in evidence_units if u.modality == "comment"):
            transcript = source.fullTranscript or source.transcriptText or source.transcriptPreview
            text = text_excerpt(
                [transcript, source.summary, source.title, *[comment.text for comment in source.comments[:3]]],
                1600,
            )
            unit = EvidenceUnit(
                id=f"evidence-{source.id}",
                text=text or source.title,
                modality="speech" if transcript else "text",
                source_id=source.id,
                artifact_id=artifact.id,
                normalized_text=(text or source.title).strip().lower(),
                extraction_method="legacy-source-adapter",
                content_hash=artifact.content_hash,
            )
            evidence_units.append(unit)

        # Create coarse assessment at the source level
        assessment = assess_evidence_quality(source, artifact=artifact, evidence_unit=evidence_units[-1] if evidence_units else None)
        assessments.append(assessment)
        source_scores[source.id] = assessment.label

    return EvidenceBatch(
        evidence_units=tuple(evidence_units),
        quality_assessments=tuple(assessments),
        source_scores=source_scores,
    ), step


def _parse_timestamp_to_ms(value: str | int | float | None) -> int | None:
    """Parse a timestamp value (str, int, or float) into milliseconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # If it looks like seconds (small number), convert to ms
        return int(value * 1000) if value < 100000 else int(value)
    text = str(value).strip()
    if not text:
        return None
    # Try parsing as a number first
    try:
        num = float(text)
        return int(num * 1000) if num < 100000 else int(num)
    except ValueError:
        pass
    # Parse MM:SS or HH:MM:SS format
    parts = text.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return (nums[0] * 60 + nums[1]) * 1000
    if len(nums) == 3:
        return (nums[0] * 3600 + nums[1] * 60 + nums[2]) * 1000
    return None


def build_claim_set_from_report(
    report: ResearchReport,
    evidence_batch: EvidenceBatch,
    coverage: CoverageReport,
    run_id: str,
) -> ClaimSet:
    """Adapt final report takeaways and insights into claim-level grounding inputs."""

    evidence_by_source = {unit.source_id: unit for unit in evidence_batch.evidence_units}
    candidates = [
        ("takeaway", item.text, item.citations)
        for item in report.takeaways
        if item.text.strip()
    ]
    candidates.extend(
        (f"insight:{item.kind}", item.summary, item.sourceIds)
        for item in report.insights
        if item.summary.strip()
    )

    claims: list[Claim] = []
    links: list[ClaimEvidence] = []
    seen: set[str] = set()
    for claim_type, text, source_ids in candidates:
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        linked_units = [evidence_by_source[source_id] for source_id in source_ids if source_id in evidence_by_source]
        claim_id = f"claim-{len(claims) + 1}"
        claims.append(Claim(
            id=claim_id,
            claim_text=normalized,
            claim_type=claim_type,
            confidence_label=_REPORT_CONFIDENCE.get(report.confidence, "Insufficient"),
            source_ids=[unit.source_id for unit in linked_units],
            evidence_unit_ids=[unit.id for unit in linked_units],
            research_run_id=run_id,
        ))
        links.extend(ClaimEvidence(
            claim_id=claim_id,
            evidence_unit_id=unit.id,
            relation="insufficient",
            relevance_score=1.0,
            rationale="The report cites this source; the grounding verifier must still assess entailment.",
        ) for unit in linked_units)
        if len(claims) >= 12:
            break
    return ClaimSet(tuple(claims), tuple(links), coverage)


async def _build_report_with_orchestrator(
    *,
    request: ResearchRequest,
    sources: list[Source],
    run_logs: list[RunLog],
    run_id: str,
    orchestrator: LLMOrchestrator,
) -> ResearchReport:
    """Route optional report synthesis through the shared LLM budget and fallback policy."""

    query = (request.query or request.need).strip()
    deterministic = build_report(
        need=request.need,
        query=query,
        lang=request.lang,
        sources=sources,
        run_logs=run_logs,
        generated_at=datetime.now(),
        run_id=run_id,
        research_mode=request.researchMode,
        enable_llm_synthesis=False,
    )
    result = await orchestrator.execute(
        LLMCallSpec(name="report_synthesis", max_tokens=5000),
        {
            "need": request.need,
            "query": query,
            "coverage": deterministic.coverage,
            "report_kind": infer_report_kind(f"{request.need} {query}", request.researchMode),
        },
        lambda: None,
    )
    if not isinstance(result, LlmSynthesisResult):
        return deterministic
    return build_report(
        need=request.need,
        query=query,
        lang=request.lang,
        sources=sources,
        run_logs=run_logs,
        generated_at=datetime.fromisoformat(deterministic.generatedAt),
        run_id=run_id,
        research_mode=request.researchMode,
        enable_llm_synthesis=False,
        llm_synthesis_result=result,
    )


def _report_llm_provider(request: ResearchRequest, sources: list[Source]):
    """Build the provider adapter used exclusively by the LLM orchestrator.

    Uses the async transport so independent LLM calls run concurrently.
    """

    async def provider(_: LLMCallSpec, inputs: dict[str, Any]) -> LlmSynthesisResult | None:
        return await synthesize_with_llm_async(
            need=request.need,
            query=str(inputs["query"]),
            lang=request.lang,
            sources=sources,
            coverage=inputs["coverage"],
            report_kind=str(inputs["report_kind"]),
        )

    return provider


def _llm_available() -> bool:
    """Return whether report LLM synthesis is configured for this process."""

    enabled = os.getenv("VOXLENS_ENABLE_LLM", "true").strip().lower() not in {"0", "false", "no", "off"}
    return enabled and bool(os.getenv("OPENROUTER_API_KEY"))


def _supplemental_plan(
    plan: ResearchPlan,
    coverage: CoverageReport,
    budget: ResearchBudget,
) -> ResearchPlan:
    """Build one bounded retry plan for uncovered platforms or entity gaps."""

    missing_platforms = set(coverage.missing_platforms)
    targets = [
        target.model_copy(deep=True)
        for target in plan.targets
        if target.role != "fallback"
        and (not missing_platforms or str(target.platform) in missing_platforms)
    ]
    remaining_slots = max(0, budget.max_sources - budget.sources_collected)
    for target in targets:
        target.limit = min(target.limit, remaining_slots)
    return plan.model_copy(update={
        "targets": [target for target in targets if target.limit > 0],
        "notes": [*plan.notes, "Supplemental acquisition generated by CoverageGate."],
    })


def _sources_for_report(batch: AcquisitionBatch, evidence_batch: EvidenceBatch) -> list[Source]:
    """Create mutable report-only copies decorated from immutable evidence outputs."""

    assessment_by_source: dict[int, Any] = {}
    ranked_source_ids = list(evidence_batch.source_scores.keys())
    for idx, assessment in enumerate(evidence_batch.quality_assessments):
        if idx < len(ranked_source_ids):
            assessment_by_source[ranked_source_ids[idx]] = assessment
    sources = [source.model_copy(deep=True) for source in batch.sources]
    for source in sources:
        assessment = assessment_by_source.get(source.id)
        if assessment is None:
            continue
        source.quality = assessment.model_dump()
        source.metrics["evidence_quality_label"] = assessment.label
        source.metrics["evidence_quality_rank"] = _QUALITY_RANK[assessment.label]
        source.status = "weak" if assessment.label in {"Weak", "Insufficient"} else "ranked"
    return sorted(
        sources,
        key=lambda source: _QUALITY_RANK.get(evidence_batch.source_scores.get(source.id, "Insufficient"), 0),
        reverse=True,
    )


def _coverage_step(coverage: CoverageReport, gaps: list[str], blocked: bool) -> AgentStep:
    """Build the trace entry for the coverage and counterevidence gate."""

    return AgentStep(
        name="CoverageGateAgent",
        role="Checks platform, entity, evidence-type, and counterevidence coverage before LLM synthesis.",
        status="partial" if blocked else "ok",
        message=(
            "LLM synthesis blocked by severe coverage gaps; deterministic reporting remains available."
            if blocked
            else "Coverage gate allows report synthesis."
        ),
        metrics={
            "platformsCovered": len(coverage.platforms_covered),
            "entitiesCovered": len(coverage.entities_covered),
            "counterevidenceFound": coverage.counterevidence_found,
            "gaps": gaps,
        },
    )


def _verification_step(verifications: list[ClaimVerification]) -> AgentStep:
    """Build the trace entry for claim-level grounding verification."""

    relation_counts = {
        relation: sum(result.relation == relation for result in verifications)
        for relation in ("support", "contradict", "insufficient")
    }
    return AgentStep(
        name="ClaimGroundingVerifierAgent",
        role="Verifies final report claims against linked evidence units before delivery.",
        status="partial" if relation_counts["insufficient"] or relation_counts["contradict"] else "ok",
        message=f"Verified {len(verifications)} final report claims with relation-level grounding results.",
        metrics=relation_counts,
    )


def _append_architecture_warnings(
    report: ResearchReport,
    coverage: CoverageReport,
    blocked: bool,
    verifications: list[ClaimVerification],
) -> None:
    """Expose architecture gate failures through existing report warning fields."""

    if blocked:
        report.warnings.append(
            "Coverage gate blocked LLM synthesis after the bounded acquisition retry; conclusions use deterministic evidence-limited reporting."
        )
    if not coverage.counterevidence_found:
        report.warnings.append("No explicit counterevidence source was found before synthesis.")
    insufficient = sum(result.relation == "insufficient" for result in verifications)
    contradictory = sum(result.relation == "contradict" for result in verifications)
    if insufficient or contradictory:
        report.warnings.append(
            f"Claim grounding review: {insufficient} insufficient and {contradictory} contradictory claim results."
        )


def _run_async(coroutine: Coroutine[Any, Any, _ResultT]) -> _ResultT:
    """Run one architecture coroutine from the synchronous API coordinator."""

    return asyncio.run(coroutine)


def _asr_top_n(request: ResearchRequest) -> int:
    """Bound deferred ASR work independently from broad metadata search limits."""

    return min(5, max(1, request.detailVideosPerPlatform * len(request.platforms)))
