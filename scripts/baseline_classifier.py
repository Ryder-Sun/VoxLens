"""Breakpoint classifier for VoxLens baseline sampling.

Reads run artifacts (record.json, events.jsonl, report.json) and classifies
each run's failures or degradations into one of five categories:

1. structural_gap      — platform catalog has no transcript capability
2. adapter_not_wired  — ASRProviderUnavailable raised explicitly
3. runtime_volatility  — crawler errors, timeouts, auth challenges, rate limits
4. synthesis_issue     — citation IDs fail to resolve to source cards
5. needs_manual_review — none of the above evidence matched

Platform-level attribution uses RunLog.platform + RunLog.ok and
Source.platform + Source.transcriptPreview, never run-level aggregates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.models import ResearchReport, RunRecord


BreakpointCategory = Literal[
    "structural_gap",
    "adapter_not_wired",
    "runtime_volatility",
    "synthesis_issue",
    "needs_manual_review",
]

CONFIDENCE_HIGH = "high"
CONFIDENCE_LOW = "low"

# Platforms that have no transcript capability in platform_catalog.py
_PLATFORMS_WITHOUT_TRANSCRIPT = frozenset({"douyin", "xiaohongshu", "kuaishou"})


@dataclass(frozen=True)
class BreakpointFinding:
    """A single classified finding from one run."""

    category: BreakpointCategory
    confidence: str  # "high" or "low"
    evidence: str
    platform: str = ""  # empty string = run-level, not platform-specific


@dataclass(frozen=True)
class RunBreakpoints:
    """All breakpoints found in a single run."""

    run_id: str
    findings: tuple[BreakpointFinding, ...] = ()

    def findings_for(self, category: BreakpointCategory) -> tuple[BreakpointFinding, ...]:
        return tuple(f for f in self.findings if f.category == category)


@dataclass(frozen=True)
class CitationCheckResult:
    """Result of checking one citation ID against source cards."""

    citation_id: str
    source_id: int | None
    resolved: bool
    text_match: bool = False
    run_id: str = ""
    query: str = ""


def classify_run(
    run_dir: Path,
    citation_checks: tuple[CitationCheckResult, ...] = (),
) -> RunBreakpoints:
    """Classify breakpoints for a single run by reading its artifacts.

    Args:
        run_dir: Path to the run directory containing record.json, events.jsonl, report.json.
        citation_checks: Pre-computed citation resolution results for this run.

    Returns:
        RunBreakpoints with all findings.
    """
    findings: list[BreakpointFinding] = []

    record = _load_record(run_dir)
    events = _load_events(run_dir)
    report = _load_report(run_dir)

    if record is None:
        return RunBreakpoints(
            run_id=run_dir.name,
            findings=(BreakpointFinding(
                category="needs_manual_review",
                confidence=CONFIDENCE_LOW,
                evidence="record.json not found or unparseable",
            ),),
        )

    run_id = record.runId

    # --- Category 2: adapter_not_wired — explicit ASRProviderUnavailable ---
    asr_findings = _detect_asr_adapter_not_wired(run_id, record, events, report)
    findings.extend(asr_findings)

    # --- Category 1: structural_gap — platform has no transcript capability ---
    if report is not None:
        structural_findings = _detect_structural_gaps(run_id, report)
        findings.extend(structural_findings)

    # --- Category 3: runtime_volatility — crawler/provider errors ---
    runtime_findings = _detect_runtime_volatility(run_id, record, events, report)
    findings.extend(runtime_findings)

    # --- Category 4: synthesis_issue — citation IDs don't resolve ---
    synthesis_findings = _detect_synthesis_issues(run_id, citation_checks)
    findings.extend(synthesis_findings)

    # --- Category 5: needs_manual_review — unknown record.error not classified above ---
    if (
        record.error
        and record.status == "failed"
        and "ASRProviderUnavailable" not in record.error
        and not _is_explicit_runtime_error(record.error)
    ):
        findings.append(BreakpointFinding(
            category="needs_manual_review",
            confidence=CONFIDENCE_LOW,
            evidence=f"Unclassified error in record.error: {record.error}",
        ))

    # --- Category 5: needs_manual_review — if no findings but run failed ---
    if not findings and record.status == "failed":
        findings.append(BreakpointFinding(
            category="needs_manual_review",
            confidence=CONFIDENCE_LOW,
            evidence=f"Run failed but no specific breakpoint evidence found. record.error={record.error}",
        ))

    # --- Category 5: needs_manual_review — if no findings at all ---
    if not findings:
        findings.append(BreakpointFinding(
            category="needs_manual_review",
            confidence=CONFIDENCE_LOW,
            evidence="No breakpoint evidence found; run may have succeeded or findings are ambiguous.",
        ))

    return RunBreakpoints(run_id=run_id, findings=tuple(findings))


def _normalize_asr_key(step_name: str, message: str) -> str:
    """Extract a normalized dedup key from an ASR exception message.

    The same ASRProviderUnavailable occurrence can appear in record.error,
    events.jsonl (as agent_step event), and report.agentTrace (as AgentStep).
    The step name differs ("record.error" vs "ASRPipeline"), so we dedup
    by the normalized exception text alone, not the source label.
    """
    # Extract the core exception message after "ASRProviderUnavailable"
    normalized_msg = message.strip().lower()
    if "asrproviderunavailable" in normalized_msg:
        idx = normalized_msg.index("asrproviderunavailable")
        normalized_msg = normalized_msg[idx:]
    return normalized_msg


def _detect_asr_adapter_not_wired(
    run_id: str,
    record: RunRecord,
    events: list[dict[str, Any]],
    report: ResearchReport | None,
) -> list[BreakpointFinding]:
    """Detect ASRProviderUnavailable in events, agentTrace, or record.error.

    Deduplicates by normalized exception text so the same occurrence across
    record.error, events.jsonl, and report.agentTrace produces exactly one finding.
    """
    findings: list[BreakpointFinding] = []
    seen_keys: set[str] = set()

    # Check record.error first — ASRProviderUnavailable here is high confidence
    if record.error and "ASRProviderUnavailable" in record.error:
        dedup_key = _normalize_asr_key("record.error", record.error)
        if dedup_key not in seen_keys:
            findings.append(BreakpointFinding(
                category="adapter_not_wired",
                confidence=CONFIDENCE_HIGH,
                evidence=f"record.error: {record.error}",
            ))
            seen_keys.add(dedup_key)

    # Scan events.jsonl
    for event in events:
        event_str = json.dumps(event, ensure_ascii=False)
        if "ASRProviderUnavailable" not in event_str:
            continue

        if event.get("type") == "agent_step":
            step = event.get("step", event)
            name = step.get("name", "")
            message = step.get("message", "")
        else:
            name = event.get("type", "")
            message = event.get("message", "")

        dedup_key = _normalize_asr_key(name, message)
        if dedup_key in seen_keys:
            continue

        findings.append(BreakpointFinding(
            category="adapter_not_wired",
            confidence=CONFIDENCE_HIGH,
            evidence=f"agent_step '{name}': {message}",
        ))
        seen_keys.add(dedup_key)

    # Scan report.agentTrace — skip entries already seen in events
    if report is not None:
        for step in report.agentTrace:
            msg = step.message or ""
            metrics_str = str(step.metrics)
            if "ASRProviderUnavailable" not in msg and "ASRProviderUnavailable" not in metrics_str:
                continue

            # Use the message that contains the exception
            source_msg = msg if "ASRProviderUnavailable" in msg else metrics_str
            dedup_key = _normalize_asr_key(step.name, source_msg)
            if dedup_key in seen_keys:
                continue

            findings.append(BreakpointFinding(
                category="adapter_not_wired",
                confidence=CONFIDENCE_HIGH,
                evidence=f"agentTrace step '{step.name}' (status={step.status}): {source_msg}",
            ))
            seen_keys.add(dedup_key)

    return findings


def _detect_structural_gaps(
    run_id: str,
    report: ResearchReport,
) -> list[BreakpointFinding]:
    """Detect platforms with no transcript capability using per-source data."""
    findings: list[BreakpointFinding] = []

    # Group sources by platform
    platform_sources: dict[str, list[Any]] = {}
    for source in report.sources:
        platform = source.platform
        platform_sources.setdefault(platform, []).append(source)

    # For platforms without transcript capability, check if any source has transcript content
    for platform, sources in platform_sources.items():
        if platform not in _PLATFORMS_WITHOUT_TRANSCRIPT:
            continue

        has_transcript = any(
            bool(s.transcriptPreview) or bool(s.fullTranscript) or bool(s.transcriptText)
            for s in sources
        )
        if not has_transcript:
            findings.append(BreakpointFinding(
                category="structural_gap",
                confidence=CONFIDENCE_HIGH,
                evidence=f"Platform '{platform}' has {len(sources)} source(s) but none have transcript content; platform catalog declares no transcript capability",
                platform=platform,
            ))

    return findings


# Keywords that indicate explicit runtime volatility (not unknown errors)
_RUNTIME_ERROR_KEYWORDS = frozenset({
    "timeout", "timed out", "connection", "refused", "reset", "unreachable",
    "rate limit", "rate-limit", "429", "too many requests",
    "captcha", "verification", "login", "auth", "cookie", "forbidden",
    "403", "401", "blocked", "banned", "restricted",
    "network", "dns", "socket", "ssl", "certificate",
    "crawler", "provider", "playwright", "browser",
})


def _is_explicit_runtime_error(error_text: str) -> bool:
    """Check if an error message contains explicit runtime volatility evidence.

    Only errors with recognizable runtime keywords (timeout, connection refused,
    rate limit, auth challenge, etc.) are classified as runtime_volatility.
    Unknown errors without these keywords are left for needs_manual_review.
    """
    lower = error_text.lower()
    return any(kw in lower for kw in _RUNTIME_ERROR_KEYWORDS)


def _detect_runtime_volatility(
    run_id: str,
    record: RunRecord,
    events: list[dict[str, Any]],
    report: ResearchReport | None,
) -> list[BreakpointFinding]:
    """Detect crawler/provider errors from RunLogs, events, and record.error."""
    findings: list[BreakpointFinding] = []

    # Check RunLogs for failed provider runs (per-platform)
    if report is not None:
        for log in report.runLogs:
            if not log.ok:
                findings.append(BreakpointFinding(
                    category="runtime_volatility",
                    confidence=CONFIDENCE_HIGH,
                    evidence=f"RunLog platform='{log.platform}' provider='{log.provider}' ok=False count={log.count} note={log.note}",
                    platform=log.platform,
                ))

    # Check record.error for explicit runtime errors only
    # Unknown errors without runtime keywords go to needs_manual_review, not here
    if record.error and record.status == "failed":
        # Skip ASR errors (already caught by adapter detection)
        if "ASRProviderUnavailable" not in record.error:
            if _is_explicit_runtime_error(record.error):
                findings.append(BreakpointFinding(
                    category="runtime_volatility",
                    confidence=CONFIDENCE_HIGH,
                    evidence=f"record.error: {record.error}",
                ))
            # else: unknown error — will be caught by needs_manual_review fallback

    # Check events for error-type events with explicit runtime evidence
    for event in events:
        if event.get("type") == "error":
            message = event.get("message", "")
            if "ASRProviderUnavailable" not in message:
                if _is_explicit_runtime_error(message):
                    findings.append(BreakpointFinding(
                        category="runtime_volatility",
                        confidence=CONFIDENCE_HIGH,
                        evidence=f"error event: {message}",
                    ))
                # else: unknown error event — left for needs_manual_review
        # Check stage events for error status — always runtime (explicit failure)
        if event.get("type") == "stage":
            status = event.get("status", "")
            if status == "error":
                message = event.get("message", "")
                platform = event.get("platform", "")
                findings.append(BreakpointFinding(
                    category="runtime_volatility",
                    confidence=CONFIDENCE_HIGH,
                    evidence=f"stage event status=error platform={platform}: {message}",
                    platform=platform,
                ))
            elif status == "partial":
                # partial can be coverage shortfall or synthesis downgrade, not just runtime
                # only classify as runtime_volatility if message has runtime keywords
                message = event.get("message", "")
                platform = event.get("platform", "")
                if _is_explicit_runtime_error(message):
                    findings.append(BreakpointFinding(
                        category="runtime_volatility",
                        confidence=CONFIDENCE_HIGH,
                        evidence=f"stage event status=partial platform={platform}: {message}",
                        platform=platform,
                    ))
                # else: partial without runtime evidence — not classified here

    # Check report.warnings for runtime-related warnings
    if report is not None:
        for warning in report.warnings:
            # Skip ASR-related warnings (caught by adapter detection)
            if "ASRProviderUnavailable" in warning:
                continue
            # Skip structural warnings about coverage (caught by structural detection)
            if "transcript" in warning.lower() and "capability" in warning.lower():
                continue
            findings.append(BreakpointFinding(
                category="runtime_volatility",
                confidence=CONFIDENCE_LOW,
                evidence=f"report.warnings: {warning}",
            ))

    return findings


def _detect_synthesis_issues(
    run_id: str,
    citation_checks: tuple[CitationCheckResult, ...],
) -> list[BreakpointFinding]:
    """Detect citation resolution failures from pre-computed checks."""
    findings: list[BreakpointFinding] = []

    for check in citation_checks:
        if check.run_id != run_id:
            continue
        if not check.resolved:
            findings.append(BreakpointFinding(
                category="synthesis_issue",
                confidence=CONFIDENCE_HIGH,
                evidence=f"Citation ID '{check.citation_id}' could not resolve to source {check.source_id}",
            ))
        elif not check.text_match:
            findings.append(BreakpointFinding(
                category="synthesis_issue",
                confidence=CONFIDENCE_LOW,
                evidence=f"Citation ID '{check.citation_id}' resolved but text mismatch",
            ))

    return findings


def _load_record(run_dir: Path) -> RunRecord | None:
    """Load record.json from a run directory."""
    path = run_dir / "record.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RunRecord.model_validate(data)
    except Exception:
        return None


def _load_events(run_dir: Path) -> list[dict[str, Any]]:
    """Load all events from events.jsonl."""
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _load_report(run_dir: Path) -> ResearchReport | None:
    """Load report.json from a run directory."""
    path = run_dir / "report.json"
    if not path.exists():
        return None
    try:
        return ResearchReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None
