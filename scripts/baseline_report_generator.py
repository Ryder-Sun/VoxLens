"""Report generator for VoxLens baseline sampling.

Produces the dual-view baseline report:
1. Alpha acceptance baseline — supported-platform queries only
2. Capability gap matrix — all default platforms, structural vs runtime vs synthesis

Output: reports/baseline-2026-08-03.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from baseline_classifier import RunBreakpoints, BreakpointFinding


# Platforms with transcript/text capability (supported for alpha metrics)
SUPPORTED_PLATFORMS = frozenset({"bilibili", "youtube", "zhihu", "weibo"})

# Platforms without transcript capability (capability gap)
GAP_PLATFORMS = frozenset({"douyin", "xiaohongshu", "kuaishou"})


def generate_baseline_report(
    run_results: list[dict[str, Any]],
    citation_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate the full baseline report markdown.

    Args:
        run_results: List of dicts per run with keys:
            - run_id, query, query_index, platforms, status, run_breakpoints,
              first_source_seconds, evidence_coverage, total_sources, cited_sources
        citation_results: List of dicts per citation check with keys:
            - run_id, citation_id, source_id, resolved, text_match, query
        output_path: Path to write the markdown report.
    """
    lines: list[str] = []

    lines.append("# VoxLens Alpha Baseline Report")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Total Runs**: {len(run_results)}")
    lines.append(f"**Citation Sample Size**: {len(citation_results)}")
    lines.append(f"**Random Seed**: 20260803")
    lines.append("")

    # === Section 1: Summary ===
    lines.extend(_summary_section(run_results))
    lines.append("")

    # === Section 2: Alpha Acceptance Baseline ===
    lines.extend(_alpha_acceptance_section(run_results))
    lines.append("")

    # === Section 3: Capability Gap Matrix ===
    lines.extend(_capability_gap_section(run_results))
    lines.append("")

    # === Section 4: Per-Run Detail ===
    lines.extend(_per_run_detail_section(run_results))
    lines.append("")

    # === Section 5: Citation Integrity ===
    lines.extend(_citation_integrity_section(citation_results))
    lines.append("")

    # === Section 6: Breakpoint Distribution ===
    lines.extend(_breakpoint_distribution_section(run_results))
    lines.append("")

    # === Section 7: Priority Recommendations ===
    lines.extend(_priority_recommendations_section(run_results))
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _summary_section(run_results: list[dict[str, Any]]) -> list[str]:
    """Generate the executive summary section."""
    total = len(run_results)
    completed = sum(1 for r in run_results if r["status"] in ("completed", "partial"))
    failed = sum(1 for r in run_results if r["status"] == "failed")
    success_rate = (completed / total * 100) if total > 0 else 0

    # Time to first source
    ttf_values = [r["first_source_seconds"] for r in run_results if r["first_source_seconds"] is not None]
    avg_ttf = sum(ttf_values) / len(ttf_values) if ttf_values else 0
    max_ttf = max(ttf_values) if ttf_values else 0

    lines = [
        "## 1. Executive Summary",
        "",
        f"| Metric | Value | Alpha Target | Status |",
        f"|---|---|---|---|",
        f"| Run reliability | {completed}/{total} ({success_rate:.0f}%) | >= 90% | {'PASS' if success_rate >= 90 else 'FAIL'} |",
        f"| Failed runs | {failed} | — | — |",
        f"| Avg time to first source | {avg_ttf:.0f}s | <= 90s | {'PASS' if avg_ttf <= 90 else 'FAIL'} |",
        f"| Max time to first source | {max_ttf:.0f}s | <= 90s | {'PASS' if max_ttf <= 90 else 'FAIL'} |",
    ]
    return lines


def _alpha_acceptance_section(run_results: list[dict[str, Any]]) -> list[str]:
    """Generate alpha acceptance baseline for supported-platform queries only."""
    lines = [
        "## 2. Alpha Acceptance Baseline (Supported-Platform Queries)",
        "",
        "Only queries targeting platforms with transcript/text evidence paths are included.",
        f"Supported platforms: {', '.join(sorted(SUPPORTED_PLATFORMS))}",
        "",
    ]

    # Identify which query indices are "supported-only" (all platforms in SUPPORTED_PLATFORMS)
    supported_runs = []
    mixed_runs = []
    for r in run_results:
        platforms_set = set(r["platforms"])
        if platforms_set.issubset(SUPPORTED_PLATFORMS):
            supported_runs.append(r)
        else:
            mixed_runs.append(r)

    lines.append(f"- Supported-only runs: {len(supported_runs)}")
    lines.append(f"- Mixed (supported + gap) runs: {len(mixed_runs)}")
    lines.append("")

    # Metrics for supported-only runs
    if supported_runs:
        total = len(supported_runs)
        completed = sum(1 for r in supported_runs if r["status"] in ("completed", "partial"))
        success_rate = (completed / total * 100) if total > 0 else 0

        # Evidence coverage: reports with >= 3 citable sources
        with_3plus = sum(1 for r in supported_runs if r["cited_sources"] >= 3)
        coverage_rate = (with_3plus / total * 100) if total > 0 else 0

        lines.extend([
            "### Alpha Metrics (Supported-Platform Subset)",
            "",
            "| Metric | Value | Target | Status |",
            "|---|---|---|---|",
            f"| Run reliability | {completed}/{total} ({success_rate:.0f}%) | >= 90% | {'PASS' if success_rate >= 90 else 'FAIL'} |",
            f"| Evidence coverage (>=3 sources) | {with_3plus}/{total} ({coverage_rate:.0f}%) | >= 80% | {'PASS' if coverage_rate >= 80 else 'FAIL'} |",
            "",
        ])

        # Per-query breakdown
        lines.extend([
            "### Per-Query Breakdown",
            "",
            "| Query # | Query | Runs | Completed | Avg Sources | Avg Cited | TTF (avg) |",
            "|---|---|---|---|---|---|---|",
        ])

        query_groups: dict[int, list[dict[str, Any]]] = {}
        for r in supported_runs:
            query_groups.setdefault(r["query_index"], []).append(r)

        for qidx in sorted(query_groups.keys()):
            group = query_groups[qidx]
            query_text = group[0]["query"][:60]
            runs_count = len(group)
            completed_count = sum(1 for g in group if g["status"] in ("completed", "partial"))
            avg_sources = sum(g["total_sources"] for g in group) / runs_count
            avg_cited = sum(g["cited_sources"] for g in group) / runs_count
            ttf_vals = [g["first_source_seconds"] for g in group if g["first_source_seconds"] is not None]
            avg_ttf = sum(ttf_vals) / len(ttf_vals) if ttf_vals else 0
            lines.append(
                f"| {qidx} | {query_text} | {runs_count} | {completed_count} | "
                f"{avg_sources:.1f} | {avg_cited:.1f} | {avg_ttf:.0f}s |"
            )
    else:
        lines.append("No supported-only runs in the baseline.")

    return lines


def _capability_gap_section(run_results: list[dict[str, Any]]) -> list[str]:
    """Generate the capability gap matrix for all platforms."""
    lines = [
        "## 3. Capability Gap Matrix (All Default Platforms)",
        "",
        "This view includes all 7 default platforms, separating structural gaps from runtime issues.",
        "",
    ]

    # Collect platform-level stats
    platform_stats: dict[str, dict[str, Any]] = {}
    for r in run_results:
        for platform in r["platforms"]:
            if platform not in platform_stats:
                platform_stats[platform] = {
                    "total_runs": 0,
                    "structural_gaps": 0,
                    "runtime_failures": 0,
                    "adapter_not_wired": 0,
                    "synthesis_issues": 0,
                    "needs_review": 0,
                    "category": "supported" if platform in SUPPORTED_PLATFORMS else "gap",
                }
            ps = platform_stats[platform]
            ps["total_runs"] += 1

            # Count findings per platform
            breakpoints: RunBreakpoints = r.get("run_breakpoints", RunBreakpoints(run_id=r["run_id"]))
            for finding in breakpoints.findings:
                if finding.platform == platform:
                    if finding.category == "structural_gap":
                        ps["structural_gaps"] += 1
                    elif finding.category == "runtime_volatility":
                        ps["runtime_failures"] += 1
                    elif finding.category == "adapter_not_wired":
                        ps["adapter_not_wired"] += 1
                    elif finding.category == "synthesis_issue":
                        ps["synthesis_issues"] += 1
                    elif finding.category == "needs_manual_review":
                        ps["needs_review"] += 1

    # Also count run-level findings (platform="") for runs that include each platform
    for r in run_results:
        breakpoints: RunBreakpoints = r.get("run_breakpoints", RunBreakpoints(run_id=r["run_id"]))
        for finding in breakpoints.findings:
            if finding.platform == "" and finding.category == "needs_manual_review":
                for platform in r["platforms"]:
                    if platform in platform_stats:
                        platform_stats[platform]["needs_review"] += 1

    lines.extend([
        "| Platform | Category | Total Runs | Structural Gaps | Adapter Not Wired | Runtime Failures | Synthesis Issues | Needs Review |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for platform in sorted(platform_stats.keys()):
        ps = platform_stats[platform]
        lines.append(
            f"| {platform} | {ps['category']} | {ps['total_runs']} | "
            f"{ps['structural_gaps']} | {ps['adapter_not_wired']} | "
            f"{ps['runtime_failures']} | {ps['synthesis_issues']} | "
            f"{ps['needs_review']} |"
        )

    return lines


def _per_run_detail_section(run_results: list[dict[str, Any]]) -> list[str]:
    """Generate per-run detail table."""
    lines = [
        "## 4. Per-Run Detail",
        "",
        "| Run # | Query # | Status | Platforms | Sources | Cited | TTF (s) | Breakpoint Summary |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for i, r in enumerate(run_results, start=1):
        breakpoints: RunBreakpoints = r.get("run_breakpoints", RunBreakpoints(run_id=r["run_id"]))
        bp_summary = _summarize_breakpoints(breakpoints)
        platforms_str = "/".join(r["platforms"])
        ttf = f"{r['first_source_seconds']:.0f}" if r["first_source_seconds"] is not None else "N/A"
        lines.append(
            f"| {i} | {r['query_index']} | {r['status']} | {platforms_str} | "
            f"{r['total_sources']} | {r['cited_sources']} | {ttf} | {bp_summary} |"
        )

    return lines


def _citation_integrity_section(citation_results: list[dict[str, Any]]) -> list[str]:
    """Generate citation integrity section."""
    total = len(citation_results)
    resolved = sum(1 for c in citation_results if c["resolved"])
    text_matched = sum(1 for c in citation_results if c["text_match"])
    resolution_rate = (resolved / total * 100) if total > 0 else 0
    match_rate = (text_matched / total * 100) if total > 0 else 0

    return [
        "## 5. Citation Integrity",
        "",
        f"| Metric | Value | Target | Status |",
        f"|---|---|---|---|",
        f"| Citations sampled | {total} | >= 10 | — |",
        f"| Citation resolution rate | {resolved}/{total} ({resolution_rate:.0f}%) | 100% | {'PASS' if resolution_rate == 100 else 'FAIL'} |",
        f"| Text match rate | {text_matched}/{total} ({match_rate:.0f}%) | — | — |",
        "",
        "### Sampled Citations",
        "",
        "| Run ID | Citation ID | Source ID | Resolved | Text Match | Query |",
        "|---|---|---|---|---|---|",
    ] + [
        f"| {c['run_id']} | {c['citation_id']} | {c['source_id']} | "
        f"{'✓' if c['resolved'] else '✗'} | {'✓' if c['text_match'] else '✗'} | {c['query'][:40]} |"
        for c in citation_results
    ]


def _breakpoint_distribution_section(run_results: list[dict[str, Any]]) -> list[str]:
    """Generate breakpoint distribution pie chart data."""
    category_counts: dict[str, int] = {}
    for r in run_results:
        breakpoints: RunBreakpoints = r.get("run_breakpoints", RunBreakpoints(run_id=r["run_id"]))
        for finding in breakpoints.findings:
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

    total = sum(category_counts.values())

    lines = [
        "## 6. Breakpoint Distribution",
        "",
        f"Total findings across all runs: {total}",
        "",
        "| Category | Count | Percentage |",
        "|---|---|---|",
    ]

    labels = {
        "structural_gap": "Structural Gap (platform has no transcript capability)",
        "adapter_not_wired": "Adapter Not Wired (ASRProviderUnavailable)",
        "runtime_volatility": "Runtime Volatility (crawler/provider errors)",
        "synthesis_issue": "Synthesis Issue (citation resolution failure)",
        "needs_manual_review": "Needs Manual Review",
    }

    for cat in ["structural_gap", "adapter_not_wired", "runtime_volatility", "synthesis_issue", "needs_manual_review"]:
        count = category_counts.get(cat, 0)
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"| {labels[cat]} | {count} | {pct:.0f}% |")

    return lines


def _priority_recommendations_section(run_results: list[dict[str, Any]]) -> list[str]:
    """Generate data-driven priority recommendations."""
    category_counts: dict[str, int] = {}
    for r in run_results:
        breakpoints: RunBreakpoints = r.get("run_breakpoints", RunBreakpoints(run_id=r["run_id"]))
        for finding in breakpoints.findings:
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

    total = sum(category_counts.values())

    lines = [
        "## 7. Priority Recommendations",
        "",
        "Based on breakpoint distribution, the following priorities are recommended:",
        "",
    ]

    if total == 0:
        lines.append("No breakpoints detected. System appears stable.")
        return lines

    # Rank by count
    ranked = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    recommendations = {
        "structural_gap": (
            "ASR Adapter Integration",
            "Multiple platforms lack transcript capability. "
            "Integrating a working ASR adapter (e.g., faster-whisper) would unlock "
            "evidence extraction from douyin/xiaohongshu/kuaishou video content. "
            "Priority: HIGH — this is a structural gap, not a stability issue."
        ),
        "adapter_not_wired": (
            "ASR Provider Wiring",
            "ASRProviderUnavailable is being raised. The architecture defines the "
            "ASRProvider protocol but no execution adapter is injected. "
            "Priority: HIGH — complete the adapter implementation."
        ),
        "runtime_volatility": (
            "Crawler/Provider Stability",
            "Runtime errors from crawlers or providers are the dominant failure mode. "
            "Focus on retry/degradation strategies, cookie management, and "
            "platform-specific stop conditions. "
            "Priority: MEDIUM — incremental hardening."
        ),
        "synthesis_issue": (
            "Citation Grounding",
            "Citation IDs are failing to resolve to source cards. "
            "Investigate the report builder's citation ID assignment and "
            "the grounding verifier's source linkage. "
            "Priority: HIGH — citation integrity is an alpha exit criterion."
        ),
        "needs_manual_review": (
            "Manual Investigation Needed",
            "Some runs failed without clear breakpoint evidence. "
            "Review record.json, events.jsonl, and logs manually. "
            "Priority: LOW — batch investigate after addressing higher-priority items."
        ),
    }

    for i, (category, count) in enumerate(ranked, start=1):
        pct = (count / total * 100) if total > 0 else 0
        title, desc = recommendations.get(category, ("Unknown", "No recommendation available."))
        lines.append(f"### {i}. {title} ({count} findings, {pct:.0f}%)")
        lines.append("")
        lines.append(desc)
        lines.append("")

    return lines


def _summarize_breakpoints(breakpoints: RunBreakpoints) -> str:
    """Summarize breakpoints for a per-run table cell."""
    if not breakpoints.findings:
        return "none"

    categories: dict[str, int] = {}
    for finding in breakpoints.findings:
        categories[finding.category] = categories.get(finding.category, 0) + 1

    parts = [f"{cat}×{count}" for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)]
    return ", ".join(parts)
