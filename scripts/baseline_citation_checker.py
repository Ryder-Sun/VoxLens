"""Citation checker for VoxLens baseline sampling.

Resolves citation IDs from report sections against source cards.
Uses a fixed random seed for reproducibility.

Citation field paths verified against actual ResearchReport model:
- report.sections[].bullets[].citations  (list[int] source IDs)
- report.sections[].sourceIds            (list[int])
- report.sections[].quote.sourceId       (int)
- report.sections[].table[].evidence     (list[int])
- report.sections[].table[].dimensions[].evidence  (list[int])
- report.takeaways[].citations           (list[int] — CitationText type)
- report.insights[].sourceIds            (list[int])
- report.coverage.citedSources            (int — aggregate, used for count only)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models import ResearchReport

from baseline_classifier import CitationCheckResult


# Fixed seed for reproducible citation sampling
CITATION_SEED = 20260803

# Minimum citations to sample per baseline
MIN_CITATION_SAMPLE = 10
# Percentage of total citations to sample
CITATION_SAMPLE_RATIO = 0.20


def check_citations_for_run(
    run_id: str,
    report: ResearchReport,
    rng: random.Random,
) -> list[CitationCheckResult]:
    """Check a random sample of citations against source cards for one run.

    Args:
        run_id: The run ID.
        report: The ResearchReport to check.
        rng: Pre-seeded Random instance for reproducible sampling.

    Returns:
        List of CitationCheckResult for each sampled citation.
    """
    # Build source ID lookup from report.sources
    source_lookup: dict[int, Any] = {
        s.id: s for s in report.sources
    }

    # Collect all citation references from report using actual model field paths
    all_citations = _collect_citations(report)

    if not all_citations:
        return []

    # Determine sample size
    sample_size = max(
        MIN_CITATION_SAMPLE,
        int(len(all_citations) * CITATION_SAMPLE_RATIO),
    )
    sample_size = min(sample_size, len(all_citations))

    sampled = rng.sample(all_citations, sample_size)

    results: list[CitationCheckResult] = []
    for citation in sampled:
        source_id = citation.get("sourceId")
        citation_id = citation.get("id", "")
        quote_text = citation.get("text", "").strip()

        resolved = source_id is not None and source_id in source_lookup
        text_match = False

        if resolved:
            source = source_lookup[source_id]
            # Check if the quote text appears in the source's transcript, title, or summary
            source_text = " ".join([
                source.title,
                source.summary,
                source.transcriptPreview,
                source.fullTranscript,
            ]).strip()
            if quote_text and source_text:
                # Loose match: check if significant words from quote appear in source text
                quote_words = set(quote_text.lower().split())
                source_words = set(source_text.lower().split())
                overlap = quote_words & source_words
                text_match = len(overlap) >= len(quote_words) * 0.3  # 30% word overlap threshold

        results.append(CitationCheckResult(
            citation_id=citation_id,
            source_id=source_id,
            resolved=resolved,
            text_match=text_match,
            run_id=run_id,
            query=report.query,
        ))

    return results


def _collect_citations(report: ResearchReport) -> list[dict[str, Any]]:
    """Extract all citation references from report using actual model field paths.

    Citation sources in ResearchReport:
    1. sections[].bullets[].citations  — CitationText.citations is list[int] (source IDs)
    2. sections[].sourceIds            — list[int] source IDs at section level
    3. sections[].quote.sourceId        — VideoQuote.sourceId is int
    4. sections[].table[].evidence      — ComparisonRow.evidence is list[int]
    5. sections[].table[].dimensions[].evidence — ComparisonDimension.evidence is list[int]
    6. takeaways[].citations            — CitationText.citations is list[int]
    7. insights[].sourceIds            — list[int]
    """
    citations: list[dict[str, Any]] = []

    # 1. sections[].bullets[].citations — CitationText has .text and .citations (list[int])
    for sec_idx, section in enumerate(report.sections):
        for bullet_idx, bullet in enumerate(section.bullets):
            for source_id in bullet.citations:
                citations.append({
                    "id": f"sec-{sec_idx}-bullet-{bullet_idx}-src-{source_id}",
                    "sourceId": source_id,
                    "text": bullet.text,
                })

    # 2. sections[].sourceIds
    for sec_idx, section in enumerate(report.sections):
        for source_id in section.sourceIds:
            citations.append({
                "id": f"sec-{sec_idx}-src-{source_id}",
                "sourceId": source_id,
                "text": section.title,
            })

    # 3. sections[].quote.sourceId
    for sec_idx, section in enumerate(report.sections):
        if section.quote is not None:
            citations.append({
                "id": f"sec-{sec_idx}-quote-src-{section.quote.sourceId}",
                "sourceId": section.quote.sourceId,
                "text": section.quote.quote,
            })

    # 4. sections[].table[].evidence
    for sec_idx, section in enumerate(report.sections):
        for row_idx, row in enumerate(section.table):
            for source_id in row.evidence:
                citations.append({
                    "id": f"sec-{sec_idx}-table-row-{row_idx}-src-{source_id}",
                    "sourceId": source_id,
                    "text": row.name,
                })

    # 5. sections[].table[].dimensions[].evidence
    for sec_idx, section in enumerate(report.sections):
        for row_idx, row in enumerate(section.table):
            for dim_idx, dim in enumerate(row.dimensions):
                for source_id in dim.evidence:
                    citations.append({
                        "id": f"sec-{sec_idx}-table-row-{row_idx}-dim-{dim_idx}-src-{source_id}",
                        "sourceId": source_id,
                        "text": dim.summary,
                    })

    # 6. takeaways[].citations — CitationText type, same as bullets
    for tk_idx, takeaway in enumerate(report.takeaways):
        for source_id in takeaway.citations:
            citations.append({
                "id": f"takeaway-{tk_idx}-src-{source_id}",
                "sourceId": source_id,
                "text": takeaway.text,
            })

    # 7. insights[].sourceIds
    for ins_idx, insight in enumerate(report.insights):
        for source_id in insight.sourceIds:
            citations.append({
                "id": f"insight-{ins_idx}-src-{source_id}",
                "sourceId": source_id,
                "text": insight.summary,
            })

    return citations


def save_citation_sample(
    results: list[CitationCheckResult],
    output_path: Path,
) -> None:
    """Save the citation sample manifest for reproducibility.

    Args:
        results: All citation check results across all runs.
        output_path: Path to write the manifest.
    """
    data = [
        {
            "run_id": r.run_id,
            "citation_id": r.citation_id,
            "source_id": r.source_id,
            "resolved": r.resolved,
            "text_match": r.text_match,
            "query": r.query,
        }
        for r in results
    ]
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
