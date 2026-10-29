from __future__ import annotations

import re
from typing import Any

from pydantic import TypeAdapter

from app.models import CitationRef, CitationText, ReportInsight, ReportSection
from app.services.llm_types import LlmSynthesisResult, SemanticEvidenceExtraction


def result_from_payload(payload: dict[str, Any], allowed_ids: set[int]) -> LlmSynthesisResult:
    takeaways = TypeAdapter(list[CitationText]).validate_python(
        _sanitize_citation_items(payload.get("takeaways", []), allowed_ids)
    )
    insights = TypeAdapter(list[ReportInsight]).validate_python(
        _sanitize_insights(payload.get("insights", []), allowed_ids)
    )
    sections = TypeAdapter(list[ReportSection]).validate_python(
        _sanitize_sections(payload.get("sections", []), allowed_ids)
    )
    warnings = [str(item)[:240] for item in payload.get("warnings", []) if str(item).strip()]
    return LlmSynthesisResult(takeaways=takeaways[:8], insights=insights[:8], sections=sections[:14], warnings=warnings[:6])


def _sanitize_citation_items(items: Any, allowed_ids: set[int]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return sanitized
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        sanitized.append({
            "text": text,
            "citations": valid_ids(item.get("citations", item.get("sourceIds", [])), allowed_ids),
            "citationRefs": _sanitize_citation_refs(item.get("citationRefs", []), allowed_ids),
        })
    return sanitized


def _sanitize_citation_refs(items: Any, allowed_ids: set[int]) -> list[dict[str, Any]]:
    """Sanitize structured citation references with timestamp/comment context."""
    sanitized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return sanitized
    for item in items:
        if not isinstance(item, dict):
            continue
        source_id = item.get("sourceId", item.get("source_id"))
        try:
            source_id = int(source_id) if source_id is not None else 0
        except (TypeError, ValueError):
            continue
        if source_id not in allowed_ids:
            continue
        ref = {
            "sourceId": source_id,
            "evidenceUnitId": str(item.get("evidenceUnitId", item.get("evidence_unit_id", "")))[:120],
            "platform": str(item.get("platform", ""))[:40],
            "platformObjectId": str(item.get("platformObjectId", item.get("platform_object_id", "")))[:120],
            "commentId": str(item.get("commentId", item.get("comment_id", "")))[:120],
            "citationRef": str(item.get("citationRef", item.get("citation_ref", "")))[:200],
        }
        start_ms = item.get("startMs", item.get("start_ms"))
        end_ms = item.get("endMs", item.get("end_ms"))
        if start_ms is not None:
            ref["startMs"] = max(0, int(start_ms))
        if end_ms is not None:
            ref["endMs"] = max(0, int(end_ms))
        sanitized.append(ref)
    return sanitized[:12]


def _sanitize_insights(items: Any, allowed_ids: set[int]) -> list[dict[str, Any]]:
    allowed = {"answer", "consensus", "disagreement", "risk", "opportunity", "coverage"}
    sanitized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return sanitized
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        kind = str(item.get("kind", "consensus"))
        sanitized.append({
            "id": _safe_id(str(item.get("id", f"insight-{idx}"))),
            "kind": kind if kind in allowed else "consensus",
            "label": label,
            "summary": str(item.get("summary", ""))[:360],
            "confidence": _int_range(item.get("confidence", 3), 1, 5),
            "sourceIds": valid_ids(item.get("sourceIds", item.get("citations", [])), allowed_ids),
        })
    return sanitized


def _sanitize_sections(items: Any, allowed_ids: set[int]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return sanitized
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        body = str(item.get("body", "")).strip()
        if not title or not body:
            continue
        section = {
            "id": _safe_id(str(item.get("id", f"section-{idx}"))),
            "title": title,
            "kind": str(item.get("kind", "narrative"))[:40],
            "level": _int_range(item.get("level", 1), 1, 3),
            "body": body,
            "bullets": _sanitize_citation_items(item.get("bullets", []), allowed_ids),
            "quote": None,
            "table": _sanitize_table_rows(item.get("table", []), allowed_ids),
            "sourceIds": valid_ids(item.get("sourceIds", item.get("citations", [])), allowed_ids),
            "citationRefs": _sanitize_citation_refs(item.get("citationRefs", []), allowed_ids),
            "metrics": item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {},
            "data": item.get("data", {}) if isinstance(item.get("data"), dict) else {},
        }
        sanitized.append(section)
    return sanitized


def _sanitize_table_rows(items: Any, allowed_ids: set[int]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return sanitized
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        sanitized.append({
            "name": name[:120],
            "signal": str(item.get("signal", ""))[:260],
            "support": _int_range(item.get("support", item.get("lowLight", 3)), 1, 5),
            "risk": _int_range(item.get("risk", 3), 1, 5),
            "freshness": _int_range(item.get("freshness", item.get("battery", 3)), 1, 5),
            "confidence": _int_range(item.get("confidence", item.get("camera", 3)), 1, 5),
            "price": str(item.get("price", ""))[:80],
            "dimensions": _sanitize_dimensions(item.get("dimensions", []), allowed_ids),
            "metrics": item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {},
            "evidence": valid_ids(item.get("evidence", item.get("sourceIds", [])), allowed_ids),
        })
    return sanitized


def _sanitize_dimensions(items: Any, allowed_ids: set[int]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return sanitized
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", item.get("dimension", ""))).strip()
        label = str(item.get("label", key)).strip()
        if not key or not label:
            continue
        sanitized.append({
            "key": _safe_id(key)[:40],
            "label": label[:80],
            "score": _int_range(item.get("score"), 1, 5) if item.get("score") is not None else None,
            "summary": str(item.get("summary", item.get("conclusion", "")))[:260],
            "evidence": valid_ids(item.get("evidence", item.get("sourceIds", [])), allowed_ids),
            "metrics": item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {},
        })
    return sanitized[:8]


def sanitize_semantic_evidence(items: Any, allowed_ids: set[int]) -> list[SemanticEvidenceExtraction]:
    if not isinstance(items, list):
        return []
    evidence: list[SemanticEvidenceExtraction] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source_ids = valid_ids(item.get("sourceId", item.get("source_id", item.get("sourceIds", []))), allowed_ids, limit=1)
        source_id = source_ids[0] if source_ids else 0
        if not source_id:
            continue
        candidate = str(item.get("candidate", item.get("model", item.get("name", "")))).strip()
        dimension = str(item.get("dimension", item.get("key", ""))).strip()
        conclusion = str(item.get("conclusion", item.get("summary", ""))).strip()
        if not dimension or not conclusion:
            continue
        evidence.append(SemanticEvidenceExtraction(
            candidate=candidate[:120],
            dimension=dimension[:80],
            conclusion=conclusion[:360],
            scenario=str(item.get("scenario", ""))[:180],
            value=str(item.get("value", item.get("metric", ""))[:120] if isinstance(item.get("value", item.get("metric", "")), str) else item.get("value", item.get("metric", "")))[:120],
            quote=str(item.get("quote", item.get("excerpt", "")))[:280],
            sourceId=source_id,
            confidence=_int_range(item.get("confidence", 3), 1, 5),
        ))
    return evidence[:80]


def sanitize_decision_products(items: Any, allowed_ids: set[int]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if not isinstance(items, dict):
        return {}
    products: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for product_name, dimensions in items.items():
        name = str(product_name).strip()[:120]
        if not name or not isinstance(dimensions, dict):
            continue
        clean_dimensions: dict[str, list[dict[str, Any]]] = {}
        for dimension_name, values in dimensions.items():
            dimension = str(dimension_name).strip()[:80]
            if not dimension:
                continue
            value_items = values if isinstance(values, list) else [values]
            clean_values: list[dict[str, Any]] = []
            for value in value_items:
                if not isinstance(value, dict):
                    continue
                source_ids = valid_ids(value.get("sourceIds", value.get("sources", value.get("sourceId", []))), allowed_ids)
                if not source_ids:
                    continue
                conclusion = str(value.get("conclusion", value.get("summary", ""))).strip()
                if not conclusion:
                    continue
                clean_values.append({
                    "conclusion": conclusion[:360],
                    "condition": str(value.get("condition", value.get("testCondition", value.get("scenario", ""))) or "")[:180],
                    "sourceIds": source_ids,
                    "value": str(value.get("value", value.get("metric", "")) or "")[:140],
                })
            if clean_values:
                clean_dimensions[dimension] = clean_values[:8]
        if clean_dimensions:
            products[name] = clean_dimensions
    return products


def sanitize_contradictions(items: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(items, dict):
        return {}
    contradictions: dict[str, dict[str, Any]] = {}
    for dimension_name, value in items.items():
        if not isinstance(value, dict):
            continue
        dimension = str(dimension_name).strip()[:80]
        if not dimension:
            continue
        contradictions[dimension] = {
            "isContradictory": bool(value.get("isContradictory", value.get("contradictory", False))),
            "contradictionDescription": str(value.get("contradictionDescription", value.get("description", "")) or "")[:360],
            "possibleReason": str(value.get("possibleReason", value.get("reason", "")) or "")[:360],
        }
    return contradictions


def valid_ids(values: Any, allowed_ids: set[int], limit: int | None = None) -> list[int]:
    if isinstance(values, (str, int)):
        values = [values]
    if not isinstance(values, list):
        return []
    ids: list[int] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("id", value.get("sourceId", value.get("source_id")))
        try:
            source_id = int(value)
        except (TypeError, ValueError):
            continue
        if source_id in allowed_ids and source_id not in ids:
            ids.append(source_id)
            if limit is not None and len(ids) >= limit:
                break
    return ids


def _safe_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug[:48] or "section"


def _int_range(value: Any, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = low
    return max(low, min(high, parsed))
