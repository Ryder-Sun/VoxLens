from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import httpx

from app.models import CitationText, CoverageSummary, ReportInsight, ReportSection, Source
from app.services.llm_transport import (
    LLMTransportError,
    gather_independent_calls,
    post_with_retry,
)
from app.services.llm_types import (
    DecisionAttributeExtractionResult,
    DecisionContradictionResult,
    LlmSynthesisResult,
    SemanticEvidenceExtraction,
    SemanticEvidenceExtractionResult,
    SourceSelectionResult,
)
from app.services.llm_sanitizers import (
    result_from_payload,
    sanitize_contradictions,
    sanitize_decision_products,
    sanitize_semantic_evidence,
    valid_ids,
)
from app.services.report_templates import business_template_payload
from app.utils import extract_first_json, text_excerpt


def synthesize_with_llm(
    *,
    need: str,
    query: str,
    lang: str,
    sources: list[Source],
    coverage: CoverageSummary,
    report_kind: str,
) -> LlmSynthesisResult | None:
    if not _enabled() or not sources:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.1")
    semantic_result = extract_semantic_evidence_with_llm(
        need=need,
        query=query,
        lang=lang,
        sources=sources,
        report_kind=report_kind,
    )
    semantic_evidence = semantic_result.evidence if semantic_result else []
    decision_attributes: DecisionAttributeExtractionResult | None = None
    decision_contradictions: DecisionContradictionResult | None = None
    if report_kind in {"consumer", "decision"}:
        decision_attributes = extract_decision_attributes_with_llm(
            need=need,
            query=query,
            lang=lang,
            sources=sources,
        )
        if decision_attributes:
            decision_contradictions = analyze_decision_contradictions_with_llm(
                need=need,
                query=query,
                lang=lang,
                attributes=decision_attributes.products,
            )
    selection = select_sources_with_llm(
        need=need,
        query=query,
        lang=lang,
        sources=sources,
        coverage=coverage,
        report_kind=report_kind,
        semantic_evidence=semantic_evidence,
    )
    synthesis_sources = selection.sources if selection and selection.sources else sources
    selected_source_ids = [source.id for source in synthesis_sources]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt(lang, report_kind)},
            {
                "role": "user",
                "content": _user_prompt(
                    need=need,
                    query=query,
                    lang=lang,
                    sources=synthesis_sources,
                    coverage=coverage,
                    report_kind=report_kind,
                    selected_source_ids=selected_source_ids,
                    total_source_count=len(sources),
                    semantic_evidence=semantic_evidence,
                    decision_attributes=decision_attributes.products if decision_attributes else {},
                    contradiction_analysis=decision_contradictions.contradictions if decision_contradictions else {},
                ),
            },
        ],
        "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0.45")),
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", "5000")),
        "response_format": {"type": "json_object"},
    }
    content = _post_chat_completion(api_key, payload)
    parsed = _parse_json(content, array_key="sections")
    result = result_from_payload(parsed, set(selected_source_ids))
    if selection and selection.warnings:
        result.warnings = [*selection.warnings, *result.warnings][:6]
    if semantic_result and semantic_result.warnings:
        result.warnings = [*semantic_result.warnings, *result.warnings][:6]
    if decision_attributes and decision_attributes.warnings:
        result.warnings = [*decision_attributes.warnings, *result.warnings][:6]
    if decision_contradictions and decision_contradictions.warnings:
        result.warnings = [*decision_contradictions.warnings, *result.warnings][:6]
    result.model = model
    return result


async def synthesize_with_llm_async(
    *,
    need: str,
    query: str,
    lang: str,
    sources: list[Source],
    coverage: CoverageSummary,
    report_kind: str,
    client: httpx.AsyncClient | None = None,
) -> LlmSynthesisResult | None:
    """Async synthesis with concurrent independent LLM calls.

    Runs semantic-evidence extraction and decision-attribute extraction in
    parallel when both are needed, then sequentially runs contradiction
    analysis (depends on attributes), source selection (depends on semantic
    evidence), and final report synthesis.
    """

    if not _enabled() or not sources:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.1")
    needs_decision = report_kind in {"consumer", "decision"}

    # Build payloads for independent calls that can run concurrently
    independent_payloads: list[dict[str, Any]] = []
    semantic_index = 0
    decision_index = -1

    semantic_payload = _build_semantic_payload(model, need, query, lang, sources, report_kind)
    independent_payloads.append(semantic_payload)

    if needs_decision:
        decision_index = len(independent_payloads)
        independent_payloads.append(_build_decision_attribute_payload(model, need, query, lang, sources))

    # Fire independent calls concurrently
    results = await gather_independent_calls(independent_payloads, client=client, max_concurrency=3)

    # Parse semantic evidence
    semantic_content = results[semantic_index]
    semantic_result: SemanticEvidenceExtractionResult | None = None
    if isinstance(semantic_content, str):
        parsed_semantic = _parse_json(semantic_content, array_key="evidence")
        valid_source_ids = {source.id for source in sources}
        semantic_result = SemanticEvidenceExtractionResult(
            evidence=sanitize_semantic_evidence(parsed_semantic.get("evidence", parsed_semantic.get("semanticEvidence", [])), valid_source_ids),
            warnings=[str(item)[:240] for item in parsed_semantic.get("warnings", []) if str(item).strip()][:6],
            model=model,
        )
    semantic_evidence = semantic_result.evidence if semantic_result else []

    # Parse decision attributes if we ran that call
    decision_attributes: DecisionAttributeExtractionResult | None = None
    if needs_decision and decision_index >= 0:
        decision_content = results[decision_index]
        if isinstance(decision_content, str):
            parsed_decision = _parse_json(decision_content)
            valid_source_ids = {source.id for source in sources}
            decision_attributes = DecisionAttributeExtractionResult(
                products=sanitize_decision_products(parsed_decision.get("products", parsed_decision.get("productAttributes", {})), valid_source_ids),
                warnings=[str(item)[:240] for item in parsed_decision.get("warnings", []) if str(item).strip()][:6],
                model=model,
            )

    # Contradiction analysis depends on decision attributes
    decision_contradictions: DecisionContradictionResult | None = None
    if decision_attributes and decision_attributes.products:
        contradiction_payload = _build_contradiction_payload(model, need, query, lang, decision_attributes.products)
        contradiction_content = await post_with_retry(contradiction_payload, api_key=api_key, client=client)
        if isinstance(contradiction_content, str):
            parsed_contradiction = _parse_json(contradiction_content)
            decision_contradictions = DecisionContradictionResult(
                contradictions=sanitize_contradictions(parsed_contradiction.get("contradictions", parsed_contradiction.get("contradictionAnalysis", {}))),
                warnings=[str(item)[:240] for item in parsed_contradiction.get("warnings", []) if str(item).strip()][:6],
                model=model,
            )

    # Source selection depends on semantic evidence
    selection_payload = _build_selection_payload(model, need, query, lang, sources, coverage, report_kind, semantic_evidence)
    selection_content = await post_with_retry(selection_payload, api_key=api_key, client=client)
    selected_ids: list[int] = []
    synthesis_sources = sources
    selection_warnings: list[str] = []
    if isinstance(selection_content, str):
        parsed_selection = _parse_json(selection_content, array_key="selectedSourceIds")
        selected_ids = valid_ids(
            parsed_selection.get("selectedSourceIds", parsed_selection.get("sourceIds", parsed_selection.get("sources", []))),
            {source.id for source in sources},
        )
        sources_by_id = {source.id: source for source in sources}
        synthesis_sources = [sources_by_id[sid] for sid in selected_ids if sid in sources_by_id] or sources
        selection_warnings = [str(item)[:240] for item in parsed_selection.get("warnings", []) if str(item).strip()][:6]

    selected_source_ids = [source.id for source in synthesis_sources]

    # Final synthesis
    synthesis_payload = _build_synthesis_payload(model, lang, report_kind, need, query, synthesis_sources, coverage, selected_source_ids, len(sources), semantic_evidence, decision_attributes, decision_contradictions)
    content = await post_with_retry(synthesis_payload, api_key=api_key, client=client)
    parsed = _parse_json(content, array_key="sections")
    result = result_from_payload(parsed, set(selected_source_ids))
    if selection_warnings:
        result.warnings = [*selection_warnings, *result.warnings][:6]
    if semantic_result and semantic_result.warnings:
        result.warnings = [*semantic_result.warnings, *result.warnings][:6]
    if decision_attributes and decision_attributes.warnings:
        result.warnings = [*decision_attributes.warnings, *result.warnings][:6]
    if decision_contradictions and decision_contradictions.warnings:
        result.warnings = [*decision_contradictions.warnings, *result.warnings][:6]
    result.model = model
    return result


def _build_semantic_payload(model: str, need: str, query: str, lang: str, sources: list[Source], report_kind: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _semantic_system_prompt(lang, report_kind)},
            {"role": "user", "content": _semantic_extraction_prompt(need=need, query=query, lang=lang, sources=sources, report_kind=report_kind)},
        ],
        "temperature": float(os.getenv("OPENROUTER_EXTRACTION_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("OPENROUTER_EXTRACTION_MAX_TOKENS", "3500")),
        "response_format": {"type": "json_object"},
    }


def _build_decision_attribute_payload(model: str, need: str, query: str, lang: str, sources: list[Source]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _decision_attribute_system_prompt(lang)},
            {"role": "user", "content": _decision_attribute_prompt(need=need, query=query, lang=lang, sources=sources)},
        ],
        "temperature": float(os.getenv("OPENROUTER_EXTRACTION_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("OPENROUTER_DECISION_ATTRIBUTE_MAX_TOKENS", "3500")),
        "response_format": {"type": "json_object"},
    }


def _build_contradiction_payload(model: str, need: str, query: str, lang: str, attributes: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _decision_contradiction_system_prompt(lang)},
            {"role": "user", "content": _decision_contradiction_prompt(need=need, query=query, lang=lang, attributes=attributes)},
        ],
        "temperature": float(os.getenv("OPENROUTER_CONTRADICTION_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("OPENROUTER_CONTRADICTION_MAX_TOKENS", "2200")),
        "response_format": {"type": "json_object"},
    }


def _build_selection_payload(model: str, need: str, query: str, lang: str, sources: list[Source], coverage: CoverageSummary, report_kind: str, semantic_evidence: list[SemanticEvidenceExtraction]) -> dict[str, Any]:
    max_selected = int(os.getenv("VOXLENS_LLM_SELECTED_SOURCE_LIMIT", "24"))
    max_selected = max(1, min(max_selected, len(sources)))
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _selection_system_prompt(lang)},
            {"role": "user", "content": _selection_prompt(need=need, query=query, lang=lang, sources=sources, coverage=coverage, report_kind=report_kind, semantic_evidence=semantic_evidence, max_sources=max_selected)},
        ],
        "temperature": float(os.getenv("OPENROUTER_SELECTION_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("OPENROUTER_SELECTION_MAX_TOKENS", "1800")),
        "response_format": {"type": "json_object"},
    }


def _build_synthesis_payload(
    model: str, lang: str, report_kind: str, need: str, query: str,
    synthesis_sources: list[Source], coverage: CoverageSummary,
    selected_source_ids: list[int], total_source_count: int,
    semantic_evidence: list[SemanticEvidenceExtraction],
    decision_attributes: DecisionAttributeExtractionResult | None,
    decision_contradictions: DecisionContradictionResult | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt(lang, report_kind)},
            {
                "role": "user",
                "content": _user_prompt(
                    need=need, query=query, lang=lang, sources=synthesis_sources,
                    coverage=coverage, report_kind=report_kind,
                    selected_source_ids=selected_source_ids, total_source_count=total_source_count,
                    semantic_evidence=semantic_evidence,
                    decision_attributes=decision_attributes.products if decision_attributes else {},
                    contradiction_analysis=decision_contradictions.contradictions if decision_contradictions else {},
                ),
            },
        ],
        "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0.45")),
        "max_tokens": int(os.getenv("OPENROUTER_MAX_TOKENS", "5000")),
        "response_format": {"type": "json_object"},
    }


def extract_semantic_evidence_with_llm(
    *,
    need: str,
    query: str,
    lang: str,
    sources: list[Source],
    report_kind: str,
) -> SemanticEvidenceExtractionResult | None:
    if not _enabled() or not sources:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.1")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _semantic_system_prompt(lang, report_kind)},
            {
                "role": "user",
                "content": _semantic_extraction_prompt(
                    need=need,
                    query=query,
                    lang=lang,
                    sources=sources,
                    report_kind=report_kind,
                ),
            },
        ],
        "temperature": float(os.getenv("OPENROUTER_EXTRACTION_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("OPENROUTER_EXTRACTION_MAX_TOKENS", "3500")),
        "response_format": {"type": "json_object"},
    }
    content = _post_chat_completion(api_key, payload)
    parsed = _parse_json(content, array_key="evidence")
    valid_ids = {source.id for source in sources}
    return SemanticEvidenceExtractionResult(
        evidence=sanitize_semantic_evidence(parsed.get("evidence", parsed.get("semanticEvidence", [])), valid_ids),
        warnings=[str(item)[:240] for item in parsed.get("warnings", []) if str(item).strip()][:6],
        model=model,
    )


def extract_decision_attributes_with_llm(
    *,
    need: str,
    query: str,
    lang: str,
    sources: list[Source],
) -> DecisionAttributeExtractionResult | None:
    if not _enabled() or not sources:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.1")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _decision_attribute_system_prompt(lang)},
            {
                "role": "user",
                "content": _decision_attribute_prompt(
                    need=need,
                    query=query,
                    lang=lang,
                    sources=sources,
                ),
            },
        ],
        "temperature": float(os.getenv("OPENROUTER_EXTRACTION_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("OPENROUTER_DECISION_ATTRIBUTE_MAX_TOKENS", "3500")),
        "response_format": {"type": "json_object"},
    }
    content = _post_chat_completion(api_key, payload)
    parsed = _parse_json(content)
    valid_ids = {source.id for source in sources}
    return DecisionAttributeExtractionResult(
        products=sanitize_decision_products(parsed.get("products", parsed.get("productAttributes", {})), valid_ids),
        warnings=[str(item)[:240] for item in parsed.get("warnings", []) if str(item).strip()][:6],
        model=model,
    )


def analyze_decision_contradictions_with_llm(
    *,
    need: str,
    query: str,
    lang: str,
    attributes: dict[str, dict[str, list[dict[str, Any]]]],
) -> DecisionContradictionResult | None:
    if not _enabled() or not attributes:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.1")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _decision_contradiction_system_prompt(lang)},
            {
                "role": "user",
                "content": _decision_contradiction_prompt(
                    need=need,
                    query=query,
                    lang=lang,
                    attributes=attributes,
                ),
            },
        ],
        "temperature": float(os.getenv("OPENROUTER_CONTRADICTION_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("OPENROUTER_CONTRADICTION_MAX_TOKENS", "2200")),
        "response_format": {"type": "json_object"},
    }
    content = _post_chat_completion(api_key, payload)
    parsed = _parse_json(content)
    return DecisionContradictionResult(
        contradictions=sanitize_contradictions(parsed.get("contradictions", parsed.get("contradictionAnalysis", {}))),
        warnings=[str(item)[:240] for item in parsed.get("warnings", []) if str(item).strip()][:6],
        model=model,
    )


def select_sources_with_llm(
    *,
    need: str,
    query: str,
    lang: str,
    sources: list[Source],
    coverage: CoverageSummary,
    report_kind: str,
    semantic_evidence: list[SemanticEvidenceExtraction] | None = None,
    max_sources: int | None = None,
) -> SourceSelectionResult | None:
    if not _enabled() or not sources:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.1")
    max_selected = max_sources or int(os.getenv("VOXLENS_LLM_SELECTED_SOURCE_LIMIT", "24"))
    max_selected = max(1, min(max_selected, len(sources)))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _selection_system_prompt(lang)},
            {
                "role": "user",
                "content": _selection_prompt(
                    need=need,
                    query=query,
                    lang=lang,
                    sources=sources,
                    coverage=coverage,
                    report_kind=report_kind,
                    semantic_evidence=semantic_evidence or [],
                    max_sources=max_selected,
                ),
            },
        ],
        "temperature": float(os.getenv("OPENROUTER_SELECTION_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("OPENROUTER_SELECTION_MAX_TOKENS", "1800")),
        "response_format": {"type": "json_object"},
    }
    content = _post_chat_completion(api_key, payload)
    parsed = _parse_json(content, array_key="selectedSourceIds")
    selected_ids = valid_ids(
        parsed.get("selectedSourceIds", parsed.get("sourceIds", parsed.get("sources", []))),
        {source.id for source in sources},
        limit=max_selected,
    )
    sources_by_id = {source.id: source for source in sources}
    selected_sources = [sources_by_id[source_id] for source_id in selected_ids if source_id in sources_by_id]
    warnings = [str(item)[:240] for item in parsed.get("warnings", []) if str(item).strip()]
    return SourceSelectionResult(
        sources=selected_sources,
        source_ids=selected_ids,
        warnings=warnings[:6],
        model=model,
    )


def _enabled() -> bool:
    return os.getenv("VOXLENS_ENABLE_LLM", "true").strip().lower() not in {"0", "false", "no", "off"}


def _post_chat_completion(api_key: str, payload: dict[str, Any]) -> str:
    """Synchronous wrapper around the async transport for backward compatibility.

    Deprecated: prefer synthesize_with_llm_async for new callers. This keeps
    existing synchronous call sites (e.g. report_builder.py fallback) working.
    """

    return asyncio.run(post_with_retry(payload, api_key=api_key))


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item.get("content"), str):
                    parts.append(str(item["content"]))
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return ""


def _parse_json(content: str, *, array_key: str = "items") -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = extract_first_json(content)
    if isinstance(value, list):
        return {array_key: value}
    if not isinstance(value, dict):
        raise ValueError("LLM synthesis payload is not an object")
    return value


def _selection_system_prompt(lang: str) -> str:
    if lang == "zh":
        return (
            "你是 VoxLens 的证据筛选员。你会看到本轮收集到的全部 sourceCandidates，"
            "任务是选择最值得进入正文合成的高置信视频/内容来源。只输出 JSON。"
        )
    return (
        "You are the evidence selection analyst for VoxLens. You will see every collected sourceCandidate. "
        "Select the highest-confidence sources that should be imported into report synthesis. Return JSON only."
    )


def _semantic_system_prompt(lang: str, report_kind: str = "general") -> str:
    if report_kind == "business":
        if lang == "zh":
            return (
                "你是 VoxLens 的 B 端产品/品类研究证据提取员。你会读取完整字幕/正文分块、评论和来源元数据，"
                "只提取能支撑品类范围、竞品格局、用户需求、机会和风险的结构化证据。不要写报告，只输出 JSON。"
            )
        return (
            "You extract evidence for VoxLens B2B product/category research. Read transcript/content chunks, comments and metadata, "
            "then extract structured evidence for category scope, competitors, user demand, opportunities and risks. Return JSON only."
        )
    if lang == "zh":
        return (
            "你是 VoxLens 的语义证据提取员。你会读取完整字幕/正文分块、评论和来源元数据，"
            "只提取能支撑消费决策的结构化证据。不要写报告，只输出 JSON。"
        )
    return (
        "You are the semantic evidence extraction analyst for VoxLens. Read transcript/content chunks, "
        "comments and metadata, then extract structured evidence for decision-making. Return JSON only."
    )


def _decision_attribute_system_prompt(lang: str) -> str:
    if lang == "zh":
        return (
            "你是消费电子测评的结构化证据抽取员。你的任务是从多个视频/内容来源中抽取每个产品的属性、"
            "实测数值、场景评价和对比结论。维度必须来自来源内容，不要套用固定模板。只输出 JSON。"
        )
    return (
        "You extract structured evidence from consumer-electronics reviews. Extract each product's attributes, "
        "measured values, scenario evaluations and comparison conclusions. Dimensions must come from the sources, not a fixed template. Return JSON only."
    )


def _decision_contradiction_system_prompt(lang: str) -> str:
    if lang == "zh":
        return (
            "你是消费决策的交叉验证分析师。你会比较同一维度下的多个来源结论，明确哪些结论矛盾，"
            "并解释可能原因，例如测试条件、使用场景或评价标准不同。只输出 JSON。"
        )
    return (
        "You are a cross-validation analyst for purchase decisions. Compare conclusions under the same dimension, "
        "flag contradictions and explain likely causes such as test conditions, usage scenarios or standards. Return JSON only."
    )


def _semantic_extraction_prompt(
    *,
    need: str,
    query: str,
    lang: str,
    sources: list[Source],
    report_kind: str,
) -> str:
    zh = lang == "zh"
    if report_kind == "business":
        task = (
            "从每个来源中提取品类/产品/竞品/用户场景、研究维度、证据结论、适用上下文、具体数值或描述、短引用片段和置信度。"
            "维度名称必须来自标题、评论、字幕/正文中的真实表述；如果证据不足就不要补编。"
            if zh
            else "Extract category/product/competitor/user-scenario, research dimension, evidence conclusion, context, measured value or description, short quote and confidence from each source. Dimension names must come from real wording in titles, comments or transcripts/content; do not invent missing evidence."
        )
    else:
        task = (
            "从每个来源中提取候选产品、维度、结论、测试/使用场景、实测值或描述、短引用片段和置信度。"
            "维度名称必须来自标题、评论、字幕/正文中的真实表述；如果证据不足就不要补编。"
            if zh
            else "Extract candidate product, dimension, conclusion, test/use scenario, measured value or description, short quote and confidence from each source. Dimension names must come from real wording in titles, comments or transcripts/content; do not invent missing evidence."
        )
    return json.dumps(
        {
            "task": task,
            "need": need,
            "query": query,
            "reportKind": report_kind,
            "outputSchema": {
                "evidence": [
                    {
                        "candidate": "string",
                        "dimension": "source-derived attribute name",
                        "conclusion": "string",
                        "scenario": "string",
                        "value": "string",
                        "quote": "short source-grounded quote",
                        "sourceId": 1,
                        "confidence": 1,
                    }
                ],
                "warnings": ["string"],
            },
            "sourceEvidencePacks": [_source_evidence_pack(source) for source in sources],
        },
        ensure_ascii=False,
    )


def _decision_attribute_prompt(
    *,
    need: str,
    query: str,
    lang: str,
    sources: list[Source],
) -> str:
    zh = lang == "zh"
    task = (
        "先从所有来源中抽取结构化消费决策证据，输出 {产品: {维度: [{结论, 测试条件, 来源IDs}]}}。"
        "每个维度必须来自来源标题、评论或字幕/正文；实测数字优先；没有证据不要编造。"
        if zh
        else "Extract structured purchase-decision evidence from all sources as {product: {dimension: [{conclusion, condition, sourceIds}]}}. "
        "Each dimension must come from titles, comments or transcripts/content; measured numbers outrank subjective comments; do not invent evidence."
    )
    return json.dumps(
        {
            "task": task,
            "need": need,
            "query": query,
            "outputSchema": {
                "products": {
                    "Product name": {
                        "source-derived dimension": [
                            {
                                "conclusion": "source-grounded conclusion",
                                "condition": "test or use condition",
                                "value": "measured value if present",
                                "sourceIds": [1],
                            }
                        ]
                    }
                },
                "warnings": ["string"],
            },
            "sourceEvidencePacks": [_source_evidence_pack(source) for source in sources],
        },
        ensure_ascii=False,
    )


def _decision_contradiction_prompt(
    *,
    need: str,
    query: str,
    lang: str,
    attributes: dict[str, dict[str, list[dict[str, Any]]]],
) -> str:
    zh = lang == "zh"
    task = (
        "比较同一维度下不同产品/来源的结论，标注是否矛盾，并解释可能原因。重点说明为什么矛盾，而不是回避矛盾。"
        if zh
        else "Compare conclusions under each dimension across products/sources, flag contradictions and explain likely causes. Explain why contradictions happen instead of hiding them."
    )
    return json.dumps(
        {
            "task": task,
            "need": need,
            "query": query,
            "decisionAttributeMatrix": attributes,
            "outputSchema": {
                "contradictions": {
                    "dimension": {
                        "isContradictory": True,
                        "contradictionDescription": "what conflicts",
                        "possibleReason": "test condition, scenario, standard or source difference",
                    }
                },
                "warnings": ["string"],
            },
        },
        ensure_ascii=False,
    )


def _selection_prompt(
    *,
    need: str,
    query: str,
    lang: str,
    sources: list[Source],
    coverage: CoverageSummary,
    report_kind: str,
    semantic_evidence: list[SemanticEvidenceExtraction],
    max_sources: int,
) -> str:
    zh = lang == "zh"
    if report_kind == "business":
        instructions = (
            "从全部候选中选择最能支撑 B 端产品/品类研究的高置信来源。优先选择能覆盖品类边界、竞品、用户需求、卖点、渠道/价格、机会和风险的来源；"
            "优先有评论、字幕/文本、摘要、明确 URL、跨平台互证和 semanticEvidence 的来源；避免只因排序靠前或热度高就选择。返回 selectedSourceIds，按推荐引用优先级排序。"
            if zh
            else "Select the highest-confidence sources for B2B product/category research. Prefer sources that cover category boundaries, competitors, user demand, selling points, channel/price, opportunities and risks; prioritize comments, transcript/text, summaries, URLs, cross-platform corroboration and semanticEvidence. Do not select only because a source appears early or is popular. Return selectedSourceIds in recommended citation priority order."
        )
    else:
        instructions = (
            "从全部候选中选择最能支撑用户问题的高置信来源。优先选择有评论、字幕/文本、摘要、明确 URL、跨平台互证和与问题直接相关的来源；"
            "尤其优先选择 semanticEvidence 能覆盖用户决策维度、解释矛盾场景的来源；避免只因排序靠前或热度高就选择。返回 selectedSourceIds，按推荐引用优先级排序。"
            if zh
            else "Select the highest-confidence sources for the user question from all candidates. Prefer sources with comments, transcript/text, summaries, URLs, cross-platform corroboration, direct relevance and semanticEvidence that covers decision dimensions or explains contradictory scenarios; do not select only because a source appears early or is popular. Return selectedSourceIds in recommended citation priority order."
        )
    evidence_by_source = _semantic_evidence_by_source(semantic_evidence)
    return json.dumps(
        {
            "task": instructions,
            "need": need,
            "query": query,
            "reportKind": report_kind,
            "coverage": coverage.model_dump(),
            "maxSelectedSources": max_sources,
            "outputSchema": {
                "selectedSourceIds": [1],
                "warnings": ["string"],
            },
            "sourceCandidates": [_source_candidate_payload(source, evidence_by_source.get(source.id, [])) for source in sources],
        },
        ensure_ascii=False,
    )


def _system_prompt(lang: str, report_kind: str = "general") -> str:
    if report_kind == "consumer":
        if lang == "zh":
            return (
                "你是普通用户买前决策助手。用户不是产品经理或市场研究员，他只是想少看很多测评视频，"
                "更快知道应该买什么、为什么、什么情况下不要买。\n"
                "规则：\n"
                "1. 用用户能听懂的话，先给清晰建议，再解释证据。\n"
                "2. 不使用 B 端固定研究模板，不写市场规模、品类格局或商业建议。\n"
                "3. 评价维度从来源标题、评论、字幕/正文中提取；不要套用固定品类模板。\n"
                "4. 所有关键建议、避坑点和分歧都必须引用 source id。\n"
                "5. 如果证据不足，不要假装确定；说明还需要用户确认的预算、场景或禁区。"
            )
        return (
            "You are a pre-purchase advisor for ordinary shoppers. The user is not a product manager or market researcher; "
            "they want to avoid watching many review videos and quickly understand what to buy, why, and when not to buy.\n"
            "Rules:\n"
            "1. Use plain buyer-facing language: clear recommendation first, evidence second.\n"
            "2. Do not use a fixed B2B research template, market sizing, category landscape or business actions.\n"
            "3. Extract evaluation dimensions from source titles, comments and transcripts/content; do not apply a fixed category template.\n"
            "4. Cite source ids for every key recommendation, caveat and disagreement.\n"
            "5. If evidence is thin, say what budget, scenario or non-negotiable the user still needs to confirm."
        )
    if report_kind == "decision":
        if lang == "zh":
            return (
                "你是消费决策分析师。你收到的来源是多个视频博主对消费电子产品的测评。\n"
                "你的任务：\n"
                "1. 提取每个被测产品的结构化属性（实测数值、场景评价、对比结论等）。\n"
                "2. 当多个来源对同一属性结论矛盾时，分析可能原因（测试条件、使用场景、标准不同）。\n"
                "3. 输出对比矩阵和购买建议。\n"
                "规则：\n"
                "1. 每个属性值必须标注来源ID。\n"
                "2. 实测数据（数字）优先于主观评价。\n"
                "3. 发现矛盾时不要回避，明确说明“为什么矛盾”比“有没有矛盾”更重要。\n"
                "4. 不要编造来源中没有的数据。\n"
                "5. 不同品类的评价维度不同，从来源内容中提取，不要套用固定维度模板。"
            )
        return (
            "You are a consumer purchase-decision analyst. Sources are video creators reviewing consumer electronics.\n"
            "Tasks:\n"
            "1. Extract structured attributes for each reviewed product: measured values, scenario evaluations and comparison conclusions.\n"
            "2. When sources conflict on the same attribute, explain likely causes such as test conditions, usage scenarios or standards.\n"
            "3. Produce a comparison matrix and purchase recommendation.\n"
            "Rules:\n"
            "1. Every attribute value must cite source IDs.\n"
            "2. Measured numeric data outranks subjective evaluation.\n"
            "3. Do not hide contradictions; explaining why they conflict matters more than only saying whether they conflict.\n"
            "4. Do not invent data absent from sources.\n"
            "5. Dimensions differ by product category; extract them from source content instead of using a fixed template."
        )
    if report_kind == "business":
        if lang == "zh":
            return (
                "你是 B 端产品/品类研究分析师。你的用户是产品、市场、运营或品牌团队，不是个人购物用户。\n"
                "你的任务：\n"
                "1. 只基于给定 sources、comments、transcripts 提炼品类范围、竞品格局、用户需求、卖点、机会和风险。\n"
                "2. 覆盖这些信息目标：研究结论摘要、品类/产品范围、竞品与内容格局、用户声音与需求信号、机会/风险/分歧、证据缺口与下一步研究；段落结构和数量由你根据证据设计，不要机械套用固定模板。\n"
                "3. 明确区分来源证据和你的业务推断；所有关键判断必须引用 source id。\n"
                "规则：\n"
                "1. 不输出“买 X / 不买 X”的 C 端购买建议。\n"
                "2. 不编造市场规模、份额、价格、渠道或用户画像；来源没有就标为待验证。\n"
                "3. 跨平台重复信号优先于单条爆款内容或标题命中。"
            )
        return (
            "You are a B2B product/category research analyst. The user is a product, marketing, ops or brand team, not an individual shopper.\n"
            "Tasks:\n"
            "1. Use only the supplied sources, comments and transcripts to synthesize category scope, competitors, user demand, selling points, opportunities and risks.\n"
            "2. Cover these information goals: Research Brief, Category & Product Scope, Competitive Landscape, User Voice & Demand Signals, Opportunities/Risks/Disagreements, Evidence Gaps & Next Research — design your own section structure and count from the evidence instead of a fixed template.\n"
            "3. Separate source-backed evidence from business inference; cite source ids for every important claim.\n"
            "Rules:\n"
            "1. Do not produce C-end shopping advice such as 'buy X' or 'do not buy X'.\n"
            "2. Do not invent market size, share, prices, channels or personas; mark absent facts as validation gaps.\n"
            "3. Cross-platform repeated signals outrank single viral posts or title-only matches."
        )
    if lang == "zh":
        return (
            "你是 VoxLens 的证据型视频社媒研究分析师。你必须只依据用户提供的 sources、comments、transcripts 写报告，"
            "所有重要判断都要引用 source id。不要编造没有证据的平台、品牌、数字或用户原话。"
        )
    return (
        "You are the evidence-backed social-video research analyst for VoxLens. Only use the supplied sources, comments and transcripts. "
        "Cite source ids for every important claim. Do not invent platforms, brands, numbers or quotes."
    )


def _user_prompt(
    *,
    need: str,
    query: str,
    lang: str,
    sources: list[Source],
    coverage: CoverageSummary,
    report_kind: str,
    selected_source_ids: list[int] | None = None,
    total_source_count: int | None = None,
    semantic_evidence: list[SemanticEvidenceExtraction] | None = None,
    decision_attributes: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
    contradiction_analysis: dict[str, dict[str, Any]] | None = None,
) -> str:
    zh = lang == "zh"
    selected_source_ids = selected_source_ids or [source.id for source in sources]
    semantic_evidence = semantic_evidence or []
    decision_attributes = decision_attributes or {}
    contradiction_analysis = contradiction_analysis or {}
    evidence_by_source = _semantic_evidence_by_source(semantic_evidence)
    source_pack = [_source_payload(source, evidence_by_source.get(source.id, [])) for source in sources]
    schema = {
        "takeaways": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}],
        "insights": [{"id": "short-id", "kind": "answer|consensus|disagreement|risk|opportunity|coverage", "label": "string", "summary": "string", "confidence": 1, "sourceIds": [1]}],
        "sections": [
            {
                "id": "executive-summary",
                "title": "1. 执行摘要" if zh else "1. Executive Summary",
                "kind": "answer",
                "body": "string",
                "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}],
                "sourceIds": [1],
            }
        ],
        "warnings": ["string"],
    }
    if report_kind == "consumer":
        instructions = (
            "请输出中文 JSON，不要 markdown。写给普通消费者：不要使用固定 B 端模板，也不要写市场/品类研究。可以根据证据选择 3-5 段，但必须包含：清晰建议、为什么、适合/不适合人群、避坑点或还要确认什么。语气要像帮朋友做买前判断，直接、可读、保留出处。"
            if zh
            else "Return English JSON, no markdown. Write for an ordinary shopper: do not use a fixed B2B template or market/category research. Choose 3-5 sections based on evidence, but include a clear recommendation, why, who it fits/doesn't fit, caveats or what to verify next. Use direct buyer-facing language and cite sources."
        )
        schema["sections"] = [
            {"id": "quick-answer", "title": "先说结论" if zh else "Start with the answer", "kind": "consumer_answer", "body": "string", "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}], "sourceIds": [1]},
            {"id": "why", "title": "为什么这么判断" if zh else "Why this answer", "kind": "consumer_reasoning", "body": "string", "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}], "sourceIds": [1]},
            {"id": "fit-check", "title": "适合谁，不适合谁" if zh else "Who it fits, who should skip it", "kind": "consumer_fit", "body": "string", "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}], "sourceIds": [1]},
            {"id": "caveats", "title": "避坑点和还要确认什么" if zh else "Caveats and what to check next", "kind": "consumer_caveats", "body": "string", "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}], "sourceIds": [1]},
        ]
    elif report_kind == "decision":
        instructions = (
            "请输出中文 JSON，不要 markdown。购买决策报告必须使用四段结构：我的需求、候选筛选、维度对比矩阵、最终推荐。维度来自来源内容，不要套用固定品类模板；最终推荐必须包含明确“买X”、条件建议和不推荐情况。"
            if zh
            else "Return English JSON, no markdown. Purchase-decision reports must use four sections: My Needs, Candidate Filtering, Dimension Comparison Matrix, Final Recommendation. Dimensions come from source content, not fixed category templates; the final recommendation must clearly say 'buy X', include conditional advice and not-recommended cases."
        )
        schema["sections"] = [
            {"id": "needs", "title": "1. 我的需求" if zh else "1. My Needs", "kind": "decision_needs", "body": "string", "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}], "sourceIds": [1]},
            {"id": "candidates", "title": "2. 候选筛选" if zh else "2. Candidate Filtering", "kind": "candidate_map", "body": "string", "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}], "sourceIds": [1], "data": {"candidates": [{"name": "string", "matchReason": "string", "sourceIds": [1]}]}},
            {"id": "dimension-comparison", "title": "3. 维度对比矩阵" if zh else "3. Dimension Comparison Matrix", "kind": "decision_matrix", "body": "string", "sourceIds": [1], "data": {"comparisons": [{"dimension": "source-derived name", "values": [{"candidate": "string", "conclusion": "string", "condition": "string", "confidence": "green|yellow|gray", "sourceIds": [1]}], "isContradictory": False, "contradictionReason": "string"}]}},
            {"id": "final-recommendation", "title": "4. 最终推荐" if zh else "4. Final Recommendation", "kind": "final_recommendation", "body": "string", "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}], "sourceIds": [1]},
        ]
    elif report_kind == "business":
        instructions = (
            "请输出中文 JSON，不要 markdown。B 端产品/品类研究报告必须使用六段固定模板：研究结论摘要、品类/产品范围、竞品与内容格局、用户声音与需求信号、机会/风险/分歧、证据缺口与下一步研究。不要写个人购买建议；没有来源支撑的市场规模、份额、价格、渠道或用户画像必须标为待验证。"
            if zh
            else "Return English JSON, no markdown. B2B product/category research must use the six-section fixed template: Research Brief, Category & Product Scope, Competitive Landscape, User Voice & Demand Signals, Opportunities/Risks/Disagreements, Evidence Gaps & Next Research. Do not write personal shopping advice; mark unsupported market size, share, prices, channels or personas as validation gaps."
        )
        schema["sections"] = _business_section_schema(zh)
    else:
        instructions = (
            "请输出中文 JSON，不要 markdown。报告要像业务团队可读的研究报告：执行摘要、关键发现、证据拆解、风险/分歧、建议动作。"
            if zh
            else "Return English JSON, no markdown. Write for business users: executive summary, findings, evidence, risks/disagreements and recommended actions."
        )
    return json.dumps(
        {
            "task": instructions,
            "need": need,
            "query": query,
            "reportKind": report_kind,
            "coverage": coverage.model_dump(),
            "sourceSelection": {
                "selectedSourceIds": selected_source_ids,
                "selectedSourceCount": len(selected_source_ids),
                "totalCollectedSources": total_source_count if total_source_count is not None else len(sources),
                "selectionMethod": "LLM evidence selection over all collected source candidates",
            },
            "sourceRules": [
                "The sources array is the LLM-selected high-confidence subset from the full collected candidate pool.",
                "Use only source ids present in sources.",
                "Every takeaway and section bullet should have at least one citation when possible.",
                "Mention uncertainty when evidence is thin, single-platform, or mostly title-only.",
                "Do not cite a source unless the claim is supported by its title, summary, comments, transcriptPreview, or fullTranscript.",
                "When fullTranscript is present, use it as the richer synthesis context; transcriptPreview is only a UI preview.",
                "Use semanticEvidence first when it is present; it was extracted from fuller transcript/content chunks before synthesis.",
                "Put the source ids that support each section body in section.sourceIds, even when bullets also carry citations.",
                "When citationRefs are available (from transcript segments or comments), include them to anchor claims to precise timestamps or comment locations. Format: {sourceId, startMs, endMs, citationRef, platform, platformObjectId}.",
            ],
            "outputSchema": schema,
            "semanticEvidence": [_semantic_evidence_payload(item) for item in semantic_evidence if item.sourceId in selected_source_ids],
            "decisionAttributeMatrix": decision_attributes,
            "contradictionAnalysis": contradiction_analysis,
            "businessTemplate": business_template_payload(zh) if report_kind == "business" else [],
            "sources": source_pack,
        },
        ensure_ascii=False,
    )


def _business_section_schema(zh: bool) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "title": item["title"],
            "kind": item["kind"],
            "body": "string",
            "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}],
            "sourceIds": [1],
            "metrics": {"confidence": "high|medium|low|insufficient"},
            "data": {"validationGaps": ["unsupported market size/share/price/channel/persona claims"]},
        }
        for item in business_template_payload(zh)
    ]



def _business_section_schema(zh: bool) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "title": item["title"],
            "kind": item["kind"],
            "body": "string",
            "bullets": [{"text": "string", "citations": [1], "citationRefs": [{"sourceId": 1, "startMs": 0, "endMs": 5000, "citationRef": "[PLATFORM:id@MM:SS-MM:SS]"}]}],
            "sourceIds": [1],
            "metrics": {"confidence": "high|medium|low|insufficient"},
            "data": {"validationGaps": ["unsupported market size/share/price/channel/persona claims"]},
        }
        for item in business_template_payload(zh)
    ]


def _source_candidate_payload(source: Source, semantic_evidence: list[SemanticEvidenceExtraction] | None = None) -> dict[str, Any]:
    transcript_text = _full_transcript_text(source)
    return {
        "id": source.id,
        "platform": source.platform,
        "sourceType": source.sourceType,
        "title": source.title[:220],
        "creator": (source.creator or source.author)[:120],
        "urlPresent": bool(source.url),
        "summary": text_excerpt([source.summary], 260),
        "evidenceScore": source.evidenceScore,
        "evidenceChannels": source.evidenceChannels or source.metrics.get("evidence_channels", []),
        "commentCount": len(source.comments),
        "commentsPreview": text_excerpt([comment.text for comment in source.comments[:2]], 220),
        "hasTranscript": bool(source.transcriptPreview or transcript_text),
        "hasFullTranscript": bool(transcript_text),
        "fullTranscriptChars": len(transcript_text),
        "transcriptPreview": text_excerpt([source.transcriptPreview], 260),
        "semanticEvidence": [_semantic_evidence_payload(item) for item in semantic_evidence or []],
        "metrics": _compact_metrics(source.metrics),
    }


def _source_payload(source: Source, semantic_evidence: list[SemanticEvidenceExtraction] | None = None) -> dict[str, Any]:
    full_transcript = _full_transcript_text(source, limit=int(os.getenv("VOXLENS_LLM_SYNTHESIS_TRANSCRIPT_CHARS", "4000")))
    return {
        "id": source.id,
        "platform": source.platform,
        "sourceType": source.sourceType,
        "title": source.title,
        "creator": source.creator or source.author,
        "url": source.url,
        "summary": source.summary,
        "metrics": source.metrics,
        "evidenceScore": source.evidenceScore,
        "comments": [
            {"author": comment.author, "text": comment.text[:280], "likes": comment.likes}
            for comment in source.comments[:4]
            if comment.text
        ],
        "transcriptPreview": text_excerpt([source.transcriptPreview], 900),
        "fullTranscript": full_transcript,
        "transcriptChunks": _transcript_chunks(source),
        "transcriptSegments": _transcript_segment_payloads(source),
        "transcriptSegments": _transcript_segment_payloads(source),
        "semanticEvidence": [_semantic_evidence_payload(item) for item in semantic_evidence or []],
    }


def _source_evidence_pack(source: Source) -> dict[str, Any]:
    full_transcript = _full_transcript_text(source, limit=int(os.getenv("VOXLENS_LLM_SYNTHESIS_TRANSCRIPT_CHARS", "4000")))
    return {
        "id": source.id,
        "platform": source.platform,
        "sourceType": source.sourceType,
        "title": source.title[:240],
        "creator": (source.creator or source.author)[:120],
        "summary": text_excerpt([source.summary], 360),
        "comments": [
            {"author": comment.author, "text": comment.text[:320], "likes": comment.likes}
            for comment in source.comments[:8]
            if comment.text
        ],
        "fullTranscript": full_transcript,
        "transcriptChunks": _transcript_chunks(source),
        "metrics": _compact_metrics(source.metrics),
    }


def _transcript_chunks(source: Source) -> list[dict[str, Any]]:
    text = _full_transcript_text(source)
    if not text:
        return []
    chunk_size = max(1, int(os.getenv("VOXLENS_LLM_TRANSCRIPT_CHUNK_CHARS", "1600")))
    max_chars = max(chunk_size, int(os.getenv("VOXLENS_LLM_MAX_TRANSCRIPT_CHARS_PER_SOURCE", "12000")))
    limited = text[:max_chars]
    chunks = []
    for idx in range(0, len(limited), chunk_size):
        chunks.append({"index": len(chunks), "text": limited[idx : idx + chunk_size]})
    return chunks[: max(1, int(os.getenv("VOXLENS_LLM_MAX_TRANSCRIPT_CHUNKS_PER_SOURCE", "8")))]


def _full_transcript_text(source: Source, limit: int | None = None) -> str:
    value = max(
        [getattr(source, "fullTranscript", ""), getattr(source, "transcriptText", ""), source.transcriptPreview],
        key=len,
    )
    max_chars = limit if limit is not None else int(os.getenv("VOXLENS_LLM_MAX_TRANSCRIPT_CHARS_PER_SOURCE", "12000"))
    return text_excerpt([value], max_chars)


def _transcript_segment_payloads(source: Source) -> list[dict[str, Any]]:
    """Build timestamped transcript segment payloads for citation anchoring."""
    segments = source.transcriptSegments
    if not segments:
        return []
    payloads: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments[:30]):
        if not seg.text or not seg.text.strip():
            continue
        payloads.append({
            "index": idx,
            "text": seg.text.strip()[:600],
            "start": seg.start,
            "end": seg.end,
        })
    return payloads


def _semantic_evidence_by_source(items: list[SemanticEvidenceExtraction]) -> dict[int, list[SemanticEvidenceExtraction]]:
    grouped: dict[int, list[SemanticEvidenceExtraction]] = {}
    for item in items:
        grouped.setdefault(item.sourceId, []).append(item)
    return grouped


def _semantic_evidence_payload(item: SemanticEvidenceExtraction) -> dict[str, Any]:
    return {
        "candidate": item.candidate,
        "dimension": item.dimension,
        "conclusion": item.conclusion,
        "scenario": item.scenario,
        "value": item.value,
        "quote": item.quote,
        "sourceId": item.sourceId,
        "confidence": item.confidence,
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in metrics.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (str, int, float, bool)):
            compact[key] = str(value)[:120] if isinstance(value, str) else value
        if len(compact) >= 6:
            break
    return compact
