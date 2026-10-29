from __future__ import annotations

import re
import time
from dataclasses import dataclass

from app.models import AgentStep, CrawlTarget, ResearchPlan, ResearchRequest
from app.platform_catalog import CRAWLER_PLATFORM_IDS

YOUTUBE_MIN_LIMIT = 20
YOUTUBE_MIN_DETAIL_LIMIT = 6
_BRANDS = (
    "Apple", "iPhone", "Huawei", "华为", "小米", "Xiaomi", "Redmi", "荣耀", "Honor",
    "OPPO", "vivo", "Samsung", "三星", "Sony", "索尼", "DJI", "大疆", "Bose", "戴森",
)
_CATEGORIES = (
    "手机", "耳机", "相机", "汽车", "护肤", "美妆", "家电", "电脑", "平板", "软件", "应用",
    "phone", "headphone", "camera", "car", "skincare", "laptop", "tablet", "app", "software",
)
_EVIDENCE_PATTERNS = {
    "review": ("评测", "测评", "review", "实测", "体验"),
    "complaint": ("吐槽", "避坑", "缺点", "问题", "投诉", "complaint"),
    "recommendation": ("推荐", "值得买", "怎么选", "recommend", "best"),
    "comparison": ("对比", "横评", "vs", "versus", "比较"),
}
_PLATFORM_SUFFIXES = {
    "bilibili": "深度评测 实测 参数",
    "douyin": "真实体验 吐槽 避坑",
    "youtube": "review comparison long term experience",
    "xiaohongshu": "真实使用 推荐 避坑",
    "zhihu": "怎么样 对比 原因",
    "kuaishou": "真实体验 口碑",
    "weibo": "口碑 讨论 吐槽",
}


@dataclass(frozen=True)
class ResearchQuestionAnalysis:
    """Describe the structured entities, scope, comparisons, and evidence intent in a question."""

    canonical_query: str
    entities: tuple[str, ...]
    product_names: tuple[str, ...]
    brand_names: tuple[str, ...]
    categories: tuple[str, ...]
    time_range: str
    comparison_objects: tuple[str, ...]
    evidence_types: tuple[str, ...]


def build_research_plan(request: ResearchRequest) -> tuple[ResearchPlan, AgentStep]:
    """Build platform-specific crawl targets from a structured research-question analysis."""

    started = time.perf_counter()
    query = (request.query or request.need).strip()
    analysis = analyze_research_question(query, request.lang)
    expanded = _expand_query(query, request.lang, analysis)
    targets: list[CrawlTarget] = []

    for platform in request.platforms:
        primary = _primary_provider(platform, request.useCrawlerRuntime)
        targets.append(CrawlTarget(
            platform=platform,
            provider=primary,
            query=_platform_query(platform, analysis),
            limit=_target_limit(platform, request.limitPerPlatform),
            commentsLimit=request.commentsPerVideo,
            detailLimit=_target_detail_limit(platform, request.detailVideosPerPlatform),
            videoParallelism=request.maxParallelVideos,
            authMode=request.authMode,
            role="primary",
        ))
        # Only schedule a cross-provider fallback when the platform actually has
        # a second implemented provider (registry.py); otherwise skip to avoid
        # "unsupported local target" noise.
        if platform in {"bilibili"} and request.useCrawlerRuntime:
            targets.append(CrawlTarget(
                platform=platform,
                provider="opencli" if primary == "crawler" else "crawler",
                query=_platform_query(platform, analysis),
                limit=request.limitPerPlatform,
                commentsLimit=request.commentsPerVideo,
                detailLimit=request.detailVideosPerPlatform,
                videoParallelism=request.maxParallelVideos,
                authMode=request.authMode,
                role="fallback",
            ))

    analysis_note = (
        f"Structured question: entities={list(analysis.entities)}, time_range={analysis.time_range or 'unspecified'}, "
        f"comparisons={list(analysis.comparison_objects)}, evidence_types={list(analysis.evidence_types)}."
    )
    plan = ResearchPlan(
        canonicalQuery=query,
        expandedQueries=expanded,
        platforms=request.platforms,
        targets=targets,
        retrievalDepth="deep" if request.limitPerPlatform >= 10 or request.commentsPerVideo >= 10 else "standard",
        notes=[
            analysis_note,
            "The alpha searches Bilibili, Douyin, YouTube, Xiaohongshu, Zhihu, Kuaishou and Weibo by default; Baidu/Tieba is removed from the current crawler runtime scope.",
            "VoxLens crawler runtime is primary for Chinese social platforms and uses cookie env vars or existing-browser/CDP login state.",
            "Crawler-style access can trigger platform rate limits, verification, account restrictions or bans when misused.",
            "OpenCLI is kept as a local development fallback for YouTube and Bilibili metadata enrichment.",
            "providerMode separates local/dev providers from production online provider adapters.",
            "The frontend consumes progress events for planning, provider runs, source batches, evidence scoring and report streaming.",
        ],
    )
    step = AgentStep(
        name="QueryPlannerAgent",
        role="Decomposes entities, time scope, comparisons and evidence intent into platform-specific targets.",
        status="ok",
        message=f"Planned {len(targets)} crawl targets for {len(request.platforms)} platforms.",
        elapsedSec=round(time.perf_counter() - started, 3),
        metrics={
            "expandedQueries": len(expanded),
            "targets": len(targets),
            "entities": len(analysis.entities),
            "comparisonObjects": len(analysis.comparison_objects),
            "evidenceTypes": list(analysis.evidence_types),
            "timeRange": analysis.time_range,
        },
    )
    return plan, step


def analyze_research_question(query: str, lang: str) -> ResearchQuestionAnalysis:
    """Extract lightweight structured planning signals without invoking an external model."""

    clean = re.sub(r"\s+", " ", query).strip()
    brands = tuple(brand for brand in _BRANDS if re.search(re.escape(brand), clean, re.IGNORECASE))
    categories = tuple(category for category in _CATEGORIES if re.search(re.escape(category), clean, re.IGNORECASE))
    comparisons = _comparison_objects(clean)
    products = _product_names(clean, brands, comparisons)
    entities = _dedupe((*products, *brands, *categories, *comparisons))
    evidence_types = tuple(
        evidence_type
        for evidence_type, patterns in _EVIDENCE_PATTERNS.items()
        if any(re.search(rf"(?i)(?:^|\W){re.escape(pattern)}(?:$|\W)", clean) if pattern.isascii() else pattern in clean for pattern in patterns)
    )
    if not evidence_types:
        evidence_types = ("review", "complaint", "recommendation", "comparison")
    return ResearchQuestionAnalysis(
        canonical_query=clean,
        entities=entities,
        product_names=products,
        brand_names=brands,
        categories=categories,
        time_range=_time_range(clean, lang),
        comparison_objects=comparisons,
        evidence_types=evidence_types,
    )


def _primary_provider(platform: str, use_crawler_runtime: bool) -> str:
    """Choose the existing primary provider without changing runtime behavior."""

    if use_crawler_runtime and platform in CRAWLER_PLATFORM_IDS:
        return "crawler"
    return "opencli"


def _target_limit(platform: str, requested_limit: int) -> int:
    """Preserve the deeper YouTube search minimum."""

    if platform == "youtube":
        return max(requested_limit, YOUTUBE_MIN_LIMIT)
    return requested_limit


def _target_detail_limit(platform: str, requested_limit: int) -> int:
    """Preserve the deeper YouTube detail minimum."""

    if platform == "youtube":
        return max(requested_limit, YOUTUBE_MIN_DETAIL_LIMIT)
    return requested_limit


def _platform_query(platform: str, analysis: ResearchQuestionAnalysis) -> str:
    """Generate a query tuned to each platform's common evidence vocabulary."""

    base_parts = list(analysis.comparison_objects or analysis.product_names or analysis.entities)
    base = " ".join(base_parts) if base_parts else analysis.canonical_query
    time_range = analysis.time_range
    suffix = _PLATFORM_SUFFIXES.get(platform, "评测 对比 体验")
    if platform == "youtube" and not re.search(r"[A-Za-z]", base):
        suffix = "review comparison long term experience"
    return " ".join(part for part in (base, time_range, suffix) if part).strip()


def _expand_query(
    query: str,
    lang: str,
    analysis: ResearchQuestionAnalysis | None = None,
) -> list[str]:
    """Expand a canonical query using its inferred evidence types and time scope."""

    question = analysis or analyze_research_question(query, lang)
    clean = question.canonical_query
    values = [clean]
    evidence_terms = {
        "review": "真实体验 评测" if lang == "zh" else "review real experience",
        "complaint": "吐槽 避坑 问题" if lang == "zh" else "complaints problems",
        "recommendation": "推荐 怎么选" if lang == "zh" else "recommendation best choice",
        "comparison": "对比 优缺点" if lang == "zh" else "comparison pros cons",
    }
    for evidence_type in question.evidence_types:
        values.append(" ".join(part for part in (clean, question.time_range, evidence_terms[evidence_type]) if part))
    if lang == "zh" and not re.search(r"[A-Za-z]", clean):
        values.append(f"{clean} review comparison long term")
    return list(_dedupe(values))[:6]


def _comparison_objects(query: str) -> tuple[str, ...]:
    """Extract explicit alternatives surrounding comparison connectors."""

    scoped = re.split(r"近\s*\d+|最近|过去|20\d{2}|评测|测评|吐槽|推荐|怎么样", query, maxsplit=1)[0]
    if not re.search(r"(?i)\bvs\.?\b|对比|横评|还是|和|与", scoped):
        return ()
    parts = re.split(r"(?i)\s*(?:\bvs\.?\b|versus|对比|横评|还是|和|与)\s*", scoped)
    cleaned = tuple(part.strip(" ，,、:：") for part in parts if part.strip(" ，,、:："))
    return cleaned if len(cleaned) >= 2 else ()


def _product_names(
    query: str,
    brands: tuple[str, ...],
    comparisons: tuple[str, ...],
) -> tuple[str, ...]:
    """Extract product-like names from explicit comparisons, quotes, brands and numbered tokens."""

    candidates: list[str] = list(comparisons)
    candidates.extend(re.findall(r"[“\"']([^”\"']{2,40})[”\"']", query))
    candidates.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9+.-]*(?:\s+[A-Za-z0-9+.-]+){0,2}\b", query))
    candidates.extend(brand for brand in brands if brand not in candidates)
    filtered = []
    stop_words = {"vs", "review", "comparison", "best", "long term"}
    for candidate in candidates:
        clean = candidate.strip()
        if clean and clean.lower() not in stop_words and (any(character.isdigit() for character in clean) or clean in brands or clean in comparisons):
            filtered.append(clean)
    return _dedupe(filtered)


def _time_range(query: str, lang: str) -> str:
    """Extract an explicit calendar or relative research window phrase."""

    patterns = (
        r"20\d{2}(?:\s*[-至到]\s*20\d{2})?(?:年)?",
        r"(?:最近|近|过去)\s*\d+\s*(?:天|周|个月|月|年)",
        r"(?:last|past)\s+\d+\s+(?:days?|weeks?|months?|years?)",
        r"since\s+20\d{2}",
    )
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()
    return ""


def _dedupe(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return non-empty values in stable first-seen order."""

    deduped: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in deduped:
            deduped.append(clean)
    return tuple(deduped)
