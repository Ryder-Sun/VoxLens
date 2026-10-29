from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.platform_catalog import DEFAULT_PLATFORM_IDS


PlatformId = Literal["bilibili", "douyin", "youtube", "xiaohongshu", "zhihu", "kuaishou", "weibo"]
Lang = Literal["zh", "en"]
AuthMode = Literal["auto", "existing_browser", "cookie", "qrcode"]
AgentStatus = Literal["ok", "partial", "failed", "skipped"]
ReportStatus = Literal["queued", "running", "completed", "partial", "failed"]
StageStatus = Literal["queued", "running", "completed", "partial", "failed", "skipped"]
RunStatus = Literal["queued", "running", "completed", "partial", "failed", "cancelled"]
ProviderMode = Literal["local", "online", "hybrid"]
ResearchMode = Literal["auto", "consumer", "business"]
EvidenceModality = Literal["speech", "subtitle", "comment", "text", "ocr", "metadata"]
ClaimRelation = Literal["support", "contradict", "insufficient"]
EvidenceQualityLabel = Literal["Strong", "Moderate", "Weak", "Insufficient"]


class ResearchRequest(BaseModel):
    need: str = Field(..., min_length=1)
    query: str | None = None
    researchMode: ResearchMode = "auto"
    platforms: list[PlatformId] = Field(default_factory=lambda: list(DEFAULT_PLATFORM_IDS))
    limitPerPlatform: int = Field(default=12, ge=1, le=50)
    commentsPerVideo: int = Field(default=12, ge=0, le=50)
    detailVideosPerPlatform: int = Field(default=3, ge=0, le=8)
    useLiveProviders: bool = True
    useCrawlerRuntime: bool = True
    providerMode: ProviderMode = "local"
    authMode: AuthMode = "auto"
    includeTranscripts: bool = True
    crawlMedia: bool = False
    maxParallelPlatforms: int = Field(default=3, ge=1, le=5)
    maxParallelVideos: int = Field(default=3, ge=1, le=8)
    minLiveSources: int | None = Field(default=None, ge=1, le=30)
    lang: Lang = "zh"


class Comment(BaseModel):
    author: str = ""
    text: str = ""
    likes: str | int | None = None
    time: str = ""


class TranscriptSegment(BaseModel):
    text: str = ""
    start: str | int | float | None = None
    end: str | int | float | None = None


class Source(BaseModel):
    id: int
    platform: PlatformId
    sourceType: str = "video"
    domain: str = ""
    title: str
    author: str = ""
    creator: str = ""
    url: str = ""
    thumbnail: str = ""
    duration: str = ""
    published: str = ""
    summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    comments: list[Comment] = Field(default_factory=list)
    transcriptPreview: str = ""
    fullTranscript: str = Field(default="", exclude=True)
    transcriptText: str = Field(default="", exclude=True)
    transcriptSegments: list[TranscriptSegment] = Field(default_factory=list, exclude=True)
    evidenceChannels: list[str] = Field(default_factory=list)
    quality: dict[str, Any] = Field(default_factory=dict)
    provider: str = ""
    relevanceScore: int = 0
    evidenceScore: int = 0
    citationCount: int = 0
    badges: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    whyRelevant: str = ""
    status: Literal["collected", "ranked", "cited", "weak"] = "collected"
    collectedAt: str = ""


class Artifact(BaseModel):
    """Represent one immutable capture or extraction version of a source."""

    model_config = ConfigDict(frozen=True)

    id: str
    source_id: int
    artifact_type: str = "raw"
    content_hash: str
    captured_at: str
    collector_version: str
    collector_name: str = ""
    storage_uri: str = ""
    mime_type: str = ""
    raw_data: Any = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    full_transcript: str = ""
    transcript_segments: tuple[TranscriptSegment, ...] = ()
    parent_artifact_id: str | None = None


class EvidenceUnit(BaseModel):
    """Represent the smallest independently reviewable and citable evidence span."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    modality: EvidenceModality
    source_id: int
    artifact_id: str
    normalized_text: str = ""
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    asr_confidence: float | None = Field(default=None, ge=0, le=1)
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    comment_id: str = ""
    parent_comment_id: str = ""
    frame_index: int | None = Field(default=None, ge=0)
    speaker: str = ""
    author: str = ""
    extraction_method: str = ""
    language: str = ""
    content_hash: str = ""
    review_status: Literal["unreviewed", "low_confidence", "verified", "rejected"] = "unreviewed"

    def citation_ref(self, platform: PlatformId, platform_object_id: str) -> str:
        """Return a stable source-, timestamp-, or comment-level citation reference."""

        platform_name = platform.upper()
        if self.comment_id:
            return f"[{platform_name}:{platform_object_id}/comment/{self.comment_id}]"
        if self.start_ms is not None:
            end_ms = self.end_ms if self.end_ms is not None else self.start_ms
            return f"[{platform_name}:{platform_object_id}@{_format_timestamp(self.start_ms)}-{_format_timestamp(end_ms)}]"
        return f"[{platform_name}:{platform_object_id}]"


class Claim(BaseModel):
    """Represent a research conclusion grounded in sources and evidence units."""

    model_config = ConfigDict(frozen=True)

    id: str
    claim_text: str
    claim_type: str
    confidence_label: EvidenceQualityLabel = "Insufficient"
    source_ids: tuple[int, ...] = ()
    evidence_unit_ids: tuple[str, ...] = ()
    research_run_id: str = ""
    review_status: Literal["unreviewed", "verified", "rejected"] = "unreviewed"


class ClaimEvidence(BaseModel):
    """Represent the assessed relationship between a claim and one evidence unit."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    evidence_unit_id: str
    relation: ClaimRelation
    relevance_score: float = Field(ge=0, le=1)
    directness_score: float | None = Field(default=None, ge=0, le=1)
    independence_cluster: str = ""
    freshness_score: float | None = Field(default=None, ge=0, le=1)
    credibility_score: float | None = Field(default=None, ge=0, le=1)
    rationale: str = ""


class EvidenceQualityAssessment(BaseModel):
    """Expose categorical evidence quality with dimension-level reasons."""

    model_config = ConfigDict(frozen=True)

    label: EvidenceQualityLabel
    dimensions: dict[str, Literal["strong", "moderate", "weak", "insufficient"]]
    reasons: list[str] = Field(default_factory=list)


class PlatformSummary(BaseModel):
    id: PlatformId
    name: str
    logo: str
    count: int
    status: Literal["ok", "partial", "failed", "demo"] = "ok"
    note: str = ""


class CitationRef(BaseModel):
    """Structured citation anchored to a specific evidence unit with timestamp/comment context."""

    sourceId: int
    evidenceUnitId: str = ""
    platform: str = ""
    platformObjectId: str = ""
    startMs: int | None = Field(default=None, ge=0)
    endMs: int | None = Field(default=None, ge=0)
    commentId: str = ""
    citationRef: str = ""

    def format_ref(self) -> str:
        """Return the human-readable citation reference string."""
        if self.citationRef:
            return self.citationRef
        platform_name = self.platform.upper()
        if self.commentId:
            return f"[{platform_name}:{self.platformObjectId}/comment/{self.commentId}]"
        if self.startMs is not None:
            end_ms = self.endMs if self.endMs is not None else self.startMs
            return f"[{platform_name}:{self.platformObjectId}@{_format_timestamp(self.startMs)}-{_format_timestamp(end_ms)}]"
        return f"[{platform_name}:{self.platformObjectId}]"


class CitationText(BaseModel):
    text: str
    citations: list[int] = Field(default_factory=list)
    citationRefs: list[CitationRef] = Field(default_factory=list)


class VideoQuote(BaseModel):
    sourceId: int
    quote: str
    author: str
    thumbnail: str
    duration: str = ""


class ComparisonDimension(BaseModel):
    key: str
    label: str
    score: int | None = Field(default=None, ge=1, le=5)
    summary: str = ""
    evidence: list[int] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ComparisonRow(BaseModel):
    name: str
    signal: str = ""
    support: int = 3
    risk: int = 3
    freshness: int = 3
    confidence: int = 3
    camera: int | None = None
    lowLight: int | None = None
    video: int | None = None
    battery: int | None = None
    price: str = ""
    dimensions: list[ComparisonDimension] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence: list[int] = Field(default_factory=list)


class DecisionCandidate(BaseModel):
    name: str
    matchReason: str = ""
    sourceIds: list[int] = Field(default_factory=list)
    citationRefs: list[CitationRef] = Field(default_factory=list)


class DimensionComparisonValue(BaseModel):
    candidate: str
    conclusion: str = ""
    condition: str = ""
    confidence: str = "gray"
    sourceIds: list[int] = Field(default_factory=list)
    citationRefs: list[CitationRef] = Field(default_factory=list)


class DimensionComparison(BaseModel):
    dimension: str
    values: list[DimensionComparisonValue] = Field(default_factory=list)
    isContradictory: bool = False
    contradictionReason: str = ""


class ReportSection(BaseModel):
    id: str
    title: str
    kind: str = "narrative"
    level: int = 1
    body: str = ""
    bullets: list[CitationText] = Field(default_factory=list)
    quote: VideoQuote | None = None
    table: list[ComparisonRow] = Field(default_factory=list)
    sourceIds: list[int] = Field(default_factory=list)
    citationRefs: list[CitationRef] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


class OutlineItem(BaseModel):
    id: str
    label: str
    title: str = ""
    summary: str = ""
    status: StageStatus = "queued"
    sourceIds: list[int] = Field(default_factory=list)
    citationRefs: list[CitationRef] = Field(default_factory=list)
    citationCount: int = 0


class ResearchStage(BaseModel):
    id: str
    label: str
    status: StageStatus = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    message: str = ""
    startedAt: str = ""
    completedAt: str = ""


class SourceGroup(BaseModel):
    id: PlatformId
    name: str
    count: int = 0
    citedCount: int = 0
    status: StageStatus = "queued"
    note: str = ""


class ReportUIState(BaseModel):
    activeStage: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    stages: list[ResearchStage] = Field(default_factory=list)
    sourceGroups: list[SourceGroup] = Field(default_factory=list)
    interactionHints: dict[str, str] = Field(default_factory=dict)


class RunLog(BaseModel):
    provider: str
    platform: str
    ok: bool
    count: int
    elapsedSec: float = 0
    note: str = ""
    query: str = ""


class CrawlTarget(BaseModel):
    platform: PlatformId
    provider: str
    query: str
    limit: int
    commentsLimit: int
    detailLimit: int
    videoParallelism: int = 1
    authMode: AuthMode = "auto"
    role: str = "primary"


class ResearchPlan(BaseModel):
    canonicalQuery: str
    expandedQueries: list[str] = Field(default_factory=list)
    platforms: list[PlatformId] = Field(default_factory=list)
    targets: list[CrawlTarget] = Field(default_factory=list)
    retrievalDepth: str = "standard"
    notes: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    name: str
    role: str
    status: AgentStatus
    message: str = ""
    elapsedSec: float = 0
    metrics: dict[str, Any] = Field(default_factory=dict)


class ReportInsight(BaseModel):
    id: str
    kind: Literal["answer", "consensus", "disagreement", "risk", "opportunity", "coverage"] = "consensus"
    label: str
    summary: str = ""
    confidence: int = Field(default=3, ge=1, le=5)
    sourceIds: list[int] = Field(default_factory=list)
    citationRefs: list[CitationRef] = Field(default_factory=list)


class CoverageSummary(BaseModel):
    totalSources: int = 0
    totalComments: int = 0
    platformCount: int = 0
    citedSources: int = 0
    withComments: int = 0
    withTranscripts: int = 0
    providerRuns: int = 0
    successfulProviderRuns: int = 0
    failedProviderRuns: int = 0


class QualityMetric(BaseModel):
    score: int = Field(default=0, ge=0, le=100)
    label: str = ""
    rationale: str = ""
    sourceIds: list[int] = Field(default_factory=list)
    citationRefs: list[CitationRef] = Field(default_factory=list)


class QualityEvaluation(BaseModel):
    coverage: QualityMetric = Field(default_factory=QualityMetric)
    citationAccuracy: QualityMetric = Field(default_factory=QualityMetric)
    evidenceStrength: QualityMetric = Field(default_factory=QualityMetric)
    conclusionRisk: QualityMetric = Field(default_factory=QualityMetric)
    overall: int = Field(default=0, ge=0, le=100)
    warnings: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    runId: str = ""
    productName: str = "VoxLens Studio"
    slogan: str
    title: str
    query: str
    need: str
    lang: Lang
    status: ReportStatus = "completed"
    confidence: Literal["high", "medium", "low", "insufficient"] = "medium"
    generatedAt: str
    totalVideos: int
    totalComments: int
    platforms: list[PlatformSummary]
    outline: list[OutlineItem]
    takeaways: list[CitationText]
    insights: list[ReportInsight] = Field(default_factory=list)
    coverage: CoverageSummary = Field(default_factory=CoverageSummary)
    quality: QualityEvaluation = Field(default_factory=QualityEvaluation)
    sections: list[ReportSection]
    sources: list[Source]
    warnings: list[str] = Field(default_factory=list)
    methodology: list[str] = Field(default_factory=list)
    ui: ReportUIState | None = None
    runLogs: list[RunLog] = Field(default_factory=list)
    plan: ResearchPlan | None = None
    agentTrace: list[AgentStep] = Field(default_factory=list)
    isDemoFallback: bool = False


class RunRecord(BaseModel):
    runId: str
    status: RunStatus = "queued"
    request: ResearchRequest
    createdAt: str
    updatedAt: str
    queuedAt: str = ""
    startedAt: str = ""
    completedAt: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    eventCount: int = 0
    report: ResearchReport | None = None
    error: str = ""


def _format_timestamp(milliseconds: int) -> str:
    """Format milliseconds as a compact citation timestamp."""

    total_seconds = max(0, milliseconds // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
