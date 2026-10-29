import { createFileRoute, Link } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import gsap from "gsap";
import { toPng } from "html-to-image";
import { Download, RefreshCw, Sparkles } from "lucide-react";
import { z } from "zod";
import { BrandMark } from "@/components/BrandMark";
import { LangToggle } from "@/components/LangToggle";
import { WebSearchIcon } from "@/components/WebSearchIcon";
import {
  createResearchRun,
  getDemoReport,
  getResearchRun,
  runResearch,
  streamResearchRun,
  type PlatformId,
  type ResearchMode,
  type ResearchReport,
  type ResearchStreamEvent,
} from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { DEFAULT_RESEARCH_PLATFORMS } from "@/lib/platforms";
import {
  CitationList,
  ReportExportSheet,
  InsightStrip,
  PlatformLogoStrip,
  ProgressPanel,
  ReportSectionContent,
  SourceItem,
} from "@/features/results/results-view";
import { imageFileName } from "@/features/results/report-format";
import {
  appendSectionDelta,
  createDraftReport,
  createDraftUi,
  mergeSourcesIntoReport,
  runCreationCache,
  upsertSection,
} from "@/features/results/report-state";

const searchSchema = z.object({
  q: z.preprocess((value) => (value == null ? undefined : String(value)), z.string().optional()),
  live: z.preprocess((value) => (value == null ? undefined : String(value)), z.string().optional()),
  run: z.preprocess((value) => (value == null ? undefined : String(value)), z.string().optional()),
  mode: z
    .preprocess(
      (value) => (value == null ? undefined : String(value)),
      z.enum(["consumer", "business"]).optional(),
    )
    .catch(undefined),
});

export const Route = createFileRoute("/results")({
  validateSearch: (s) => searchSchema.parse(s),
  component: Results,
  head: () => ({
    meta: [
      { title: "VoxLens Studio Research - Evidence, not answers." },
      { name: "description", content: "Auditable, citable social-video evidence research results." },
    ],
  }),
});

function Results() {
  const { t, lang } = useI18n();
  const { q, live, run, mode } = Route.useSearch();
  const containerRef = useRef<HTMLDivElement>(null);
  const middleScrollRef = useRef<HTMLElement>(null);
  const sourcesListRef = useRef<HTMLDivElement>(null);
  const exportRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [activeFilter, setActiveFilter] = useState<"all" | PlatformId>("all");
  const [activeOutline, setActiveOutline] = useState<number>(0);
  const [highlightedSourceId, setHighlightedSourceId] = useState<string | null>(null);
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [progress, setProgress] = useState(0);
  const [streamEvents, setStreamEvents] = useState<ResearchStreamEvent[]>([]);
  const [exportingImage, setExportingImage] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pipelineExpanded, setPipelineExpanded] = useState(true);

  const handleStreamEvent = useCallback(
    (event: ResearchStreamEvent) => {
      setStreamEvents((items) => [...items.slice(-17), event]);
      if (typeof event.progress === "number") setProgress(event.progress);

      if (event.type === "run_started") {
        setReport(
          createDraftReport(event.query || q || "VoxLens", lang, event.runId, event.stages),
        );
        return;
      }

      if (event.type === "stage" && Array.isArray(event.stages)) {
        setReport((current) =>
          current
            ? {
                ...current,
                status: "running",
                ui: {
                  ...(current.ui ?? createDraftUi()),
                  stages: event.stages,
                  progress: event.progress,
                  activeStage: event.stageId,
                },
              }
            : current,
        );
        return;
      }

      if (event.type === "sources") {
        setReport((current) =>
          current ? mergeSourcesIntoReport(current, event.sources || []) : current,
        );
        return;
      }

      if (event.type === "evidence") {
        setReport((current) =>
          current ? mergeSourcesIntoReport(current, event.sources || [], true) : current,
        );
        return;
      }

      if (event.type === "outline") {
        setReport((current) =>
          current
            ? {
                ...current,
                outline: event.outline || current.outline,
                takeaways: event.takeaways || current.takeaways,
                insights: event.insights || current.insights,
                coverage: event.coverage || current.coverage,
                quality: event.quality || current.quality,
                warnings: event.warnings || current.warnings,
              }
            : current,
        );
        return;
      }

      if (event.type === "report_patch") {
        setReport((current) => ({
          ...event.report,
          sources: event.report.sources?.length ? event.report.sources : (current?.sources ?? []),
          sections: current?.sections?.length ? current.sections : event.report.sections,
        }));
        return;
      }

      if (event.type === "section_started") {
        setReport((current) => (current ? upsertSection(current, event.section) : current));
        return;
      }

      if (event.type === "section_delta") {
        setReport((current) =>
          current ? appendSectionDelta(current, event.sectionId, event.delta) : current,
        );
        return;
      }

      if (event.type === "section_complete") {
        setReport((current) => (current ? upsertSection(current, event.section) : current));
        return;
      }

      if (event.type === "final_report") {
        setReport(event.report);
        setProgress(100);
        setStreaming(false);
        setLoading(false);
        return;
      }

      if (event.type === "error") {
        setError(event.message || "Stream failed");
        setStreaming(false);
        setLoading(false);
      }
    },
    [lang, q],
  );

  const loadReport = useCallback(
    async (forceLive = false) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      setStreaming(Boolean(q));
      setProgress(0);
      setStreamEvents([]);
      setError(null);
      const selectedResearchMode: ResearchMode = mode ?? "auto";
      try {
        if (run) {
          const existing = await getResearchRun(run);
          if (existing.report) {
            setReport(existing.report);
            setProgress(existing.progress || 100);
          } else {
            setReport(
              createDraftReport(
                existing.request.query || existing.request.need || "VoxLens",
                existing.request.lang ?? lang,
                existing.runId,
              ),
            );
            setProgress(existing.progress || 0);
          }
          await streamResearchRun(run, handleStreamEvent, controller.signal);
          return;
        }

        if (!q) {
          const data = await getDemoReport(lang);
          setReport(data);
          setProgress(100);
          return;
        }

        const liveFlag = (live ?? "").replace(/^"+|"+$/g, "").toLowerCase();
        const shouldUseCrawlerRuntime = liveFlag !== "0" && liveFlag !== "false";
        const payload = {
          need: q,
          query: q,
          researchMode: selectedResearchMode,
          platforms: DEFAULT_RESEARCH_PLATFORMS,
          limitPerPlatform: forceLive ? 20 : 12,
          commentsPerVideo: forceLive ? 16 : 8,
          detailVideosPerPlatform: forceLive ? 4 : 2,
          maxParallelPlatforms: forceLive ? 4 : 3,
          maxParallelVideos: forceLive ? 4 : 3,
          useLiveProviders: true,
          useCrawlerRuntime: shouldUseCrawlerRuntime,
          providerMode: "local" as const,
          authMode: "auto" as const,
          includeTranscripts: true,
          minLiveSources: 3,
          lang,
        };
        setReport(createDraftReport(q, lang));
        const cacheKey = `${q}|${mode || "auto"}|${lang}|${liveFlag || "1"}|${forceLive ? "force" : "normal"}`;
        let pendingRun = runCreationCache.get(cacheKey);
        if (!pendingRun || forceLive) {
          pendingRun = createResearchRun(payload);
          runCreationCache.set(cacheKey, pendingRun);
        }
        const created = await pendingRun;
        if (typeof window !== "undefined") {
          const params = new URLSearchParams(window.location.search);
          params.set("q", q);
          params.set("run", created.runId);
          params.set("live", liveFlag || "1");
          if (mode) params.set("mode", mode);
          else params.delete("mode");
          window.history.replaceState(null, "", `/results?${params.toString()}`);
        }
        await streamResearchRun(created.runId, handleStreamEvent, controller.signal);
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          try {
            const data = q
              ? await runResearch({
                  need: q,
                  query: q,
                  researchMode: selectedResearchMode,
                  platforms: DEFAULT_RESEARCH_PLATFORMS,
                  limitPerPlatform: 8,
                  commentsPerVideo: 6,
                  detailVideosPerPlatform: 1,
                  useLiveProviders: true,
                  useCrawlerRuntime: true,
                  providerMode: "local",
                  minLiveSources: 2,
                  lang,
                })
              : await getDemoReport(lang);
            setReport(data);
            setProgress(100);
          } catch (fallbackErr) {
            setError(fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr));
          }
        }
      } finally {
        setLoading(false);
        setStreaming(false);
      }
    },
    [handleStreamEvent, lang, live, mode, q, run],
  );

  useEffect(() => {
    void loadReport(false);
  }, [loadReport]);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    setActiveOutline(0);
    setHighlightedSourceId(null);
  }, [report]);

  useEffect(() => {
    const bodyOverflow = document.body.style.overflow;
    const htmlOverflow = document.documentElement.style.overflow;
    document.body.style.overflow = "hidden";
    document.documentElement.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = bodyOverflow;
      document.documentElement.style.overflow = htmlOverflow;
    };
  }, []);

  const introPlayedRef = useRef(false);
  useEffect(() => {
    if (!report || introPlayedRef.current) return;
    introPlayedRef.current = true;
    const ctx = gsap.context(() => {
      gsap.from("[data-rise]", {
        y: 18,
        opacity: 0,
        duration: 0.7,
        ease: "power3.out",
        stagger: 0.06,
      });
    }, containerRef);
    return () => ctx.revert();
  }, [report]);

  const filteredSources = useMemo(() => {
    if (!report) return [];
    return activeFilter === "all"
      ? report.sources
      : report.sources.filter((s) => s.platform === activeFilter);
  }, [activeFilter, report]);

  const title = report?.title || q || "VoxLens";
  const isWorking = loading || streaming || report?.status === "running";
  const visibleOutline = report?.outline ?? [];

  useEffect(() => {
    if (!highlightedSourceId) return;
    const sourceCard = containerRef.current?.querySelector<HTMLElement>(
      `[data-source-id="${highlightedSourceId}"]`,
    );
    sourceCard?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [highlightedSourceId, filteredSources]);

  const handleOutlineClick = (sectionId: string, index: number) => {
    setActiveOutline(index);
    const container = middleScrollRef.current;
    const target = document.getElementById(sectionId);
    if (!container || !target) return;
    const containerRect = container.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    container.scrollTo({
      top: container.scrollTop + targetRect.top - containerRect.top - 8,
      behavior: "smooth",
    });
  };

  const handleReportScroll = () => {
    const container = middleScrollRef.current;
    if (!container || !report?.sections.length) return;
    const threshold = container.getBoundingClientRect().top + 80;
    let next = 0;
    report.sections.forEach((section, index) => {
      const node = document.getElementById(section.id);
      if (node && node.getBoundingClientRect().top <= threshold) {
        next = index;
      }
    });
    setActiveOutline((current) => (current === next ? current : next));
  };

  const handleCitationHover = (sourceId: number | string | null) => {
    setHighlightedSourceId(sourceId == null ? null : String(sourceId));
  };

  const handleSourceFilterChange = (filter: "all" | PlatformId) => {
    setActiveFilter(filter);
    window.requestAnimationFrame(() => {
      sourcesListRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    });
  };

  const exportReportImage = async () => {
    if (!report || typeof window === "undefined" || !exportRef.current) return;
    setExportingImage(true);
    try {
      await document.fonts?.ready;
      const node = exportRef.current;
      const pixelRatio = node.offsetHeight > 9000 ? 1.15 : 1.55;
      const dataUrl = await toPng(node, {
        backgroundColor: "#fbf7ff",
        cacheBust: true,
        imagePlaceholder: "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==",
        pixelRatio,
        skipFonts: true,
      });
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = `${imageFileName(report.title || title)}.png`;
      link.click();
    } finally {
      setExportingImage(false);
    }
  };

  return (
    <div ref={containerRef} className="results-shell relative h-screen overflow-hidden">
      <header className="flex h-[72px] shrink-0 items-center justify-between px-6 py-5 sm:px-10">
        <div className="flex items-center gap-4">
          <BrandMark />
          <span className="h-5 w-px bg-border" />
          <span className="text-sm font-medium text-muted-foreground">{t("research_title")}</span>
        </div>
        <div className="flex items-center gap-2">
          <LangToggle />
          <button
            type="button"
            onClick={() => void loadReport(true)}
            disabled={isWorking}
            className="glass inline-flex items-center gap-1.5 rounded-full px-3.5 py-2 text-xs font-medium transition hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${isWorking ? "animate-spin" : ""}`}
              style={{ color: "var(--violet)" }}
            />
            {t("re_research")}
          </button>
          <button
            type="button"
            onClick={() => void exportReportImage()}
            disabled={!report || exportingImage}
            aria-label="download report image"
            className="glass grid h-9 w-9 place-items-center rounded-full transition hover:-translate-y-0.5 disabled:cursor-wait disabled:opacity-60"
          >
            <Download className={`h-4 w-4 ${exportingImage ? "animate-bounce" : ""}`} />
          </button>
        </div>
      </header>

      {loading && !report && (
        <div className="mx-auto mt-20 max-w-md px-6 text-center">
          <div className="glass rounded-3xl p-8" style={{ boxShadow: "var(--shadow-glass)" }}>
            <Sparkles
              className="mx-auto h-8 w-8 animate-pulse"
              style={{ color: "var(--violet)" }}
            />
            <p className="mt-4 text-sm text-muted-foreground">{t("loading")}</p>
          </div>
        </div>
      )}

      {!report && error && (
        <div className="mx-auto mt-20 max-w-md px-6 text-center">
          <div className="glass rounded-3xl p-8" style={{ boxShadow: "var(--shadow-glass)" }}>
            <h1 className="text-lg font-semibold">{t("error_title")}</h1>
            <p className="mt-3 text-xs text-muted-foreground">{error}</p>
            <button
              type="button"
              onClick={() => void loadReport(false)}
              className="mt-5 rounded-full bg-primary px-5 py-2 text-sm text-primary-foreground"
            >
              {t("retry")}
            </button>
          </div>
        </div>
      )}

      {report && (
        <div className="mx-auto grid h-[calc(100vh-4.5rem)] w-full max-w-[1400px] gap-6 overflow-hidden px-6 sm:px-10 lg:grid-cols-[260px_minmax(0,1fr)_340px]">
          <aside data-rise className="space-y-6 lg:self-start">
            <div>
              <div className="mb-3 inline-flex rounded-full bg-white/60 px-3 py-1 text-[11px] font-medium text-muted-foreground ring-1 ring-black/5">
                {isWorking
                  ? `${t("streaming_badge")} ${progress}%`
                  : report.isDemoFallback
                    ? t("demo_badge")
                    : report.status === "partial"
                      ? t("partial_badge")
                      : t("live_badge")}
              </div>
              <h1 className="font-display text-3xl font-normal italic leading-[1.05] tracking-tight sm:text-4xl">
                {title}
              </h1>
            </div>

            <div className="glass rounded-2xl p-4" style={{ boxShadow: "var(--shadow-glass)" }}>
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                {t("outline")}
              </p>
              <ul className="mt-3 space-y-1">
                {visibleOutline.map((item, i) => {
                  const active = activeOutline === i;
                  return (
                    <li key={item.id}>
                      <button
                        type="button"
                        onClick={() => handleOutlineClick(item.id, i)}
                        className={`group flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left text-sm transition ${
                          active
                            ? "bg-white/70 font-medium text-foreground"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        <span
                          className={`h-4 w-[2px] rounded-full transition ${active ? "" : "bg-transparent"}`}
                          style={
                            active
                              ? {
                                  background:
                                    "linear-gradient(180deg, var(--violet), var(--indigo))",
                                }
                              : undefined
                          }
                        />
                        <span className="text-[11px] tabular-nums text-muted-foreground/70">
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <span className="truncate">{item.label}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>

            <div className="px-1 text-xs text-muted-foreground">
              <span className="font-medium text-foreground">{report.totalVideos}</span>{" "}
              {t("videos")} /{" "}
              <span className="font-medium text-foreground">
                {report.platforms.filter((p) => p.count > 0).length}
              </span>{" "}
              {t("platforms")} /{" "}
              <span className="font-medium text-foreground">{report.totalComments}</span>{" "}
              {t("comments")}
            </div>
          </aside>

          <section
            ref={middleScrollRef}
            data-rise
            data-lenis-prevent
            onScroll={handleReportScroll}
            className="scroll-col space-y-6 lg:self-start"
          >
            <div className="glass rounded-2xl p-5" style={{ boxShadow: "var(--shadow-glass)" }}>
              <ProgressPanel
                report={report}
                events={streamEvents}
                streaming={isWorking}
                progress={progress}
                lang={lang}
                expanded={pipelineExpanded}
                onExpandedChange={setPipelineExpanded}
              />

              <InsightStrip report={report} lang={lang} onCitationHover={handleCitationHover} />

              <div className="mt-6 border-t border-white/40 pt-5">
                <h3 className="text-sm font-semibold">{t("key_takeaways")}</h3>
                <ul className="mt-3 space-y-2 text-sm text-foreground/85">
                  {report.takeaways.map((item, i) => (
                    <li key={`${item.text}-${i}`} className="flex gap-2">
                      <span
                        className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full"
                        style={{
                          background: "linear-gradient(135deg, var(--violet), var(--indigo))",
                        }}
                      />
                      <span>
                        {item.text}{" "}
                        <CitationList ids={item.citations} onHover={handleCitationHover} />
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <article className="space-y-7">
              {report.sections.length === 0 && (
                <div
                  className="glass rounded-2xl p-6 text-sm text-muted-foreground"
                  style={{ boxShadow: "var(--shadow-glass)" }}
                >
                  {isWorking ? t("streaming_report") : t("empty_report")}
                </div>
              )}
              {report.sections.map((section) => (
                <section key={section.id} id={section.id} className="space-y-4">
                  <h2 className="text-xl font-bold tracking-tight">{section.title}</h2>
                  <ReportSectionContent
                    section={section}
                    sources={report.sources}
                    onCitationHover={handleCitationHover}
                  />
                </section>
              ))}
            </article>
          </section>

          <aside data-rise data-lenis-prevent className="sources-panel lg:self-start">
            <div className="sources-panel-head">
              <div className="flex items-center justify-between px-1">
                <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                  {t("sources")} / {report.sources.length}
                </p>
                <span className="rounded-full bg-white/65 px-2.5 py-1 text-[10px] font-medium text-muted-foreground ring-1 ring-white/70">
                  {filteredSources.length}
                </span>
              </div>

              <PlatformLogoStrip
                platforms={report.platforms}
                activeFilter={activeFilter}
                onSelect={handleSourceFilterChange}
                allLabel={t("all")}
                videoLabel={t("videos")}
              />
            </div>

            <div
              ref={sourcesListRef}
              className="sources-list glass rounded-2xl p-3"
              style={{ boxShadow: "var(--shadow-glass)" }}
            >
              {filteredSources.length === 0 ? (
                <p className="p-4 text-sm text-muted-foreground">{t("empty")}</p>
              ) : (
                <ul className="space-y-3">
                  {filteredSources.map((source) => (
                    <SourceItem
                      key={`${source.platform}-${source.id}-${source.url}`}
                      source={source}
                      highlighted={highlightedSourceId === String(source.id)}
                    />
                  ))}
                </ul>
              )}
            </div>
          </aside>
        </div>
      )}

      <div className="results-followup pointer-events-none">
        <form
          onSubmit={(e) => e.preventDefault()}
          className="followup-float search-shell glass pointer-events-auto flex w-full items-center gap-3 rounded-full px-3 py-2.5 pl-5"
        >
          <span
            className="inline-flex items-center rounded-full px-3 py-1.5 text-xs font-medium text-white"
            style={{ background: "linear-gradient(135deg, var(--violet), var(--indigo))" }}
          >
            {t("follow_up")}
          </span>
          <input
            placeholder={t("follow_up_ph")}
            className="min-w-0 flex-1 bg-transparent text-sm placeholder:text-muted-foreground/80 focus:outline-none"
          />
          <button
            type="submit"
            aria-label="web search"
            className="search-submit search-icon-button grid h-9 shrink-0 place-items-center"
          >
            <WebSearchIcon className="h-5 w-5" />
          </button>
        </form>
      </div>

      <div className="sr-only">{lang}</div>
      {report && (
        <div className="fixed left-[-10000px] top-0 w-[1120px] overflow-visible" aria-hidden="true">
          <div ref={exportRef}>
            <ReportExportSheet report={report} title={title} />
          </div>
        </div>
      )}
      <Link to="/" className="hidden">
        home
      </Link>
    </div>
  );
}
