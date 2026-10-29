from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

from app.models import AuthMode, Comment, RunLog, Source
from app.platform_catalog import COOKIE_ENV_ALIASES, CRAWLER_PLATFORM_CONFIG
from app.utils import CRAWLER_ROOT, RUNTIME_ROOT, first_nonempty, read_jsonl, run_command, slugify, strip_html, text_excerpt

PlatformName = Literal["douyin", "bilibili", "xiaohongshu", "zhihu", "kuaishou", "weibo"]

RUNNER_SCRIPT = Path(__file__).with_name("crawler_runner.py")

THUMBS = {
    "douyin": [
        "https://images.unsplash.com/photo-1611162616475-46b635cb6868?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1611162618071-b39a2ec055fb?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=420&q=80",
    ],
    "bilibili": [
        "https://images.unsplash.com/photo-1536240478700-b869070f9279?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1551817958-d9d86fb29431?auto=format&fit=crop&w=420&q=80",
    ],
    "xiaohongshu": [
        "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?auto=format&fit=crop&w=420&q=80",
    ],
    "zhihu": [
        "https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?auto=format&fit=crop&w=420&q=80",
    ],
    "kuaishou": [
        "https://images.unsplash.com/photo-1522869635100-9f4c5e86aa37?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1492619375914-88005aa9e8fb?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=420&q=80",
    ],
    "weibo": [
        "https://images.unsplash.com/photo-1495020689067-958852a7765e?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1516321497487-e288fb19713f?auto=format&fit=crop&w=420&q=80",
        "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=420&q=80",
    ],
}


def search_douyin(
    query: str,
    limit: int,
    comments_limit: int,
    use_crawler_runtime: bool,
    auth_mode: AuthMode = "auto",
    video_parallelism: int = 1,
) -> tuple[list[Source], RunLog]:
    return _search_with_crawler("douyin", query, limit, comments_limit, use_crawler_runtime, auth_mode, video_parallelism)


def search_bilibili_crawler(
    query: str,
    limit: int,
    comments_limit: int,
    use_crawler_runtime: bool,
    auth_mode: AuthMode = "auto",
    video_parallelism: int = 1,
) -> tuple[list[Source], RunLog]:
    return _search_with_crawler("bilibili", query, limit, comments_limit, use_crawler_runtime, auth_mode, video_parallelism)


def search_xiaohongshu(
    query: str,
    limit: int,
    comments_limit: int,
    use_crawler_runtime: bool,
    auth_mode: AuthMode = "auto",
    video_parallelism: int = 1,
) -> tuple[list[Source], RunLog]:
    return _search_with_crawler("xiaohongshu", query, limit, comments_limit, use_crawler_runtime, auth_mode, video_parallelism)


def search_zhihu(
    query: str,
    limit: int,
    comments_limit: int,
    use_crawler_runtime: bool,
    auth_mode: AuthMode = "auto",
    video_parallelism: int = 1,
) -> tuple[list[Source], RunLog]:
    return _search_with_crawler("zhihu", query, limit, comments_limit, use_crawler_runtime, auth_mode, video_parallelism)


def search_kuaishou(
    query: str,
    limit: int,
    comments_limit: int,
    use_crawler_runtime: bool,
    auth_mode: AuthMode = "auto",
    video_parallelism: int = 1,
) -> tuple[list[Source], RunLog]:
    return _search_with_crawler("kuaishou", query, limit, comments_limit, use_crawler_runtime, auth_mode, video_parallelism)


def search_weibo(
    query: str,
    limit: int,
    comments_limit: int,
    use_crawler_runtime: bool,
    auth_mode: AuthMode = "auto",
    video_parallelism: int = 1,
) -> tuple[list[Source], RunLog]:
    return _search_with_crawler("weibo", query, limit, comments_limit, use_crawler_runtime, auth_mode, video_parallelism)


def _search_with_crawler(
    platform: PlatformName,
    query: str,
    limit: int,
    comments_limit: int,
    use_crawler_runtime: bool,
    auth_mode: AuthMode,
    video_parallelism: int,
) -> tuple[list[Source], RunLog]:
    if not use_crawler_runtime:
        return [], RunLog(provider="crawler", platform=platform, ok=False, count=0, note="disabled")
    if not (CRAWLER_ROOT / "main.py").exists():
        return [], RunLog(provider="crawler", platform=platform, ok=False, count=0, note="VoxLens crawler runtime not found")

    config = CRAWLER_PLATFORM_CONFIG[platform]
    stamp = str(int(time.time()))
    save_root = RUNTIME_ROOT / "runs" / "crawler" / f"{platform}-{slugify(query)}-{stamp}"
    login_type, cookie = _resolve_login(platform, auth_mode)
    if auth_mode == "cookie" and not cookie:
        return [], RunLog(provider="crawler", platform=platform, ok=False, count=0, note=f"{platform} cookie env var is not configured")

    cmd = [
        "uv", "run", "python", str(RUNNER_SCRIPT),
        "--platform", config["arg"],
        "--lt", login_type,
        "--type", "search",
        "--keywords", query,
        "--save_data_option", "jsonl",
        "--save_data_path", str(save_root),
        "--get_comment", "true" if comments_limit > 0 else "false",
        "--get_sub_comment", "false",
        "--max_comments_count_singlenotes", str(min(max(comments_limit, 1), 50)),
        "--max_concurrency_num", str(_safe_parallelism(video_parallelism)),
        "--headless", "false",
    ]
    if cookie:
        cmd.extend(["--cookies", cookie])

    try:
        max_notes = _max_notes_for_platform(platform, limit)
        env_extra = {
            "VOXLENS_MC_MAX_NOTES": str(max_notes),
            "VOXLENS_MC_SLEEP_SEC": os.getenv("VOXLENS_MC_SLEEP_SEC", "1"),
            **_local_browser_env(),
        }
        code, stdout, stderr, elapsed = run_command(cmd, cwd=CRAWLER_ROOT, timeout=420, env_extra=env_extra)
        content_rows, comment_rows = _read_crawler_jsonl(save_root, config["folder"], limit)
        comments = _group_comments(platform, comment_rows, comments_limit)
        sources = [
            _source_from_crawler_row(platform, row, comments, idx, comments_limit)
            for idx, row in enumerate(content_rows[:limit], start=1)
        ]
        note = "" if sources else (stderr or stdout or "no results")[:240]
        if not sources:
            hint = f"{platform}: no sources returned. Configure {COOKIE_ENV_ALIASES.get(platform, ('VOXLENS_' + platform.upper() + '_COOKIE',))[0]} or log in once in the managed browser window."
            note = (note + " | " if note else "") + hint
        if login_type == "qrcode" and auth_mode in {"auto", "existing_browser"}:
            note = (note + " " if note else "") + f"auth={auth_mode} uses a local Chrome/CDP persistent profile; if Chrome:9222 is already available, VoxLens attaches to it; maxNotes={max_notes}; parallelVideos={_safe_parallelism(video_parallelism)}"
        return sources, RunLog(provider="crawler", platform=platform, ok=bool(sources) or code == 0, count=len(sources), elapsedSec=elapsed, note=note[:260])
    except Exception as exc:  # noqa: BLE001
        return [], RunLog(provider="crawler", platform=platform, ok=False, count=0, note=str(exc)[:240])


def _resolve_login(platform: PlatformName, auth_mode: AuthMode) -> tuple[str, str]:
    cookie = _cookie_for_platform(platform)
    if cookie:
        return "cookie", cookie
    if auth_mode == "cookie":
        return "cookie", ""
    # VoxLens crawler runtime's config has CDP existing-browser mode enabled; qrcode remains a safe fallback.
    return "qrcode", ""


def _local_browser_env() -> dict[str, str]:
    """Default crawler subprocesses to real Chrome/CDP with persistent local login state."""
    return {
        "VOXLENS_MC_ENABLE_CDP": os.getenv("VOXLENS_MC_ENABLE_CDP", "1"),
        "VOXLENS_MC_CDP_CONNECT_EXISTING": os.getenv("VOXLENS_MC_CDP_CONNECT_EXISTING", "1"),
        # Attach to an existing logged-in Chrome when available, otherwise launch a
        # managed local browser with persistent login state instead of failing hard.
        "VOXLENS_MC_CDP_REQUIRE_EXISTING": os.getenv("VOXLENS_MC_CDP_REQUIRE_EXISTING", "0"),
        "VOXLENS_MC_SAVE_LOGIN_STATE": os.getenv("VOXLENS_MC_SAVE_LOGIN_STATE", "1"),
        "VOXLENS_MC_AUTO_CLOSE_BROWSER": os.getenv("VOXLENS_MC_AUTO_CLOSE_BROWSER", "0"),
        "VOXLENS_MC_HEADLESS": os.getenv("VOXLENS_MC_HEADLESS", "0"),
        "VOXLENS_MC_CDP_HEADLESS": os.getenv("VOXLENS_MC_CDP_HEADLESS", "0"),
        "VOXLENS_MC_BROWSER_LAUNCH_TIMEOUT": os.getenv("VOXLENS_MC_BROWSER_LAUNCH_TIMEOUT", "60"),
    }


def _cookie_for_platform(platform: PlatformName) -> str:
    for name in COOKIE_ENV_ALIASES.get(platform, ()):
        value = os.getenv(name)
        if value:
            return value
    return ""


def _max_notes_for_platform(platform: PlatformName, limit: int) -> int:
    page_size = 10 if platform == "douyin" else 20
    return max(page_size, min(max(limit, 1), 50))


def _safe_parallelism(value: int) -> int:
    return max(1, min(value, 4))


def _read_crawler_jsonl(save_root: Path, platform_folder: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    content_dir = save_root / platform_folder / "jsonl"
    content_rows: list[dict[str, Any]] = []
    comment_rows: list[dict[str, Any]] = []
    for file in sorted(content_dir.glob("search_contents_*.jsonl"), reverse=True):
        content_rows.extend(read_jsonl(file, limit=limit - len(content_rows)))
        if len(content_rows) >= limit:
            break
    for file in sorted(content_dir.glob("search_comments_*.jsonl"), reverse=True):
        comment_rows.extend(read_jsonl(file))
    return content_rows, comment_rows


def _group_comments(platform: PlatformName, rows: list[dict[str, Any]], comments_limit: int) -> dict[str, list[Comment]]:
    grouped: dict[str, list[Comment]] = {}
    for row in rows:
        if platform == "douyin":
            item_id = first_nonempty(row.get("aweme_id"))
        elif platform == "bilibili":
            item_id = first_nonempty(row.get("video_id"), row.get("aid"))
        elif platform == "xiaohongshu":
            item_id = first_nonempty(row.get("note_id"))
        elif platform == "zhihu":
            item_id = first_nonempty(row.get("content_id"))
        elif platform == "kuaishou":
            item_id = first_nonempty(row.get("video_id"), row.get("photo_id"))
        else:
            item_id = first_nonempty(row.get("note_id"), row.get("mblogid"), row.get("weibo_id"))
        if not item_id:
            continue
        grouped.setdefault(item_id, [])
        if len(grouped[item_id]) >= comments_limit:
            continue
        grouped[item_id].append(Comment(
            author=first_nonempty(row.get("nickname"), row.get("user_nickname"), row.get("user_name"), row.get("author")),
            text=strip_html(first_nonempty(row.get("content"), row.get("text"), row.get("message"))),
            likes=row.get("like_count") or row.get("likes"),
            time=first_nonempty(row.get("create_time"), row.get("publish_time"), row.get("time")),
        ))
    return grouped


def _source_from_crawler_row(
    platform: PlatformName,
    row: dict[str, Any],
    comments_by_id: dict[str, list[Comment]],
    idx: int,
    comments_limit: int,
) -> Source:
    if platform == "douyin":
        item_id = first_nonempty(row.get("aweme_id"))
        comments = comments_by_id.get(item_id, [])[:comments_limit]
        title = strip_html(first_nonempty(row.get("title"), row.get("desc"), default="Untitled"))
        return Source(
            id=0,
            platform="douyin",
            title=title,
            creator=first_nonempty(row.get("nickname"), row.get("user_nickname")),
            url=first_nonempty(row.get("aweme_url"), default=f"https://www.douyin.com/video/{item_id}" if item_id else ""),
            thumbnail=first_nonempty(row.get("cover_url"), row.get("video_cover_url"), default=THUMBS["douyin"][(idx - 1) % len(THUMBS["douyin"])]),
            published=first_nonempty(row.get("create_time")),
            summary=text_excerpt([first_nonempty(row.get("desc")), *[c.text for c in comments[:4]]], 300),
            metrics={
                "likes": row.get("liked_count") or row.get("like_count"),
                "comments": row.get("comment_count"),
                "shares": row.get("share_count"),
                "downloads": bool(row.get("video_download_url")),
                "source_provider": "crawler",
            },
            comments=comments,
        )

    if platform == "xiaohongshu":
        item_id = first_nonempty(row.get("note_id"))
        comments = comments_by_id.get(item_id, [])[:comments_limit]
        title = strip_html(first_nonempty(row.get("title"), row.get("desc"), default="Untitled"))
        note_text = strip_html(first_nonempty(row.get("desc"), row.get("title")))
        image_list = [item for item in first_nonempty(row.get("image_list")).split(",") if item]
        return Source(
            id=0,
            platform="xiaohongshu",
            title=title,
            creator=first_nonempty(row.get("nickname"), row.get("user_nickname")),
            url=first_nonempty(row.get("note_url"), default=f"https://www.xiaohongshu.com/explore/{item_id}" if item_id else ""),
            thumbnail=first_nonempty(row.get("cover_url"), image_list[0] if image_list else "", row.get("avatar"), default=THUMBS["xiaohongshu"][(idx - 1) % len(THUMBS["xiaohongshu"])]),
            published=first_nonempty(row.get("time"), row.get("last_update_time")),
            summary=text_excerpt([note_text, *[c.text for c in comments[:4]]], 300),
            transcriptPreview=text_excerpt([note_text], 420),
            fullTranscript=note_text,
            transcriptText=note_text,
            metrics={
                "likes": row.get("liked_count"),
                "comments": row.get("comment_count"),
                "shares": row.get("share_count"),
                "collects": row.get("collected_count"),
                "media_type": row.get("type"),
                "video_url": row.get("video_url"),
                "image_count": len(image_list),
                "source_provider": "crawler",
            },
            comments=comments,
        )

    if platform == "zhihu":
        item_id = first_nonempty(row.get("content_id"))
        comments = comments_by_id.get(item_id, [])[:comments_limit]
        title = strip_html(first_nonempty(row.get("title"), row.get("desc"), default="Untitled"))
        content_text = strip_html(first_nonempty(row.get("content_text"), row.get("desc")))
        return Source(
            id=0,
            platform="zhihu",
            title=title,
            creator=first_nonempty(row.get("user_nickname"), row.get("nickname"), row.get("author")),
            url=first_nonempty(row.get("content_url"), default=f"https://www.zhihu.com/zvideo/{item_id}" if item_id and row.get("content_type") == "zvideo" else ""),
            thumbnail=first_nonempty(row.get("user_avatar"), row.get("avatar"), default=THUMBS["zhihu"][(idx - 1) % len(THUMBS["zhihu"])]),
            published=first_nonempty(row.get("created_time"), row.get("updated_time")),
            summary=text_excerpt([first_nonempty(row.get("desc"), row.get("title")), content_text, *[c.text for c in comments[:4]]], 300),
            transcriptPreview=text_excerpt([content_text], 420),
            fullTranscript=content_text,
            transcriptText=content_text,
            metrics={
                "upvotes": row.get("voteup_count"),
                "comments": row.get("comment_count"),
                "content_type": row.get("content_type"),
                "source_provider": "crawler",
            },
            comments=comments,
        )

    if platform == "kuaishou":
        item_id = first_nonempty(row.get("video_id"), row.get("photo_id"))
        comments = comments_by_id.get(item_id, [])[:comments_limit]
        title = strip_html(first_nonempty(row.get("title"), row.get("caption"), row.get("desc"), default="Untitled"))
        return Source(
            id=0,
            platform="kuaishou",
            title=title,
            creator=first_nonempty(row.get("nickname"), row.get("user_name"), row.get("author")),
            url=first_nonempty(row.get("video_url"), row.get("photo_url"), row.get("url"), default=f"https://www.kuaishou.com/short-video/{item_id}" if item_id else ""),
            thumbnail=first_nonempty(row.get("cover_url"), row.get("video_cover_url"), row.get("photo_cover"), default=THUMBS["kuaishou"][(idx - 1) % len(THUMBS["kuaishou"])]),
            published=first_nonempty(row.get("create_time"), row.get("publish_time")),
            summary=text_excerpt([first_nonempty(row.get("desc"), row.get("caption"), row.get("title")), *[c.text for c in comments[:4]]], 300),
            metrics={
                "likes": row.get("liked_count") or row.get("like_count"),
                "comments": row.get("comment_count"),
                "shares": row.get("share_count"),
                "plays": row.get("view_count") or row.get("play_count"),
                "source_provider": "crawler",
            },
            comments=comments,
        )

    if platform == "weibo":
        item_id = first_nonempty(row.get("note_id"), row.get("mblogid"), row.get("weibo_id"))
        comments = comments_by_id.get(item_id, [])[:comments_limit]
        text = strip_html(first_nonempty(row.get("content"), row.get("desc"), row.get("text")))
        title = strip_html(first_nonempty(row.get("title"), default=text[:48] or "Untitled"))
        return Source(
            id=0,
            platform="weibo",
            sourceType="post",
            title=title,
            creator=first_nonempty(row.get("nickname"), row.get("user_nickname"), row.get("user_name")),
            url=first_nonempty(row.get("note_url"), row.get("weibo_url"), row.get("url"), default=f"https://weibo.com/detail/{item_id}" if item_id else ""),
            thumbnail=first_nonempty(row.get("pic_url"), row.get("cover_url"), row.get("user_avatar"), default=THUMBS["weibo"][(idx - 1) % len(THUMBS["weibo"])]),
            published=first_nonempty(row.get("create_time"), row.get("publish_time")),
            summary=text_excerpt([text, *[c.text for c in comments[:4]]], 300),
            transcriptPreview=text_excerpt([text], 420),
            fullTranscript=text,
            transcriptText=text,
            metrics={
                "likes": row.get("liked_count") or row.get("like_count"),
                "comments": row.get("comment_count"),
                "shares": row.get("share_count") or row.get("repost_count"),
                "source_provider": "crawler",
            },
            comments=comments,
        )

    item_id = first_nonempty(row.get("video_id"), row.get("aid"))
    comments = comments_by_id.get(item_id, [])[:comments_limit]
    title = strip_html(first_nonempty(row.get("title"), row.get("desc"), default="Untitled"))
    return Source(
        id=0,
        platform="bilibili",
        title=title,
        creator=first_nonempty(row.get("nickname"), row.get("author"), row.get("up_name")),
        url=first_nonempty(row.get("video_url"), row.get("url"), default=f"https://www.bilibili.com/video/av{item_id}" if item_id else ""),
        thumbnail=first_nonempty(row.get("video_cover_url"), row.get("cover"), row.get("pic"), default=THUMBS["bilibili"][(idx - 1) % len(THUMBS["bilibili"])]),
        published=first_nonempty(row.get("create_time"), row.get("pubdate")),
        summary=text_excerpt([first_nonempty(row.get("desc"), row.get("title")), *[c.text for c in comments[:4]]], 300),
        metrics={
            "likes": row.get("liked_count"),
            "views": row.get("video_play_count"),
            "comments": row.get("video_comment"),
            "shares": row.get("video_share_count"),
            "coins": row.get("video_coin_count"),
            "source_provider": "crawler",
        },
        comments=comments,
    )
