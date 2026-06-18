import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
import psycopg
from psycopg.rows import dict_row
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from core import settings
from memory.postgres import get_postgres_connection_string
from schema import (
    IngestedArticleAsset,
    IngestedArticleChunk,
    IngestedArticleDetail,
    IngestedArticleListItem,
    IngestedArticleListResponse,
    XhsIngestResponse,
)

logger = logging.getLogger(__name__)

XHS_LOCAL_DIR = Path(os.getenv("MYAGENT_DATA_DIR") or os.getenv("LOCALAPPDATA", ".local")) / "myAgent"
XHS_PROFILE_DIR = Path(os.getenv("XHS_PROFILE_DIR", str(XHS_LOCAL_DIR / "xhs-playwright-profile")))
XHS_ASSET_DIR = Path(os.getenv("XHS_ASSET_DIR", str(XHS_LOCAL_DIR / "xhs-assets")))
XHS_PLATFORM = "xiaohongshu"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
BAIDU_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
BAIDU_OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
MAX_XHS_SCREENSHOT_ASSETS = 6
MAX_BAIDU_IMAGE_BYTES = 3_500_000
CHROME_PATHS = [
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/google-chrome-stable"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


class XhsIngestError(Exception):
    pass


class XhsLoginRequired(XhsIngestError):
    pass


@dataclass
class XhsAsset:
    image_url: str
    local_path: str | None = None
    ocr_text: str = ""
    ocr_status: str = "pending"
    ocr_error: str | None = None
    content_type: str = "image/jpeg"
    image_bytes: bytes | None = field(default=None, repr=False)


@dataclass
class XhsArticle:
    note_key: str
    source_url: str
    canonical_url: str
    platform: str = XHS_PLATFORM
    title: str = ""
    author: str = ""
    published_at: str | None = None
    body_text: str = ""
    tags: list[str] = field(default_factory=list)
    assets: list[XhsAsset] = field(default_factory=list)

    @property
    def ocr_text(self) -> str:
        return "\n\n".join(asset.ocr_text for asset in self.assets if asset.ocr_text).strip()

    @property
    def combined_text(self) -> str:
        parts = [self.title, self.body_text]
        if self.tags:
            parts.append(" ".join(f"#{tag}" for tag in self.tags))
        if self.ocr_text:
            parts.append(f"Image OCR:\n{self.ocr_text}")
        return "\n\n".join(part for part in parts if part).strip()


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Invalid URL")
    return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", parsed.query, ""))


def normalize_xhs_url(url: str) -> str:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    host = parsed.netloc.lower()
    if not any(domain in host for domain in ("xiaohongshu.com", "xhslink.com")):
        raise ValueError("URL is not a Xiaohongshu link")
    return normalized


def is_xhs_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(domain in host for domain in ("xiaohongshu.com", "xhslink.com"))


def platform_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    host = host.removeprefix("www.")
    if "xiaohongshu.com" in host or "xhslink.com" in host:
        return XHS_PLATFORM
    return re.sub(r"[^a-z0-9]+", "_", host).strip("_") or "web"


def extract_note_key(url: str) -> str | None:
    parsed = urlparse(url)
    patterns = [
        r"/explore/([^/?#]+)",
        r"/discovery/item/([^/?#]+)",
        r"/search_result/([^/?#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, parsed.path)
        if match:
            return match.group(1)
    return None


def fallback_note_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def looks_like_login_page(html: str, url: str) -> bool:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    login_markers = ("登录", "验证码", "扫码登录", "login")
    if "login" in url.lower():
        return True
    return any(marker in text for marker in login_markers) and len(text) < 800


def extract_article_from_html(
    html: str, final_url: str, source_url: str, platform: str = XHS_PLATFORM
) -> XhsArticle:
    soup = BeautifulSoup(html, "html.parser")
    title = _first_meta(soup, ["og:title", "twitter:title"]) or (soup.title.string if soup.title else "")
    description = _first_meta(soup, ["description", "og:description", "twitter:description"])
    author = _first_meta(soup, ["author"])
    canonical = _canonical_url(soup) or final_url
    note_key = (
        extract_note_key(canonical) or extract_note_key(final_url)
        if platform == XHS_PLATFORM
        else None
    ) or fallback_note_key(canonical)

    json_texts = [
        script.get_text(strip=True)
        for script in soup.find_all("script")
        if script.get_text(strip=True).startswith(("{", "["))
    ]
    body_candidates = [description or ""]
    tags: set[str] = set()
    image_urls: set[str] = set()
    published_at = None

    for raw_json in json_texts:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        for value in _walk_json(payload):
            if isinstance(value, str):
                stripped = value.strip()
                if len(stripped) > 40 and _looks_like_note_text(stripped):
                    body_candidates.append(stripped)
                if stripped.startswith("#") and len(stripped) < 80:
                    tags.add(stripped.lstrip("#"))
                if _looks_like_image_url(stripped):
                    image_urls.add(stripped)
                if not published_at and re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}.*", stripped):
                    published_at = stripped

    body_candidates.extend(_extract_body_text_candidates(soup))
    title = title or _selector_text(soup, "#activity-name, h1, .article-title, .post-title")
    author = author or _selector_text(soup, "#js_name, .author, .byline")

    for img in soup.find_all("img"):
        for attr in ("data-src", "data-original", "data-backsrc", "data-lazy-src", "src"):
            src = img.get(attr)
            if src and _looks_like_image_url(src):
                image_urls.add(src)

    body_text = max(body_candidates, key=len, default="").strip()
    body_text = _clean_text(body_text)
    if not title and body_text:
        title = body_text[:40]

    if not body_text and not image_urls:
        raise XhsIngestError("No readable note content was found")

    return XhsArticle(
        note_key=note_key,
        source_url=source_url,
        canonical_url=canonical,
        platform=platform,
        title=_clean_text(title),
        author=_clean_text(author or ""),
        published_at=published_at,
        body_text=body_text,
        tags=sorted(tags),
        assets=[XhsAsset(image_url=url) for url in sorted(image_urls)],
    )


async def open_xhs_login_window() -> None:
    XHS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    browser_path = _find_system_browser()
    if browser_path is None:
        raise XhsIngestError("No system browser executable was found for opening a login window.")
    launcher = XHS_LOCAL_DIR / "open-xhs-login.cmd"
    launcher.write_text(
        "\r\n".join(
            [
                "@echo off",
                (
                    f'start "XHS Login" "{browser_path}" '
                    f'"--user-data-dir={XHS_PROFILE_DIR.resolve()}" '
                    '--new-window "https://www.xiaohongshu.com"'
                ),
            ]
        ),
        encoding="utf-8",
    )
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", f"Start-Process -FilePath '{launcher.resolve()}'"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class XhsImporter:
    def __init__(self, chroma_dir: str = CHROMA_DIR):
        self.chroma_dir = chroma_dir
        self._baidu_access_token: str | None = None
        self._baidu_token_expires_at = 0.0

    async def import_url(self, url: str, force_refresh: bool = False) -> XhsIngestResponse:
        try:
            normalized_url = normalize_xhs_url(url)
        except ValueError as e:
            return XhsIngestResponse(status="failed", message=str(e))

        try:
            article = await self.fetch_article(normalized_url)
            await self.enrich_assets_with_ocr(article)
            return self.persist_article(article, force_refresh=force_refresh)
        except XhsLoginRequired:
            return XhsIngestResponse(
                status="login_required",
                message="Xiaohongshu login is required. Open the login window and retry.",
                source_url=normalized_url,
                needs_login=True,
            )
        except Exception as e:
            logger.exception("Failed to import Xiaohongshu note")
            message = str(e) or e.__class__.__name__
            if _profile_is_locked_error(message):
                message = (
                    "Xiaohongshu login browser is still open. Complete login, close that "
                    "Chrome window, then retry import."
                )
                return XhsIngestResponse(
                    status="login_required",
                    message=message,
                    source_url=normalized_url,
                    needs_login=True,
                )
            return XhsIngestResponse(status="failed", message=message, source_url=normalized_url)

    async def fetch_article(self, url: str) -> XhsArticle:
        return await asyncio.to_thread(self._fetch_article_sync, url)

    def _fetch_article_sync(self, url: str) -> XhsArticle:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright

        html = ""
        final_url = url
        XHS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        previous_policy = asyncio.get_event_loop_policy()
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            with sync_playwright() as playwright:
                browser_path = _find_system_browser()
                launch_options: dict[str, Any] = {
                    "headless": True,
                    "viewport": {"width": 1440, "height": 1000},
                    "args": ["--disable-blink-features=AutomationControlled"],
                }
                if browser_path is not None:
                    launch_options["executable_path"] = str(browser_path)
                context = playwright.chromium.launch_persistent_context(
                    str(XHS_PROFILE_DIR),
                    **launch_options,
                )
                page = context.pages[0] if context.pages else context.new_page()
                screenshot_assets: list[XhsAsset] = []
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except PlaywrightTimeoutError:
                        pass
                    if self._dismiss_xhs_login_dialog(page):
                        page.wait_for_timeout(800)
                    html = page.content()
                    final_url = page.url
                    if not looks_like_login_page(html, final_url):
                        screenshot_assets = self._capture_main_visual_assets(page, XHS_PLATFORM)
                finally:
                    context.close()
        finally:
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(previous_policy)

        if looks_like_login_page(html, final_url):
            raise XhsLoginRequired()
        article = extract_article_from_html(html, final_url=final_url, source_url=url)
        article.assets = _merge_assets(screenshot_assets + article.assets)
        return article

    def _dismiss_xhs_login_dialog(self, page: Any) -> bool:
        selectors = [
            "[class*='login'] [class*='close']",
            "[class*='modal'] [class*='close']",
            "[class*='mask'] [class*='close']",
            "[aria-label='close']",
            "[aria-label='Close']",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if locator.is_visible(timeout=500):
                    locator.click(force=True, timeout=1000)
                    return True
            except Exception:
                continue

        try:
            return bool(
                page.evaluate(
                    """
                    () => {
                      const candidates = Array.from(document.querySelectorAll('button, div, span'));
                      for (const el of candidates) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width < 12 || rect.height < 12 || rect.width > 90 || rect.height > 90) {
                          continue;
                        }
                        const text = (el.innerText || el.getAttribute('aria-label') || '').trim();
                        const cls = String(el.className || '').toLowerCase();
                        const nearTopRightOfModal =
                          rect.left > window.innerWidth * 0.45 &&
                          rect.top < window.innerHeight * 0.35;
                        const looksClose =
                          text === '×' || text.toLowerCase() === 'x' || cls.includes('close');
                        if (nearTopRightOfModal && looksClose) {
                          el.click();
                          return true;
                        }
                      }
                      return false;
                    }
                    """
                )
            )
        except Exception:
            return False

    def _capture_main_visual_assets(
        self, page: Any, platform: str = XHS_PLATFORM
    ) -> list[XhsAsset]:
        clip = self._find_main_visual_clip(page)
        if not clip:
            return []

        assets: list[XhsAsset] = []
        seen_hashes: set[str] = set()
        for index in range(MAX_XHS_SCREENSHOT_ASSETS):
            try:
                image_bytes = page.screenshot(clip=clip)
                image_bytes, content_type = _prepare_image_for_baidu(image_bytes)
            except Exception as e:
                logger.info("Failed to capture Xiaohongshu main visual screenshot: %s", e)
                break

            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest in seen_hashes:
                if index == 0:
                    break
                break
            seen_hashes.add(digest)
            assets.append(
                XhsAsset(
                    image_url=f"screenshot://{platform}/main/{index}",
                    content_type=content_type,
                    image_bytes=image_bytes,
                )
            )

            if not self._advance_xhs_carousel(page, clip):
                break
            page.wait_for_timeout(700)
            next_clip = self._find_main_visual_clip(page)
            if next_clip:
                clip = next_clip

        return assets

    def _find_main_visual_clip(self, page: Any) -> dict[str, float] | None:
        try:
            clip = page.evaluate(
                """
                () => {
                  const vw = window.innerWidth;
                  const vh = window.innerHeight;
                  const candidates = [];
                  const nodes = Array.from(document.querySelectorAll('img, canvas, video, div'));
                  for (const el of nodes) {
                    const style = window.getComputedStyle(el);
                    if (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) {
                      continue;
                    }
                    const rect = el.getBoundingClientRect();
                    const width = rect.width;
                    const height = rect.height;
                    if (width < 260 || height < 260) continue;
                    if (width > vw * 0.82 || height > vh * 0.96) continue;
                    if (rect.right < vw * 0.18 || rect.left > vw * 0.72) continue;
                    if (rect.bottom < 80 || rect.top > vh * 0.92) continue;

                    const cls = String(el.className || '').toLowerCase();
                    const tag = el.tagName.toLowerCase();
                    const hasBackground = style.backgroundImage && style.backgroundImage !== 'none';
                    const imageLike = ['img', 'canvas', 'video'].includes(tag) || hasBackground ||
                      /media|swiper|slider|carousel|image|picture|note/.test(cls);
                    if (!imageLike) continue;

                    const centerX = rect.left + width / 2;
                    let score = width * height;
                    if (/media|swiper|slider|carousel|image|picture/.test(cls)) score *= 2.2;
                    if (centerX < vw * 0.62) score *= 1.8;
                    if (centerX > vw * 0.72) score *= 0.15;
                    if (rect.left < 180 && width < 360) score *= 0.1;
                    candidates.push({ score, rect });
                  }
                  candidates.sort((a, b) => b.score - a.score);
                  if (!candidates.length) return null;
                  const r = candidates[0].rect;
                  return {
                    x: Math.max(0, r.left),
                    y: Math.max(0, r.top),
                    width: Math.min(r.width, vw - Math.max(0, r.left)),
                    height: Math.min(r.height, vh - Math.max(0, r.top))
                  };
                }
                """
            )
            if not clip or clip["width"] < 100 or clip["height"] < 100:
                return None
            return clip
        except Exception as e:
            logger.info("Failed to locate Xiaohongshu main visual area: %s", e)
            return None

    def _advance_xhs_carousel(self, page: Any, clip: dict[str, float]) -> bool:
        try:
            page.mouse.click(clip["x"] + clip["width"] - 32, clip["y"] + clip["height"] / 2)
            return True
        except Exception:
            return False

    async def enrich_assets_with_ocr(self, article: XhsArticle) -> None:
        if (
            not article.assets
            or not settings.BAIDU_OCR_API_KEY
        ):
            return
        for asset in article.assets:
            self._cache_embedded_asset(asset)
        semaphore = asyncio.Semaphore(3)
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                auths = await self._get_baidu_ocr_auths(client)
            except Exception as e:
                message = str(e)
                for asset in article.assets:
                    asset.ocr_status = "failed"
                    asset.ocr_error = message
                return
            await asyncio.gather(
                *(
                    self._ocr_asset(asset, client, semaphore, auths)
                    for asset in article.assets
                )
            )

    async def _get_baidu_ocr_auths(self, client: httpx.AsyncClient) -> list[dict[str, str]]:
        auths: list[dict[str, str]] = []
        if settings.BAIDU_OCR_API_KEY:
            auths.append(
                {
                    "mode": "api_key",
                    "credential": settings.BAIDU_OCR_API_KEY.get_secret_value(),
                }
            )
        if settings.BAIDU_OCR_API_KEY and settings.BAIDU_OCR_SECRET_KEY:
            try:
                auths.append(
                    {
                        "mode": "access_token",
                        "credential": await self._get_baidu_access_token(client),
                    }
                )
            except Exception as e:
                auths.append({"mode": "access_token", "error": str(e)})
        if not auths:
            raise XhsIngestError("Baidu OCR API key is required.")
        return auths

    async def _get_baidu_access_token(self, client: httpx.AsyncClient) -> str:
        if self._baidu_access_token and time.time() < self._baidu_token_expires_at:
            return self._baidu_access_token
        if not settings.BAIDU_OCR_API_KEY or not settings.BAIDU_OCR_SECRET_KEY:
            raise XhsIngestError("Baidu OCR API key and secret key are required.")

        response = await client.post(
            BAIDU_TOKEN_URL,
            params={
                "grant_type": "client_credentials",
                "client_id": settings.BAIDU_OCR_API_KEY.get_secret_value(),
                "client_secret": settings.BAIDU_OCR_SECRET_KEY.get_secret_value(),
            },
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"error_description": response.text}
        if response.status_code >= 400:
            detail = payload.get("error_description") or payload.get("error") or response.text
            raise XhsIngestError(f"Baidu OCR token request failed: {detail}")
        if "access_token" not in payload:
            raise XhsIngestError(
                f"Baidu OCR token request failed: {payload.get('error_description') or payload}"
            )
        self._baidu_access_token = payload["access_token"]
        self._baidu_token_expires_at = time.time() + int(payload.get("expires_in", 3600)) - 60
        return self._baidu_access_token

    def _cache_embedded_asset(self, asset: XhsAsset) -> None:
        if asset.local_path or asset.image_bytes is None:
            return
        try:
            image_bytes, asset.content_type = _prepare_image_for_baidu(asset.image_bytes)
            asset.image_bytes = image_bytes
            asset.local_path = self._write_asset(
                asset.image_url, image_bytes, content_type=asset.content_type
            )
        except Exception as e:
            logger.info("Failed to cache embedded Xiaohongshu asset: %s", e)

    async def _ocr_asset(
        self,
        asset: XhsAsset,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        auths: list[dict[str, str]],
    ) -> None:
        async with semaphore:
            try:
                if asset.image_bytes is None:
                    response = await client.get(asset.image_url)
                    response.raise_for_status()
                    asset.content_type = response.headers.get(
                        "content-type", "image/jpeg"
                    ).split(";")[0]
                    image_bytes = response.content
                else:
                    image_bytes = asset.image_bytes
                image_bytes, asset.content_type = _prepare_image_for_baidu(image_bytes)
                asset.local_path = self._write_asset(
                    asset.image_url, image_bytes, content_type=asset.content_type
                )
                asset.ocr_text = await self._ocr_image_with_auths(client, image_bytes, auths)
                asset.ocr_status = "success"
                asset.image_bytes = None
            except Exception as e:
                asset.ocr_status = "failed"
                asset.ocr_error = str(e)

    async def _ocr_image_with_auths(
        self, client: httpx.AsyncClient, image_bytes: bytes, auths: list[dict[str, str]]
    ) -> str:
        errors: list[str] = []
        for auth in auths:
            if auth.get("error"):
                errors.append(f"{auth['mode']}: {auth['error']}")
                continue
            try:
                return await self._ocr_image(client, image_bytes, auth)
            except Exception as e:
                errors.append(f"{auth['mode']}: {e}")
        raise XhsIngestError("Baidu OCR failed with all auth modes: " + "; ".join(errors))

    async def _ocr_image(
        self, client: httpx.AsyncClient, image_bytes: bytes, auth: dict[str, str] | str
    ) -> str:
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        if isinstance(auth, str):
            auth = {"mode": "access_token", "credential": auth}
        params = {}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if auth["mode"] == "api_key":
            headers["Authorization"] = f"Bearer {auth['credential']}"
        else:
            params["access_token"] = auth["credential"]
        response = await client.post(
            BAIDU_OCR_URL,
            params=params,
            data={
                "image": image_b64,
                "detect_direction": "true",
                "paragraph": "false",
            },
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        if "error_code" in payload:
            raise XhsIngestError(
                f"Baidu OCR failed: {payload.get('error_msg')} ({payload.get('error_code')})"
            )
        words = [
            str(item.get("words", "")).strip()
            for item in payload.get("words_result", [])
            if str(item.get("words", "")).strip()
        ]
        return "\n".join(words)

    def persist_article(self, article: XhsArticle, force_refresh: bool = False) -> XhsIngestResponse:
        if not article.combined_text:
            return XhsIngestResponse(
                status="failed",
                message="No text content was extracted from the note.",
                source_url=article.canonical_url,
            )

        self._ensure_tables()
        chunks = self._split_article(article)
        article_id = self._upsert_article(article, force_refresh=force_refresh)
        for chunk in chunks:
            chunk.metadata["article_id"] = article_id
        self._delete_existing_chunks(article_id)
        chroma_ids = self._write_chroma(article_id, article, chunks)
        self._write_assets_and_chunks(article_id, article, chunks, chroma_ids)

        ocr_failed_count = sum(1 for asset in article.assets if asset.ocr_status == "failed")
        status = "partial_success" if ocr_failed_count else "success"
        message = "Imported Xiaohongshu note."
        if ocr_failed_count:
            message = "Imported note, but some image OCR tasks failed."

        return XhsIngestResponse(
            status=status,
            message=message,
            article_id=article_id,
            title=article.title,
            source_url=article.canonical_url,
            chunk_count=len(chunks),
            asset_count=len(article.assets),
            ocr_failed_count=ocr_failed_count,
        )

    def _ensure_tables(self) -> None:
        with psycopg.connect(get_postgres_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_articles (
                        article_id TEXT PRIMARY KEY,
                        source_platform TEXT NOT NULL,
                        note_key TEXT NOT NULL UNIQUE,
                        source_url TEXT NOT NULL,
                        canonical_url TEXT NOT NULL,
                        title TEXT,
                        author TEXT,
                        published_at TEXT,
                        body_text TEXT,
                        ocr_text TEXT,
                        combined_text TEXT NOT NULL,
                        tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        fetched_at TIMESTAMPTZ NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_article_assets (
                        asset_id TEXT PRIMARY KEY,
                        article_id TEXT NOT NULL REFERENCES rag_articles(article_id) ON DELETE CASCADE,
                        image_url TEXT NOT NULL,
                        local_path TEXT,
                        ocr_text TEXT,
                        ocr_status TEXT NOT NULL,
                        ocr_error TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        article_id TEXT NOT NULL REFERENCES rag_articles(article_id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        chroma_document_id TEXT NOT NULL,
                        chunk_text TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                conn.commit()

    def _upsert_article(self, article: XhsArticle, force_refresh: bool) -> str:
        with psycopg.connect(get_postgres_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT article_id FROM rag_articles WHERE note_key = %s",
                    (article.note_key,),
                )
                existing = cur.fetchone()
                article_id = existing[0] if existing else str(uuid.uuid4())
                if existing and not force_refresh:
                    # Updating is the chosen duplicate policy, so force_refresh only exists for API symmetry.
                    pass
                cur.execute(
                    """
                    INSERT INTO rag_articles (
                        article_id, source_platform, note_key, source_url, canonical_url,
                        title, author, published_at, body_text, ocr_text, combined_text,
                        tags, status, error_message, fetched_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s)
                    ON CONFLICT (note_key) DO UPDATE SET
                        source_url = EXCLUDED.source_url,
                        canonical_url = EXCLUDED.canonical_url,
                        title = EXCLUDED.title,
                        author = EXCLUDED.author,
                        published_at = EXCLUDED.published_at,
                        body_text = EXCLUDED.body_text,
                        ocr_text = EXCLUDED.ocr_text,
                        combined_text = EXCLUDED.combined_text,
                        tags = EXCLUDED.tags,
                        status = EXCLUDED.status,
                        error_message = NULL,
                        fetched_at = EXCLUDED.fetched_at,
                        updated_at = now()
                    """,
                    (
                        article_id,
                        article.platform,
                        article.note_key,
                        article.source_url,
                        article.canonical_url,
                        article.title,
                        article.author,
                        article.published_at,
                        article.body_text,
                        article.ocr_text,
                        article.combined_text,
                        json.dumps(article.tags, ensure_ascii=False),
                        "imported",
                        datetime.now(UTC),
                    ),
                )
                conn.commit()
        return article_id

    def _delete_existing_chunks(self, article_id: str) -> None:
        chroma = self._chroma()
        try:
            chroma.delete(where={"article_id": article_id})
        except Exception:
            logger.info("No existing Chroma chunks found for article %s", article_id)
        with psycopg.connect(get_postgres_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rag_article_assets WHERE article_id = %s", (article_id,))
                cur.execute("DELETE FROM rag_chunks WHERE article_id = %s", (article_id,))
                conn.commit()

    def _write_assets_and_chunks(
        self, article_id: str, article: XhsArticle, chunks: list[Document], chroma_ids: list[str]
    ) -> None:
        with psycopg.connect(get_postgres_connection_string()) as conn:
            with conn.cursor() as cur:
                for asset in article.assets:
                    cur.execute(
                        """
                        INSERT INTO rag_article_assets (
                            asset_id, article_id, image_url, local_path,
                            ocr_text, ocr_status, ocr_error
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(uuid.uuid4()),
                            article_id,
                            asset.image_url,
                            asset.local_path,
                            asset.ocr_text,
                            asset.ocr_status,
                            asset.ocr_error,
                        ),
                    )
                for index, (chunk, chroma_id) in enumerate(zip(chunks, chroma_ids, strict=True)):
                    cur.execute(
                        """
                        INSERT INTO rag_chunks (
                            chunk_id, article_id, chunk_index, chroma_document_id, chunk_text
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            chunk.metadata["chunk_id"],
                            article_id,
                            index,
                            chroma_id,
                            chunk.page_content,
                        ),
                    )
                conn.commit()

    def _split_article(self, article: XhsArticle) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        text = article.combined_text
        docs = splitter.create_documents([text])
        for index, doc in enumerate(docs):
            doc.metadata.update(
                {
                    "chunk_id": f"{article.note_key}:{index}",
                    "source_platform": article.platform,
                    "source": article.canonical_url,
                    "source_url": article.canonical_url,
                    "title": article.title,
                    "note_key": article.note_key,
                    "chunk_index": index,
                }
            )
        return docs

    def _write_chroma(self, article_id: str, article: XhsArticle, chunks: list[Document]) -> list[str]:
        return self._chroma().add_documents(chunks)

    def _chroma(self) -> Chroma:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory=self.chroma_dir, embedding_function=embeddings)

    def _write_asset(self, image_url: str, content: bytes, content_type: str | None = None) -> str:
        XHS_ASSET_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(urlparse(image_url).path).suffix or _suffix_for_content_type(content_type)
        file_name = hashlib.sha256(image_url.encode("utf-8")).hexdigest() + suffix
        path = XHS_ASSET_DIR / file_name
        path.write_bytes(content)
        return str(path)


class ArticleImporter(XhsImporter):
    async def import_url(self, url: str, force_refresh: bool = False) -> XhsIngestResponse:
        try:
            normalized_url = normalize_url(url)
        except ValueError as e:
            return XhsIngestResponse(status="failed", message=str(e))

        if is_xhs_url(normalized_url):
            return await super().import_url(normalized_url, force_refresh=force_refresh)

        try:
            article = await self.fetch_article(normalized_url)
            await self.enrich_assets_with_ocr(article)
            return self.persist_article(article, force_refresh=force_refresh)
        except Exception as e:
            logger.exception("Failed to import article")
            message = str(e) or e.__class__.__name__
            return XhsIngestResponse(status="failed", message=message, source_url=normalized_url)

    async def fetch_article(self, url: str) -> XhsArticle:
        if is_xhs_url(url):
            return await super().fetch_article(url)
        return await asyncio.to_thread(self._fetch_generic_article_sync, url)

    def _fetch_generic_article_sync(self, url: str) -> XhsArticle:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright

        html = ""
        final_url = url
        platform = platform_from_url(url)
        XHS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        previous_policy = asyncio.get_event_loop_policy()
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            with sync_playwright() as playwright:
                browser_path = _find_system_browser()
                launch_options: dict[str, Any] = {
                    "headless": True,
                    "viewport": {"width": 1440, "height": 1000},
                    "args": ["--disable-blink-features=AutomationControlled"],
                }
                if browser_path is not None:
                    launch_options["executable_path"] = str(browser_path)
                context = playwright.chromium.launch_persistent_context(
                    str(XHS_PROFILE_DIR),
                    **launch_options,
                )
                page = context.pages[0] if context.pages else context.new_page()
                screenshot_assets: list[XhsAsset] = []
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except PlaywrightTimeoutError:
                        pass
                    self._dismiss_xhs_login_dialog(page)
                    page.wait_for_timeout(500)
                    html = page.content()
                    final_url = page.url
                    screenshot_assets = self._capture_main_visual_assets(page, platform)
                finally:
                    context.close()
        finally:
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(previous_policy)

        article = extract_article_from_html(
            html,
            final_url=final_url,
            source_url=url,
            platform=platform_from_url(final_url),
        )
        article.assets = _merge_assets(screenshot_assets + article.assets)
        return article


def list_ingested_articles(
    *,
    q: str | None = None,
    platform: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    has_ocr_failed: bool | None = None,
    sort: str = "newest",
    page: int = 1,
    page_size: int = 20,
    search_mode: str = "keyword",
) -> IngestedArticleListResponse:
    XhsImporter()._ensure_tables()
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)
    where, params = _article_where_params(
        q=q,
        platform=platform,
        status=status,
        date_from=date_from,
        date_to=date_to,
        has_ocr_failed=has_ocr_failed,
        search_mode=search_mode,
    )
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    order_sql = _article_order_sql(sort)
    offset = (page - 1) * page_size

    with psycopg.connect(get_postgres_connection_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS total FROM rag_articles a {where_sql}", params)
            total = int(cur.fetchone()["total"])
            cur.execute(
                f"""
                SELECT
                    a.article_id,
                    a.source_platform,
                    a.source_url,
                    a.canonical_url,
                    a.title,
                    a.author,
                    a.published_at,
                    a.tags,
                    a.status,
                    a.fetched_at,
                    a.created_at,
                    a.updated_at,
                    LEFT(COALESCE(NULLIF(a.body_text, ''), NULLIF(a.ocr_text, ''), a.combined_text, ''), 240) AS snippet,
                    COALESCE(chunk_stats.chunk_count, 0) AS chunk_count,
                    COALESCE(asset_stats.asset_count, 0) AS asset_count,
                    COALESCE(asset_stats.ocr_failed_count, 0) AS ocr_failed_count
                FROM rag_articles a
                LEFT JOIN (
                    SELECT article_id, COUNT(*)::int AS chunk_count
                    FROM rag_chunks
                    GROUP BY article_id
                ) chunk_stats ON chunk_stats.article_id = a.article_id
                LEFT JOIN (
                    SELECT
                        article_id,
                        COUNT(*)::int AS asset_count,
                        COUNT(*) FILTER (WHERE ocr_status = 'failed')::int AS ocr_failed_count
                    FROM rag_article_assets
                    GROUP BY article_id
                ) asset_stats ON asset_stats.article_id = a.article_id
                {where_sql}
                {order_sql}
                LIMIT %s OFFSET %s
                """,
                [*params, page_size, offset],
            )
            rows = cur.fetchall()

    total_pages = (total + page_size - 1) // page_size if total else 0
    return IngestedArticleListResponse(
        items=[_article_list_item_from_row(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


def get_ingested_article(article_id: str) -> IngestedArticleDetail | None:
    XhsImporter()._ensure_tables()
    with psycopg.connect(get_postgres_connection_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    a.article_id,
                    a.source_platform,
                    a.source_url,
                    a.canonical_url,
                    a.title,
                    a.author,
                    a.published_at,
                    a.body_text,
                    a.ocr_text,
                    a.combined_text,
                    a.tags,
                    a.status,
                    a.fetched_at,
                    a.created_at,
                    a.updated_at,
                    LEFT(COALESCE(NULLIF(a.body_text, ''), NULLIF(a.ocr_text, ''), a.combined_text, ''), 240) AS snippet,
                    COALESCE(chunk_stats.chunk_count, 0) AS chunk_count,
                    COALESCE(asset_stats.asset_count, 0) AS asset_count,
                    COALESCE(asset_stats.ocr_failed_count, 0) AS ocr_failed_count
                FROM rag_articles a
                LEFT JOIN (
                    SELECT article_id, COUNT(*)::int AS chunk_count
                    FROM rag_chunks
                    GROUP BY article_id
                ) chunk_stats ON chunk_stats.article_id = a.article_id
                LEFT JOIN (
                    SELECT
                        article_id,
                        COUNT(*)::int AS asset_count,
                        COUNT(*) FILTER (WHERE ocr_status = 'failed')::int AS ocr_failed_count
                    FROM rag_article_assets
                    GROUP BY article_id
                ) asset_stats ON asset_stats.article_id = a.article_id
                WHERE a.article_id = %s
                """,
                (article_id,),
            )
            article = cur.fetchone()
            if not article:
                return None

            cur.execute(
                """
                SELECT asset_id, image_url, local_path, ocr_text, ocr_status, ocr_error, created_at
                FROM rag_article_assets
                WHERE article_id = %s
                ORDER BY created_at ASC, asset_id ASC
                """,
                (article_id,),
            )
            assets = [IngestedArticleAsset(**_serialize_row(row)) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT chunk_id, chunk_index, chroma_document_id, chunk_text, created_at
                FROM rag_chunks
                WHERE article_id = %s
                ORDER BY chunk_index ASC
                """,
                (article_id,),
            )
            chunks = [IngestedArticleChunk(**_serialize_row(row)) for row in cur.fetchall()]

    base = _article_list_item_from_row(article).model_dump()
    return IngestedArticleDetail(
        **base,
        body_text=article.get("body_text"),
        ocr_text=article.get("ocr_text"),
        combined_text=article.get("combined_text") or "",
        assets=assets,
        chunks=chunks,
    )


def _article_where_params(
    *,
    q: str | None,
    platform: str | None,
    status: str | None,
    date_from: str | None,
    date_to: str | None,
    has_ocr_failed: bool | None,
    search_mode: str,
) -> tuple[list[str], list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    if q:
        keyword = f"%{q.strip()}%"
        # search_mode is accepted for API compatibility; v1 uses keyword search for list retrieval.
        where.append(
            """
            (
                a.title ILIKE %s
                OR a.body_text ILIKE %s
                OR a.source_platform ILIKE %s
                OR a.source_url ILIKE %s
                OR a.canonical_url ILIKE %s
            )
            """
        )
        params.extend([keyword] * 5)
    if platform:
        where.append("a.source_platform = %s")
        params.append(platform)
    if status:
        where.append("a.status = %s")
        params.append(status)
    if date_from:
        where.append("a.fetched_at::date >= %s::date")
        params.append(date_from)
    if date_to:
        where.append("a.fetched_at::date <= %s::date")
        params.append(date_to)
    if has_ocr_failed is True:
        where.append(
            """
            EXISTS (
                SELECT 1 FROM rag_article_assets aa
                WHERE aa.article_id = a.article_id AND aa.ocr_status = 'failed'
            )
            """
        )
    elif has_ocr_failed is False:
        where.append(
            """
            NOT EXISTS (
                SELECT 1 FROM rag_article_assets aa
                WHERE aa.article_id = a.article_id AND aa.ocr_status = 'failed'
            )
            """
        )
    return where, params


def _article_order_sql(sort: str) -> str:
    match sort:
        case "oldest":
            return "ORDER BY a.fetched_at ASC, a.created_at ASC"
        case "title":
            return "ORDER BY a.title ASC NULLS LAST, a.fetched_at DESC"
        case "platform":
            return "ORDER BY a.source_platform ASC, a.fetched_at DESC"
        case _:
            return "ORDER BY a.fetched_at DESC, a.created_at DESC"


def _article_list_item_from_row(row: dict[str, Any]) -> IngestedArticleListItem:
    data = _serialize_row(row)
    data["tags"] = _normalize_tags(data.get("tags"))
    data["snippet"] = _clean_text(data.get("snippet") or "")
    return IngestedArticleListItem(**data)


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return []


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized


def _first_meta(soup: BeautifulSoup, names: list[str]) -> str | None:
    for name in names:
        node = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if node and node.get("content"):
            return str(node["content"])
    return None


def _canonical_url(soup: BeautifulSoup) -> str | None:
    node = soup.find("link", attrs={"rel": "canonical"})
    if node and node.get("href"):
        return str(node["href"])
    return None


def _walk_json(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _extract_body_text_candidates(soup: BeautifulSoup) -> list[str]:
    selectors = [
        "#js_content",
        ".rich_media_content",
        "article",
        "main",
        "[role='main']",
        ".article-content",
        ".article_content",
        ".post-content",
        ".entry-content",
        ".content",
        "#content",
    ]
    candidates: list[str] = []
    seen: set[str] = set()
    for selector in selectors:
        for node in soup.select(selector):
            for removable in node.select("script, style, noscript, svg"):
                removable.decompose()
            text = _clean_text(node.get_text("\n", strip=True))
            if len(text) > 40 and text not in seen:
                candidates.append(text)
                seen.add(text)
    return candidates


def _selector_text(soup: BeautifulSoup, selector: str) -> str:
    node = soup.select_one(selector)
    if not node:
        return ""
    return _clean_text(node.get_text(" ", strip=True))


def _looks_like_note_text(text: str) -> bool:
    if text.startswith(("http://", "https://", "{", "[")):
        return False
    return any(char in text for char in ("。", "，", "\n", "#")) or len(text) > 80


def _looks_like_image_url(value: str) -> bool:
    lower = value.lower()
    if "sns-avatar" in lower or "avatar" in lower:
        return False
    if "mmbiz.qpic.cn" in lower or "qpic.cn" in lower or "wx_fmt=" in lower:
        return value.startswith(("http://", "https://"))
    return value.startswith(("http://", "https://")) and any(
        marker in value.lower() for marker in (".jpg", ".jpeg", ".png", ".webp", "image")
    )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _merge_assets(assets: list[XhsAsset]) -> list[XhsAsset]:
    merged: list[XhsAsset] = []
    seen: set[str] = set()
    for asset in assets:
        key = asset.image_url
        if key in seen:
            continue
        seen.add(key)
        merged.append(asset)
    return merged


def _prepare_image_for_baidu(image_bytes: bytes) -> tuple[bytes, str]:
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")

        longest = max(image.size)
        if longest > 4096:
            scale = 4096 / longest
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            )

        quality = 92
        while quality >= 55:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            output = buffer.getvalue()
            if len(output) <= MAX_BAIDU_IMAGE_BYTES:
                return output, "image/jpeg"
            quality -= 10

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=45, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, "image/jpeg"


def _suffix_for_content_type(content_type: str | None) -> str:
    match (content_type or "").lower():
        case "image/png":
            return ".png"
        case "image/webp":
            return ".webp"
        case "image/bmp":
            return ".bmp"
        case _:
            return ".jpg"


def _find_system_browser() -> Path | None:
    configured = os.getenv("XHS_BROWSER_PATH")
    if configured:
        path = Path(configured)
        if path.exists():
            return path
        raise XhsIngestError(f"Configured XHS_BROWSER_PATH does not exist: {configured}")
    for path in CHROME_PATHS:
        if path.exists():
            return path
    return None


def _profile_is_locked_error(message: str) -> bool:
    lower = message.lower()
    return any(
        marker in lower
        for marker in (
            "user data directory is already in use",
            "browser closed",
            "targetclosederror",
            "processsingleton",
        )
    )
