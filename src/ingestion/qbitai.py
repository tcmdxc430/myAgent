import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup

from ingestion.xhs import ArticleImporter, XhsArticle, extract_article_from_html, normalize_url
from schema import QbitaiHotNewsImportItem, QbitaiHotNewsImportResponse

logger = logging.getLogger(__name__)

QBITAI_HOT_NEWS_URL = "https://www.qbitai.com/category/%e8%b5%84%e8%ae%af"
QBITAI_PLATFORM = "qbitai_hot_news"
QBITAI_TIMEOUT = 30
QBITAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class QbitaiHotNewsItem:
    rank: int
    title: str
    url: str
    published_at: str | None = None


class QbitaiHotNewsImporter(ArticleImporter):
    """Import QbitAI hot news into the existing RAG knowledge base."""

    async def import_hot_news(
        self,
        limit: int = 3,
        force_refresh: bool = True,
        source_url: str = QBITAI_HOT_NEWS_URL,
    ) -> QbitaiHotNewsImportResponse:
        try:
            candidates = await self.fetch_hot_news(source_url=source_url, limit=limit)
        except Exception as exc:
            logger.exception("Failed to fetch QbitAI hot news list")
            return QbitaiHotNewsImportResponse(
                status="failed",
                message=str(exc) or exc.__class__.__name__,
                source_url=source_url,
                item_count=0,
                imported_count=0,
                items=[],
            )

        results = await asyncio.gather(
            *(self._import_hot_item(item, force_refresh=force_refresh) for item in candidates),
            return_exceptions=True,
        )
        items: list[QbitaiHotNewsImportItem] = []
        for candidate, result in zip(candidates, results, strict=True):
            if isinstance(result, Exception):
                logger.error("Failed to import QbitAI hot news item: %s: %s", candidate.url, result)
                items.append(
                    QbitaiHotNewsImportItem(
                        rank=candidate.rank,
                        title=candidate.title,
                        url=candidate.url,
                        published_at=candidate.published_at,
                        status="failed",
                        message=str(result) or result.__class__.__name__,
                    )
                )
                continue
            items.append(result)

        imported_count = sum(1 for item in items if item.status in {"success", "partial_success"})
        if imported_count == len(items) and items:
            status = "success"
            message = f"Imported {imported_count} QbitAI hot news items."
        elif imported_count:
            status = "partial_success"
            message = f"Imported {imported_count} of {len(items)} QbitAI hot news items."
        else:
            status = "failed"
            message = "No QbitAI hot news items were imported."

        return QbitaiHotNewsImportResponse(
            status=status,
            message=message,
            source_url=source_url,
            item_count=len(items),
            imported_count=imported_count,
            items=items,
        )

    async def fetch_hot_news(
        self,
        source_url: str = QBITAI_HOT_NEWS_URL,
        limit: int = 3,
    ) -> list[QbitaiHotNewsItem]:
        normalized_url = normalize_url(source_url)
        async with httpx.AsyncClient(timeout=QBITAI_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(
                normalized_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()
        items = extract_qbitai_hot_news_items(response.text, str(response.url), limit=limit)
        if not items:
            raise ValueError("No QbitAI hot news items found on the source page")
        return items

    async def _import_hot_item(
        self,
        item: QbitaiHotNewsItem,
        force_refresh: bool,
    ) -> QbitaiHotNewsImportItem:
        try:
            article = await self._fetch_qbitai_article(item)
            response = self.persist_article(article, force_refresh=force_refresh)
            return QbitaiHotNewsImportItem(
                rank=item.rank,
                title=response.title or item.title,
                url=response.source_url or item.url,
                published_at=item.published_at,
                status=response.status,
                message=response.message,
                article_id=response.article_id,
                chunk_count=response.chunk_count,
            )
        except Exception as exc:
            logger.exception("Failed to import QbitAI hot news article: %s", item.url)
            return QbitaiHotNewsImportItem(
                rank=item.rank,
                title=item.title,
                url=item.url,
                published_at=item.published_at,
                status="failed",
                message=str(exc) or exc.__class__.__name__,
            )

    async def _fetch_qbitai_article(self, item: QbitaiHotNewsItem) -> XhsArticle:
        async with httpx.AsyncClient(timeout=QBITAI_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(
                item.url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()

        article = extract_article_from_html(
            response.text,
            final_url=str(response.url),
            source_url=item.url,
            platform=QBITAI_PLATFORM,
        )
        article.title = article.title or item.title
        article.published_at = article.published_at or item.published_at
        article.tags = _merge_tags(
            article.tags,
            [
                "量子位",
                "资讯",
                "热门文章",
                f"hot_rank_{item.rank}",
                f"hot_snapshot_{_snapshot_date()}",
            ],
        )
        article.body_text = _prepend_structured_metadata(article, item)
        return article


def extract_qbitai_hot_news_items(
    html: str,
    base_url: str = QBITAI_HOT_NEWS_URL,
    limit: int = 3,
) -> list[QbitaiHotNewsItem]:
    soup = BeautifulSoup(html, "html.parser")
    items = _extract_hot_sidebar_items(soup, base_url)
    if not items:
        items = _extract_category_list_items(soup, base_url)
    return items[:limit]


def _extract_hot_sidebar_items(soup: BeautifulSoup, base_url: str) -> list[QbitaiHotNewsItem]:
    items: list[QbitaiHotNewsItem] = []
    seen: set[str] = set()
    for index, anchor in enumerate(soup.select(".content_right .yaowen .picture_text > a"), start=1):
        href = anchor.get("href")
        title_node = anchor.select_one(".text_box h4") or anchor.select_one("h4")
        title = title_node.get_text(" ", strip=True) if title_node else ""
        if not href or not title:
            continue
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        published_node = anchor.select_one(".info")
        items.append(
            QbitaiHotNewsItem(
                rank=len(items) + 1,
                title=title,
                url=url,
                published_at=published_node.get_text(" ", strip=True) if published_node else None,
            )
        )
    return items


def _extract_category_list_items(soup: BeautifulSoup, base_url: str) -> list[QbitaiHotNewsItem]:
    items: list[QbitaiHotNewsItem] = []
    seen: set[str] = set()
    for block in soup.select(".article_list .picture_text"):
        anchor = block.select_one(".text_box h4 a") or block.select_one("h4 a")
        if not anchor:
            continue
        href = anchor.get("href")
        title = anchor.get_text(" ", strip=True)
        if not href or not title:
            continue
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        published_node = block.select_one(".info .time")
        items.append(
            QbitaiHotNewsItem(
                rank=len(items) + 1,
                title=title,
                url=url,
                published_at=published_node.get_text(" ", strip=True) if published_node else None,
            )
        )
    return items


def _merge_tags(existing: list[str], additions: list[str]) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for tag in existing + additions:
        cleaned = str(tag).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        tags.append(cleaned)
    return tags


def _prepend_structured_metadata(article: XhsArticle, item: QbitaiHotNewsItem) -> str:
    metadata = "\n".join(
        [
            "结构化元数据：",
            "来源平台：量子位",
            "栏目：资讯",
            "榜单类型：热门文章",
            f"热点排名：{item.rank}",
            f"榜单标题：{item.title}",
            f"榜单日期：{_snapshot_date()}",
            f"发布时间：{item.published_at or article.published_at or '未知'}",
            f"原文链接：{item.url}",
        ]
    )
    return f"{metadata}\n\n{article.body_text}".strip()


def _snapshot_date() -> str:
    return datetime.now(QBITAI_TIMEZONE).date().isoformat()
