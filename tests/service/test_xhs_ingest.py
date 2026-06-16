from unittest.mock import AsyncMock, patch

from schema import XhsIngestResponse
from schema.schema import QbitaiHotNewsImportResponse


def test_ingest_xhs_endpoint(test_client):
    response_payload = XhsIngestResponse(
        status="success",
        message="Imported",
        article_id="article-1",
        title="测试标题",
        source_url="https://www.xiaohongshu.com/explore/note123",
        chunk_count=2,
        asset_count=1,
    )

    with patch("service.service.xhs_importer.import_url", new=AsyncMock(return_value=response_payload)):
        response = test_client.post(
            "/ingest/xhs",
            json={"url": "https://www.xiaohongshu.com/explore/note123", "force_refresh": True},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["chunk_count"] == 2


def test_ingest_article_endpoint(test_client):
    response_payload = XhsIngestResponse(
        status="success",
        message="Imported",
        article_id="article-2",
        title="Generic Article",
        source_url="https://example.com/post/hello",
        chunk_count=1,
        asset_count=1,
    )

    with patch(
        "service.service.article_importer.import_url",
        new=AsyncMock(return_value=response_payload),
    ):
        response = test_client.post(
            "/ingest/article",
            json={"url": "https://example.com/post/hello", "force_refresh": True},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["source_url"] == "https://example.com/post/hello"


def test_xhs_login_endpoint(test_client):
    with patch("service.service.open_xhs_login_window", new=AsyncMock()):
        response = test_client.post("/ingest/xhs/login")

    assert response.status_code == 200
    assert response.json()["status"] == "started"


def test_ingest_qbitai_hot_news_endpoint(test_client):
    response_payload = QbitaiHotNewsImportResponse(
        status="success",
        message="Imported 3 QbitAI hot news items.",
        source_url="https://www.qbitai.com/category/%e8%b5%84%e8%ae%af",
        item_count=3,
        imported_count=3,
        items=[],
    )

    with patch(
        "service.service.qbitai_hot_news_importer.import_hot_news",
        new=AsyncMock(return_value=response_payload),
    ) as mock_import:
        response = test_client.post("/ingest/qbitai/hot-news")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["imported_count"] == 3
    mock_import.assert_awaited_once()
