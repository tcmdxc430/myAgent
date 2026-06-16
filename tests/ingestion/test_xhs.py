import httpx
import pytest
from pydantic import SecretStr

from ingestion.xhs import (
    XhsArticle,
    XhsAsset,
    XhsImporter,
    _looks_like_image_url,
    extract_article_from_html,
    extract_note_key,
    normalize_url,
    normalize_xhs_url,
    platform_from_url,
)


def test_normalize_and_extract_note_key():
    url = normalize_xhs_url("https://www.xiaohongshu.com/explore/abc123/?xsec=1#hash")

    assert url == "https://www.xiaohongshu.com/explore/abc123?xsec=1"
    assert extract_note_key(url) == "abc123"


def test_generic_url_platform_and_article_extraction():
    url = normalize_url("https://example.com/post/hello/?utm=1#hash")
    assert url == "https://example.com/post/hello?utm=1"
    assert platform_from_url(url) == "example_com"

    html = """
    <html>
      <head>
        <meta property="og:title" content="通用文章标题">
        <meta name="description" content="这是一段足够长的通用网页正文内容，包含中文标点，用于导入知识库。">
        <link rel="canonical" href="https://example.com/post/hello">
      </head>
      <body><img src="https://example.com/image.jpg"></body>
    </html>
    """
    article = extract_article_from_html(
        html,
        final_url="https://example.com/post/hello",
        source_url=url,
        platform="example_com",
    )

    assert article.platform == "example_com"
    assert article.note_key
    assert article.title == "通用文章标题"
    assert "通用网页正文内容" in article.body_text


def test_wechat_article_dom_content_extraction():
    html = """
    <html>
      <head><title>页面标题</title></head>
      <body>
        <h1 id="activity-name">公众号文章标题</h1>
        <span id="js_name">公众号作者</span>
        <div id="js_content">
          这是微信公众号正文内容，包含足够多的中文段落，用于验证通用图文链接导入可以读取正文 DOM。
          这里继续补充内容，让正文长度超过解析阈值，并且保留中文标点。
        </div>
        <img data-src="https://mmbiz.qpic.cn/mmbiz_jpg/example/0?wx_fmt=jpeg">
      </body>
    </html>
    """

    article = extract_article_from_html(
        html,
        final_url="https://mp.weixin.qq.com/s/example",
        source_url="https://mp.weixin.qq.com/s/example",
        platform="mp_weixin_qq_com",
    )

    assert article.title == "页面标题"
    assert article.author == "公众号作者"
    assert "微信公众号正文内容" in article.body_text
    assert article.assets[0].image_url.startswith("https://mmbiz.qpic.cn/")


def test_extract_article_from_html():
    html = """
    <html>
      <head>
        <meta property="og:title" content="测试标题">
        <meta name="description" content="这是一段足够长的小红书正文内容，包含中文标点，也包含 #测试 标签。">
        <link rel="canonical" href="https://www.xiaohongshu.com/explore/note123">
      </head>
      <body><img src="https://example.com/image.jpg"></body>
    </html>
    """

    article = extract_article_from_html(
        html,
        final_url="https://www.xiaohongshu.com/explore/note123",
        source_url="https://xhslink.com/a/b",
    )

    assert article.note_key == "note123"
    assert article.title == "测试标题"
    assert "小红书正文内容" in article.body_text
    assert article.assets[0].image_url == "https://example.com/image.jpg"


def test_persist_article_builds_chroma_metadata(monkeypatch):
    importer = XhsImporter()
    article = XhsArticle(
        note_key="note123",
        source_url="https://xhslink.com/a/b",
        canonical_url="https://www.xiaohongshu.com/explore/note123",
        title="标题",
        body_text="这是一段用于切块的正文。" * 80,
        assets=[XhsAsset(image_url="https://example.com/image.jpg", ocr_text="图片文字", ocr_status="success")],
    )
    captured = {}

    monkeypatch.setattr(importer, "_ensure_tables", lambda: None)
    monkeypatch.setattr(importer, "_upsert_article", lambda article, force_refresh: "article-1")
    monkeypatch.setattr(importer, "_delete_existing_chunks", lambda article_id: None)

    def fake_write_chroma(article_id, article, chunks):
        captured["chunks"] = chunks
        return [f"chroma-{i}" for i, _ in enumerate(chunks)]

    monkeypatch.setattr(importer, "_write_chroma", fake_write_chroma)
    monkeypatch.setattr(
        importer,
        "_write_assets_and_chunks",
        lambda article_id, article, chunks, chroma_ids: captured.update(
            {"article_id": article_id, "chroma_ids": chroma_ids}
        ),
    )

    response = importer.persist_article(article, force_refresh=True)

    assert response.status == "success"
    assert response.article_id == "article-1"
    assert response.chunk_count == len(captured["chunks"])
    assert captured["chunks"][0].metadata["source_platform"] == "xiaohongshu"
    assert captured["chunks"][0].metadata["source_url"] == article.canonical_url
    assert captured["chunks"][0].metadata["article_id"] == "article-1"


def test_image_url_filter_skips_xhs_avatars():
    assert not _looks_like_image_url("https://sns-avatar-qc.xhscdn.com/avatar/a?imageView2")
    assert _looks_like_image_url("https://ci.xiaohongshu.com/image/abc")


@pytest.mark.asyncio
async def test_baidu_ocr_api_key_auth_response_words_are_joined():
    importer = XhsImporter()
    captured = {}

    def handle_request(request):
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"words_result": [{"words": "第一行"}, {"words": "second line"}]},
            request=request,
        )

    transport = httpx.MockTransport(handle_request)
    async with httpx.AsyncClient(transport=transport) as client:
        text = await importer._ocr_image(
            client, b"fake-image", {"mode": "api_key", "credential": "api-key"}
        )

    assert text == "第一行\nsecond line"
    assert captured["authorization"] == "Bearer api-key"


@pytest.mark.asyncio
async def test_baidu_ocr_access_token_response_words_are_joined():
    importer = XhsImporter()
    captured = {}

    def handle_request(request):
        captured["query"] = str(request.url)
        return httpx.Response(
            200,
            json={"words_result": [{"words": "第一行"}, {"words": "second line"}]},
            request=request,
        )

    transport = httpx.MockTransport(handle_request)
    async with httpx.AsyncClient(transport=transport) as client:
        text = await importer._ocr_image(client, b"fake-image", "token")

    assert text == "第一行\nsecond line"
    assert "access_token=token" in captured["query"]


@pytest.mark.asyncio
async def test_baidu_token_failure_marks_assets_failed(monkeypatch):
    importer = XhsImporter()
    article = XhsArticle(
        note_key="note123",
        source_url="https://xhslink.com/a/b",
        canonical_url="https://www.xiaohongshu.com/explore/note123",
        title="标题",
        assets=[XhsAsset(image_url="screenshot://xiaohongshu/main/0", image_bytes=b"fake")],
    )
    written_assets = []

    monkeypatch.setattr("ingestion.xhs.settings.BAIDU_OCR_API_KEY", SecretStr("bad-key"))
    monkeypatch.setattr("ingestion.xhs.settings.BAIDU_OCR_SECRET_KEY", SecretStr("bad-secret"))
    monkeypatch.setattr(
        "ingestion.xhs._prepare_image_for_baidu",
        lambda image_bytes: (image_bytes, "image/jpeg"),
    )
    monkeypatch.setattr(
        importer,
        "_write_asset",
        lambda image_url, content, content_type=None: written_assets.append(
            (image_url, content, content_type)
        )
        or "cached.jpg",
    )

    async def fail_auths(client):
        raise RuntimeError("Baidu OCR token request failed: invalid_client")

    monkeypatch.setattr(importer, "_get_baidu_ocr_auths", fail_auths)

    await importer.enrich_assets_with_ocr(article)

    assert article.assets[0].ocr_status == "failed"
    assert "invalid_client" in article.assets[0].ocr_error
    assert article.assets[0].local_path == "cached.jpg"
    assert written_assets[0][0] == "screenshot://xiaohongshu/main/0"
