from pathlib import Path

from langchain_core.documents import Document

from agents.tools import _is_catalog_query, _merge_retrieved_documents, _query_terms
from ingestion.pdf import (
    PdfImporter,
    normalize_pdf_text,
    pdf_title,
    split_standard_sections,
    standard_heading,
    strip_pdf_page_furniture,
)
from ingestion.xhs import XhsArticle


def test_normalize_pdf_text_removes_spacing_between_chinese_characters():
    assert normalize_pdf_text("人 工 智 能  \n\n\n第 1 部 分") == "人工智能\n\n第 1 部分"


def test_pdf_title_recognizes_gbz_185_series():
    title = pdf_title(Path("GBZ+185.7-2026.pdf"), "")

    assert title == "GB/Z 185.7—2026《人工智能 智能体互联 第7部分：智能体工具调用》"


def test_pdf_chunks_keep_page_number_and_source_metadata():
    article = XhsArticle(
        note_key="pdf:test",
        source_url="file:///C:/standards/test.pdf",
        canonical_url="file:///C:/standards/test.pdf",
        platform="pdf",
        title="测试标准",
        body_text="【PDF 第 2 页】\n第一页正文\n\n【PDF 第 5 页】\n第二页正文",
    )

    chunks = PdfImporter()._split_article(article)

    assert [chunk.metadata["page_number"] for chunk in chunks] == [2, 5]
    assert chunks[0].page_content.startswith("【PDF 第 2 页】")
    assert chunks[1].metadata["title"] == "测试标准"


def test_pdf_chunks_keep_standard_clause_heading():
    article = XhsArticle(
        note_key="pdf:test",
        source_url="file:///C:/standards/test.pdf",
        canonical_url="file:///C:/standards/test.pdf",
        platform="pdf",
        title="测试标准",
        body_text=(
            "【PDF 第 8 页】\n6 . 1  基于智能体发现服务的发现流程\n"
            "智能体描述管理方应提供智能体发现服务。"
        ),
    )

    chunks = PdfImporter()._split_article(article)

    assert chunks[0].metadata["section_title"] == "6.1 基于智能体发现服务的发现流程"
    assert "【章节：6.1 基于智能体发现服务的发现流程】" in chunks[0].page_content


def test_split_standard_sections_handles_bare_clause_number():
    sections = split_standard_sections("3 . 1\n智能体发现\n匹配并获得符合要求的智能体描述的过程。")

    assert sections == [("3.1 智能体发现", "3 . 1\n智能体发现\n匹配并获得符合要求的智能体描述的过程。")]


def test_standard_heading_rejects_release_date_and_numbered_list():
    assert standard_heading("20 26 ⁃ 05 ⁃ 22 发布") is None
    assert standard_heading("1）发现条件可由智能体生成") is None


def test_strip_pdf_page_furniture_removes_running_header_and_page_number():
    text = "GB/Z 185 . 5 — 2026\n6 . 1 发现流程\n正文\n3"

    assert strip_pdf_page_furniture(text) == "6 . 1 发现流程\n正文"


def test_chinese_query_terms_keep_standard_specific_phrases():
    identity_terms = _query_terms("智能体身份码的编码结构和分配原则是什么")
    discovery_terms = _query_terms("智能体发现流程")

    assert "身份码" in identity_terms
    assert "编码结构" in identity_terms
    assert "分配原则" in identity_terms
    assert "智能体发现" in discovery_terms


def test_catalog_query_extracts_core_phrase():
    query = "关于智能体互联国标 你了解什么"

    assert "智能体互联" in _query_terms(query)
    assert _is_catalog_query(query) is True
    assert _is_catalog_query("智能体互联国标中身份码如何编码") is False


def test_catalog_merge_prioritizes_distinct_articles():
    docs = [
        Document(page_content="a1", metadata={"chunk_id": "a1", "article_id": "a"}),
        Document(page_content="a2", metadata={"chunk_id": "a2", "article_id": "a"}),
        Document(page_content="b1", metadata={"chunk_id": "b1", "article_id": "b"}),
    ]

    merged = _merge_retrieved_documents(docs, limit=3, diversify_articles=True)

    assert [doc.metadata["chunk_id"] for doc in merged] == ["a1", "b1", "a2"]


def test_pdf_importer_stores_original_in_persistent_directory(tmp_path):
    source = tmp_path / "source" / "standard.pdf"
    source.parent.mkdir()
    source.write_bytes(b"pdf source bytes")
    importer = PdfImporter(document_dir=tmp_path / "knowledge-base")

    stored = importer._store_original(source)

    assert stored.read_bytes() == b"pdf source bytes"
    assert stored.parent == (tmp_path / "knowledge-base").resolve()
