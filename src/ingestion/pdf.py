import hashlib
import logging
import os
import re
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from ingestion.xhs import CHROMA_DIR, XHS_LOCAL_DIR, XhsArticle, XhsImporter
from schema import XhsIngestResponse

logger = logging.getLogger(__name__)

PDF_PLATFORM = "pdf"
PDF_DOCUMENT_DIR = Path(
    os.getenv("PDF_DOCUMENT_DIR", str(XHS_LOCAL_DIR / "pdf-documents"))
)
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "800"))
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "120"))
_PAGE_MARKER = re.compile(r"(?m)^【PDF 第 (\d+) 页】$")
_NUMBERED_HEADING = re.compile(
    r"^(?P<number>\d{1,2}(?:\s*\.\s*\d{1,3}){0,4})\s*(?P<title>[^。；，]{0,80})$"
)
_APPENDIX_HEADING = re.compile(r"^附录\s*[A-ZＡ-Ｚ](?:\s*\([^)]*\))?(?:\s+.*)?$")
_SPECIAL_HEADINGS = {"目次", "前言", "引言", "参考文献", "索引"}
_GBZ_185_PARTS = {
    1: "总体架构",
    2: "身份码",
    3: "身份管理",
    4: "智能体描述",
    5: "智能体发现",
    6: "智能体交互",
    7: "智能体工具调用",
}


def normalize_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts without flattening page structure."""

    text = (text or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=[\u3400-\u9fff])[ \t]+(?=[\u3400-\u9fff])", "", text)
    text = re.sub(r"[ \t]+([，。；：！？、）】])", r"\1", text)
    text = re.sub(r"([（【])[ \t]+", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_title(path: Path, first_page_text: str, metadata_title: str | None = None) -> str:
    """Return a stable, human-readable source title for citations."""

    part_match = re.search(r"GBZ\+185\.(\d+)-2026", path.name, re.IGNORECASE)
    if part_match:
        part = int(part_match.group(1))
        subtitle = _GBZ_185_PARTS.get(part)
        if subtitle:
            return f"GB/Z 185.{part}—2026《人工智能 智能体互联 第{part}部分：{subtitle}》"

    if metadata_title and metadata_title.strip():
        return normalize_pdf_text(metadata_title)

    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    return lines[0][:160] if lines else path.stem


def load_pdf_article(path: str | Path) -> XhsArticle:
    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    page_sections: list[str] = []
    first_page_text = ""
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = normalize_pdf_text(page.extract_text() or "")
        if page_number == 1:
            first_page_text = page_text
        if page_text:
            page_sections.append(f"【PDF 第 {page_number} 页】\n{page_text}")

    if not page_sections:
        raise ValueError(f"No searchable text found in PDF: {pdf_path.name}")

    source_uri = pdf_path.as_uri()
    note_hash = hashlib.sha256(str(pdf_path).casefold().encode("utf-8")).hexdigest()[:32]
    metadata_title = None
    if reader.metadata:
        metadata_title = reader.metadata.title

    return XhsArticle(
        note_key=f"pdf:{note_hash}",
        source_url=source_uri,
        canonical_url=source_uri,
        platform=PDF_PLATFORM,
        title=pdf_title(pdf_path, first_page_text, metadata_title),
        published_at=_published_date(first_page_text),
        body_text="\n\n".join(page_sections),
        tags=["国家标准", "人工智能", "智能体互联", "GB/Z 185"],
    )


class PdfImporter(XhsImporter):
    """Import searchable local PDFs into the existing PostgreSQL + Chroma knowledge base."""

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        document_dir: str | Path = PDF_DOCUMENT_DIR,
    ) -> None:
        super().__init__(chroma_dir=chroma_dir)
        self.document_dir = Path(document_dir)

    def import_path(self, path: str | Path, force_refresh: bool = False) -> XhsIngestResponse:
        try:
            article = load_pdf_article(path)
            stored_path = self._store_original(path)
            article.source_url = stored_path.as_uri()
            article.canonical_url = stored_path.as_uri()
            response = self.persist_article(article, force_refresh=force_refresh)
            response.message = "Imported PDF document."
            return response
        except Exception as exc:
            logger.exception("Failed to import PDF: %s", path)
            return XhsIngestResponse(
                status="failed",
                message=str(exc) or exc.__class__.__name__,
                source_url=str(Path(path).expanduser()),
            )

    def _store_original(self, path: str | Path) -> Path:
        source = Path(path).expanduser().resolve()
        self.document_dir.mkdir(parents=True, exist_ok=True)
        target = (self.document_dir / source.name).resolve()
        if source != target:
            shutil.copy2(source, target)
        return target

    def _split_article(self, article: XhsArticle) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=PDF_CHUNK_SIZE,
            chunk_overlap=PDF_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            keep_separator=True,
        )
        matches = list(_PAGE_MARKER.finditer(article.body_text))
        chunks: list[Document] = []
        active_section = ""

        for page_index, match in enumerate(matches):
            page_number = int(match.group(1))
            body_start = match.end()
            body_end = matches[page_index + 1].start() if page_index + 1 < len(matches) else None
            page_text = strip_pdf_page_furniture(article.body_text[body_start:body_end])
            if not page_text:
                continue
            for section_title, section_text in split_standard_sections(page_text):
                if section_title:
                    active_section = section_title
                effective_section = section_title or active_section
                if section_title and _heading_only(section_text):
                    continue
                for section_chunk in splitter.split_text(section_text):
                    prefix = [f"【PDF 第 {page_number} 页】"]
                    if effective_section:
                        prefix.append(f"【章节：{effective_section}】")
                    content = "\n".join([*prefix, section_chunk])
                    chunk_index = len(chunks)
                    chunks.append(
                        Document(
                            page_content=content,
                            metadata={
                                "chunk_id": f"{article.note_key}:{chunk_index}",
                                "source_platform": article.platform,
                                "source": article.canonical_url,
                                "source_url": article.canonical_url,
                                "title": article.title,
                                "note_key": article.note_key,
                                "chunk_index": chunk_index,
                                "page_number": page_number,
                                "section_title": effective_section,
                            },
                        )
                    )
        return chunks


def split_standard_sections(page_text: str) -> list[tuple[str, str]]:
    """Split one PDF page at Chinese standard section and clause headings."""

    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []
    index = 0

    def flush() -> None:
        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

    while index < len(lines):
        line = lines[index]
        heading = standard_heading(line)
        if heading:
            flush()
            current_lines = []
            current_title = heading
            if _NUMBERED_HEADING.fullmatch(line):
                match = _NUMBERED_HEADING.fullmatch(line)
                assert match is not None
                if not match.group("title").strip() and index + 1 < len(lines):
                    candidate = lines[index + 1]
                    if len(candidate) <= 80 and not candidate.endswith(("。", "；")):
                        current_title = f"{heading} {candidate}"
                        current_lines.extend([line, candidate])
                        index += 2
                        continue
            current_lines.append(line)
        else:
            current_lines.append(line)
        index += 1

    flush()
    return sections or [("", page_text)]


def standard_heading(line: str) -> str | None:
    compact = re.sub(r"[\s　]+", "", line)
    if compact in _SPECIAL_HEADINGS:
        return compact
    if _APPENDIX_HEADING.fullmatch(line):
        return re.sub(r"[\s　]+", " ", line).strip()
    match = _NUMBERED_HEADING.fullmatch(line)
    if not match:
        return None
    number = re.sub(r"\s*\.\s*", ".", match.group("number"))
    title = re.sub(r"[\s　]+", " ", match.group("title")).strip()
    if title.startswith(("）", ")", "、")) or (title and title[0].isdigit()):
        return None
    if not title and "." not in number:
        return None
    return f"{number} {title}".strip()


def strip_pdf_page_furniture(page_text: str) -> str:
    lines = []
    for line in page_text.strip().splitlines():
        stripped = line.strip()
        if re.fullmatch(r"GB\s*/\s*Z\s+185\s*\.\s*\d+\s*[—-]\s*2026", stripped):
            continue
        if re.fullmatch(r"(?:\d{1,3}|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]+)", stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _heading_only(section_text: str) -> bool:
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    return len(lines) == 1 and standard_heading(lines[0]) is not None


def _published_date(text: str) -> str | None:
    match = re.search(r"(20\d{2})\s*[-—]\s*(\d{1,2})\s*[-—]\s*(\d{1,2})\s*发布", text)
    if not match:
        return None
    year, month, day = (int(value) for value in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"
