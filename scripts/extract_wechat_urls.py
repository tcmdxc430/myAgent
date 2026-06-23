"""Extract URLs shared in WeChat group exports into structured JSON.

The script accepts plain text, HTML, CSV/TSV, or JSON exports. It does not need
access to WeChat's local database; export or copy chat history into a file first.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

URL_RE = re.compile(r"(?P<url>(?:https?://|www\.)[^\s<>'\"\u3000]+)", re.IGNORECASE)
CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
XML_TAG_RE = re.compile(r"<(?P<tag>title|des|url)>(?P<value>.*?)</(?P=tag)>", re.DOTALL)
XML_BLOCK_RE = re.compile(r"<(?P<tag>msg|appmsg)\b.*?</(?P=tag)>", re.DOTALL | re.IGNORECASE)
HEADER_PATTERNS = [
    re.compile(
        r"^\s*\[(?P<timestamp>\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}.*?\d{1,2}:\d{2}(?::\d{2})?)\]\s*"
        r"(?P<sender>[^:：]{1,80})[:：]\s*(?P<message>.*)$"
    ),
    re.compile(
        r"^\s*(?P<timestamp>\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}.*?\d{1,2}:\d{2}(?::\d{2})?)\s+"
        r"(?P<sender>[^:：]{1,80})[:：]?\s*(?P<message>.*)$"
    ),
]
TRAILING_CHARS = " \t\r\n,.;:!?。，、；：！？)]}）】》>'\""
SENDER_KEYS = ("sender", "from", "nickname", "name", "user", "发送者", "昵称", "用户名")
TIME_KEYS = ("timestamp", "time", "datetime", "date", "created_at", "时间", "日期")
TEXT_KEYS = ("text", "content", "message", "body", "msg", "消息", "内容")


@dataclass
class SourceRecord:
    text: str
    line: int | None = None
    sender: str | None = None
    timestamp: str | None = None


def clean_xml_text(value: str | None) -> str | None:
    if not value:
        return None
    value = html.unescape(value)
    value = CDATA_RE.sub(lambda match: match.group(1), value)
    return re.sub(r"\s+", " ", value).strip() or None


def extract_xml_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in XML_TAG_RE.finditer(text):
        value = clean_xml_text(match.group("value"))
        if value:
            fields.setdefault(match.group("tag"), value)
    return fields


def strip_url(url: str) -> str:
    url = html.unescape(url).strip()
    url = CDATA_RE.sub(lambda match: match.group(1), url)
    while url and url[-1] in TRAILING_CHARS:
        url = url[:-1]
    if url.startswith("www."):
        url = f"https://{url}"
    return url


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""
    fragment = f"#{parsed.fragment}" if parsed.fragment else ""
    return f"{scheme}://{netloc}{path}{query}{fragment}"


def parse_header(line: str) -> tuple[str | None, str | None, str]:
    for pattern in HEADER_PATTERNS:
        match = pattern.match(line)
        if match:
            return (
                match.group("timestamp").strip(),
                match.group("sender").strip(),
                match.group("message").strip(),
            )
    return None, None, line


def find_urls(text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    expanded_text = html.unescape(text)
    for match in URL_RE.finditer(expanded_text):
        url = normalize_url(strip_url(match.group("url")))
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def best_field(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return str(value)
    return None


def flatten_json_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(flatten_json_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(flatten_json_strings(item))
        return strings
    return []


def records_from_json(value: Any) -> list[SourceRecord]:
    if isinstance(value, list):
        return [record for item in value for record in records_from_json(item)]
    if isinstance(value, dict):
        text = best_field(value, TEXT_KEYS)
        sender = best_field(value, SENDER_KEYS)
        timestamp = best_field(value, TIME_KEYS)
        if text:
            return [SourceRecord(text=text, sender=sender, timestamp=timestamp)]
        return [SourceRecord(text="\n".join(flatten_json_strings(value)))]
    if isinstance(value, str):
        return [SourceRecord(text=value)]
    return []


def records_from_delimited(path: Path, delimiter: str, encoding: str) -> list[SourceRecord]:
    with path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if not reader.fieldnames:
            return []
        records = []
        for index, row in enumerate(reader, start=2):
            text = best_field(row, TEXT_KEYS) or " ".join(str(value) for value in row.values())
            records.append(
                SourceRecord(
                    text=text,
                    line=index,
                    sender=best_field(row, SENDER_KEYS),
                    timestamp=best_field(row, TIME_KEYS),
                )
            )
        return records


def records_from_text(path: Path, encoding: str) -> list[SourceRecord]:
    text = path.read_text(encoding=encoding)
    records: list[SourceRecord] = []
    xml_line_ranges: list[range] = []

    for match in XML_BLOCK_RE.finditer(text):
        start_line = text[: match.start()].count("\n") + 1
        end_line = start_line + match.group(0).count("\n")
        xml_line_ranges.append(range(start_line, end_line + 1))
        records.append(SourceRecord(text=match.group(0), line=start_line))

    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(line_number in line_range for line_range in xml_line_ranges):
            continue
        timestamp, sender, message = parse_header(line)
        records.append(
            SourceRecord(
                text=message,
                line=line_number,
                sender=sender,
                timestamp=timestamp,
            )
        )
    records.sort(key=lambda record: record.line or 0)
    return records


def load_records(path: Path, encoding: str = "utf-8") -> list[SourceRecord]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return records_from_json(json.loads(path.read_text(encoding=encoding)))
    if suffix == ".csv":
        return records_from_delimited(path, ",", encoding)
    if suffix == ".tsv":
        return records_from_delimited(path, "\t", encoding)
    return records_from_text(path, encoding)


def extract_items(records: list[SourceRecord]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    unique: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for record in records:
        urls = find_urls(record.text)
        if not urls:
            continue

        xml_fields = extract_xml_fields(record.text)
        title = xml_fields.get("title")
        description = xml_fields.get("des")
        for url in urls:
            parsed = urlparse(url)
            item_id = len(items) + 1
            duplicate_of = unique[url]["first_item_id"] if url in unique else None
            item = {
                "id": item_id,
                "url": url,
                "domain": parsed.netloc,
                "scheme": parsed.scheme,
                "sender": record.sender,
                "timestamp": record.timestamp,
                "source_line": record.line,
                "title": title,
                "description": description,
                "message": record.text,
                "duplicate_of": duplicate_of,
            }
            items.append(item)

            if url not in unique:
                unique[url] = {
                    "url": url,
                    "domain": parsed.netloc,
                    "first_item_id": item_id,
                    "count": 0,
                    "item_ids": [],
                }
            unique[url]["count"] += 1
            unique[url]["item_ids"].append(item_id)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "stats": {
            "total_urls": len(items),
            "unique_urls": len(unique),
            "duplicate_urls": len(items) - len(unique),
        },
        "items": items,
        "unique_urls": list(unique.values()),
    }


def extract_wechat_urls(path: Path, encoding: str = "utf-8") -> dict[str, Any]:
    records = load_records(path, encoding)
    result = extract_items(records)
    result["source"] = str(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract URLs shared in WeChat group chat exports into structured JSON."
    )
    parser.add_argument("input", type=Path, help="Path to exported/copied WeChat chat file.")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path.")
    parser.add_argument("--encoding", default="utf-8", help="Input file encoding. Default: utf-8.")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = extract_wechat_urls(args.input, args.encoding)
    content = json.dumps(
        result,
        ensure_ascii=False,
        indent=None if args.compact else 2,
    )
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
