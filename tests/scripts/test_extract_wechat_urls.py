import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_wechat_urls.py"
SPEC = importlib.util.spec_from_file_location("extract_wechat_urls", SCRIPT_PATH)
extract_wechat_urls = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["extract_wechat_urls"] = extract_wechat_urls
SPEC.loader.exec_module(extract_wechat_urls)


def test_extract_text_urls_with_sender_timestamp_and_duplicates(tmp_path):
    chat_file = tmp_path / "wechat.txt"
    chat_file.write_text(
        "\n".join(
            [
                "[2026-06-16 09:30:00] 张三: 推荐 https://example.com/Post?a=1。",
                "2026-06-16 09:31 李四: 再发一次 https://example.com/Post?a=1",
                "普通消息 www.example.org/path,",
            ]
        ),
        encoding="utf-8",
    )

    result = extract_wechat_urls.extract_wechat_urls(chat_file)

    assert result["stats"] == {"total_urls": 3, "unique_urls": 2, "duplicate_urls": 1}
    assert result["items"][0]["sender"] == "张三"
    assert result["items"][0]["timestamp"] == "2026-06-16 09:30:00"
    assert result["items"][0]["url"] == "https://example.com/Post?a=1"
    assert result["items"][1]["duplicate_of"] == 1
    assert result["items"][2]["url"] == "https://www.example.org/path"


def test_extract_wechat_xml_title_description_and_escaped_url(tmp_path):
    chat_file = tmp_path / "wechat.txt"
    chat_file.write_text(
        """
        <msg>
          <title><![CDATA[文章标题]]></title>
          <des><![CDATA[文章摘要]]></des>
          <url><![CDATA[https://mp.weixin.qq.com/s/a?x=1&amp;y=2]]></url>
        </msg>
        """,
        encoding="utf-8",
    )

    result = extract_wechat_urls.extract_wechat_urls(chat_file)

    assert result["stats"]["unique_urls"] == 1
    assert result["items"][0]["url"] == "https://mp.weixin.qq.com/s/a?x=1&y=2"
    assert result["items"][0]["title"] == "文章标题"
    assert result["items"][0]["description"] == "文章摘要"


def test_extract_json_rows(tmp_path):
    chat_file = tmp_path / "wechat.json"
    chat_file.write_text(
        """
        [
          {"sender": "王五", "time": "2026-06-16 10:00", "content": "看这个 https://a.test/1"},
          {"发送者": "赵六", "时间": "2026-06-16 10:01", "消息": "还有 https://b.test/2"}
        ]
        """,
        encoding="utf-8",
    )

    result = extract_wechat_urls.extract_wechat_urls(chat_file)

    assert [item["sender"] for item in result["items"]] == ["王五", "赵六"]
    assert [item["domain"] for item in result["items"]] == ["a.test", "b.test"]
