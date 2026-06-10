"""
用 browser-use 对 myAgent Streamlit 前端做 E2E 冒烟测试。

运行前请先启动服务（两个终端）：
  1. python src/run_service.py          # 后端 http://localhost:8080
  2. streamlit run src/streamlit_app.py # 前端 http://localhost:8501

然后执行：
  python playground/browser_use_test/02_test_myagent.py
"""

import asyncio
import os
import sys
from pathlib import Path

from browser_use import Agent
from dotenv import load_dotenv

from llm_config import make_llm

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

STREAMLIT_URL = os.getenv("STREAMLIT_URL", "http://localhost:8501")
AGENT_ID = os.getenv("BROWSER_TEST_AGENT", "research-assistant")


def build_task() -> str:
    return f"""
测试 myAgent Streamlit 聊天应用，按顺序执行：

1. 打开 {STREAMLIT_URL}
2. 等待页面完全加载（出现侧边栏和聊天区域）
3. 在左侧边栏点击「设置」展开配置（可能在「:material/settings: 设置」按钮里）
4. 在「使用的智能体」下拉框中选择：{AGENT_ID}
5. 在页面底部聊天输入框输入：1+1等于几？请只回答数字。
6. 发送消息（按 Enter 或点击发送）
7. 等待助手回复出现（可能需要 10～60 秒，请耐心等待流式输出结束）
8. 检查回复中是否包含数字 2
9. 最后一句话必须写「测试通过」或「测试失败」，并简要说明原因
"""


def assert_passed(final_text: str | None) -> None:
    text = (final_text or "").strip()
    if not text:
        print("❌ 断言失败：无最终结果")
        sys.exit(1)
    if "测试失败" in text:
        print(f"❌ 断言失败：{text}")
        sys.exit(1)
    if "测试通过" not in text and "通过" not in text:
        print(f"❌ 断言失败：结果未包含「通过」→ {text}")
        sys.exit(1)
    print(f"✅ 断言通过：{text}")


async def main() -> None:
    print(f"目标页面: {STREAMLIT_URL}")
    print(f"测试智能体: {AGENT_ID}")
    print("请确认 run_service.py 与 streamlit 已启动。\n")

    agent = Agent(task=build_task(), llm=make_llm())
    result = await agent.run()

    final = result.final_result()
    print("\n=== 测试结果 ===")
    print(final)
    assert_passed(final)


if __name__ == "__main__":
    asyncio.run(main())
