"""browser-use 测试共用的 LLM 配置。"""

import os

from browser_use import ChatGoogle, ChatOpenAI
from browser_use.llm.base import BaseChatModel


def make_llm() -> BaseChatModel:
    provider = os.getenv("BROWSER_USE_LLM", "gemini").lower()

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("请在 .env 中配置 DEEPSEEK_API_KEY")
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            dont_force_structured_output=True,
            add_schema_to_system_prompt=True,
        )

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("请在 .env 中配置 GOOGLE_API_KEY，或设置 BROWSER_USE_LLM=deepseek")
    return ChatGoogle(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        api_key=api_key,
        temperature=0.2,
    )
