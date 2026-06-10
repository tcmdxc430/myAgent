import asyncio
from pathlib import Path

from browser_use import Agent
from dotenv import load_dotenv

from llm_config import make_llm

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


async def main():
    agent = Agent(
        task="打开 https://example.com ，读取页面标题，用中文告诉我标题是什么",
        llm=make_llm(),
    )
    result = await agent.run()
    print(result.final_result())


asyncio.run(main())
