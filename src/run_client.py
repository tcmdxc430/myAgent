import asyncio

from client import AgentClient
from core import settings
from schema import ChatMessage


async def amain() -> None:
    #### 异步 ####
    client = AgentClient(settings.BASE_URL)

    print("智能体信息:")
    print(client.info)

    print("聊天示例:")
    response = await client.ainvoke("给我讲个笑话？", model="gpt-5-nano")
    response.pretty_print()

    print("\n流式输出示例:")
    async for message in client.astream("分享一个有趣的冷知识？"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"错误: 未知类型 - {type(message)}")


def main() -> None:
    #### 同步 ####
    client = AgentClient(settings.BASE_URL)

    print("智能体信息:")
    print(client.info)

    print("聊天示例:")
    response = client.invoke("给我讲个笑话？", model="gpt-5-nano")
    response.pretty_print()

    print("\n流式输出示例:")
    for message in client.stream("分享一个有趣的冷知识？"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"错误: 未知类型 - {type(message)}")


if __name__ == "__main__":
    print("正在以同步模式运行")
    main()
    print("\n\n\n\n\n")
    print("正在以异步模式运行")
    asyncio.run(amain())
