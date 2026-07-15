import asyncio
import json
from pathlib import Path

from anp.authentication import DIDWbaAuthHeader
from anp.openanp import RemoteAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREDENTIAL_DIR = PROJECT_ROOT / "privatecredentials" / "anp-learning"


async def main() -> None:
    auth = DIDWbaAuthHeader(
        did_document_path=str(CREDENTIAL_DIR / "did.json"),
        private_key_path=str(CREDENTIAL_DIR / "key-1_private.pem"),
    )

    print("正在发现远程 Agent……")

    agent = await RemoteAgent.discover(
        "http://127.0.0.1:8080/anp/rag/ad.json",
        auth,
    )

    print(f"Agent 名称：{agent.name}")
    print(f"Agent 描述：{agent.description}")
    print(f"发现的方法：{[method.name for method in agent.methods]}")

    result = await agent.ask(
        question="你是谁？",
        thread_id="anp-python-client-001",
    )

    print("\n调用结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())