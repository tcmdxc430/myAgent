from pathlib import Path

from anp.authentication import DIDWbaAuthHeader
from anp.openanp import RemoteAgent
from langchain.agents import create_agent
from langchain_core.tools import tool

from core import get_model, settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREDENTIAL_DIR = PROJECT_ROOT / "privatecredentials" / "anp-learning"

EXTERNAL_AGENT_URL = "http://127.0.0.1:8090/agent/ad.json"


@tool
async def greet_via_external_anp(name: str) -> str:
    """调用独立运行的外部 ANP Agent 向指定用户问好。"""

    auth = DIDWbaAuthHeader(
        did_document_path=str(CREDENTIAL_DIR / "did.json"),
        private_key_path=str(CREDENTIAL_DIR / "key-1_private.pem"),
    )

    external_agent = await RemoteAgent.discover(
        EXTERNAL_AGENT_URL,
        auth,
    )

    result = await external_agent.hello(name=name)

    return str(result)


SYSTEM_PROMPT = """
你是 myAgent 中的 ANP 协作助手。

规则：
1. 当用户要求调用外部 Agent 问好时，必须调用 greet_via_external_anp。
2. 不得自己编造外部 Agent 的回答。
3. 工具调用完成后，明确说明结果来自独立的 8090 ANP Agent。
4. 始终使用中文回答。
"""


external_anp_client_agent = create_agent(
    model=get_model(settings.DEFAULT_MODEL),
    tools=[greet_via_external_anp],
    system_prompt=SYSTEM_PROMPT,
    name="external-anp-client-agent",
)