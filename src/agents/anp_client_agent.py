import json
from pathlib import Path
from uuid import uuid4

from anp.authentication import DIDWbaAuthHeader
from anp.openanp import RemoteAgent
from langchain.agents import create_agent
from langchain_core.tools import tool

from core import get_model, settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREDENTIAL_DIR = PROJECT_ROOT / "privatecredentials" / "anp-learning"

RAG_AGENT_DESCRIPTION_URL = (
    "http://127.0.0.1:8080/anp/rag/ad.json"
)


@tool
async def query_anp_rag(question: str) -> str:
    """通过 ANP 协议查询远程企业知识库。

    当用户询问知识库、已导入文档、文章、报告或标准时，
    必须使用这个工具。
    """

    auth = DIDWbaAuthHeader(
        did_document_path=str(CREDENTIAL_DIR / "did.json"),
        private_key_path=str(CREDENTIAL_DIR / "key-1_private.pem"),
    )

    remote_agent = await RemoteAgent.discover(
        RAG_AGENT_DESCRIPTION_URL,
        auth,
    )

    result = await remote_agent.ask(
        question=question,
        thread_id=f"anp-tool-{uuid4()}",
    )

    if isinstance(result, dict) and "answer" in result:
        return result["answer"]

    return json.dumps(result, ensure_ascii=False)


SYSTEM_PROMPT = """
你是一个负责协调远程智能体的 ANP 协作助手。

工作规则：
1. 用户询问知识库、文档、文章、报告或标准时，必须调用 query_anp_rag。
2. 不得假装自己已经查询过知识库。
3. 调用工具后，根据工具返回结果回答用户。
4. 明确告诉用户，信息来自远程 ANP RAG Agent。
5. 始终使用中文回答。
"""


anp_client_agent = create_agent(
    model=get_model(settings.DEFAULT_MODEL),
    tools=[query_anp_rag],
    system_prompt=SYSTEM_PROMPT,
    name="anp-client-agent",
)