from uuid import uuid4

from anp.openanp import AgentConfig, anp_agent, interface
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents import get_agent
from service.utils import convert_message_content_to_string


@anp_agent(
    AgentConfig(
        name="myAgent RAG Assistant",
        did="did:wba:example.com:rag-assistant",
        prefix="/anp/rag",
        description="基于 myAgent 知识库回答问题的 RAG 智能体",
    )
)
class RagAnpAgent:
    @interface
    async def ask(
        self,
        question: str,
        thread_id: str = "",
    ) -> dict[str, str]:
        """向 myAgent 知识库提问。"""

        actual_thread_id = thread_id or str(uuid4())

        agent = get_agent("rag-assistant")

        config = RunnableConfig(
            configurable={
                "thread_id": actual_thread_id,
                "user_id": "anp-learning-user",
            }
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=question),
                ]
            },
            config=config,
        )

        last_message = result["messages"][-1]
        answer = convert_message_content_to_string(last_message.content)

        return {
            "answer": answer,
            "thread_id": actual_thread_id,
        }


anp_router = RagAnpAgent.router()
