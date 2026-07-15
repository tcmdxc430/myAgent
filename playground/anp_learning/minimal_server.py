from anp.openanp import AgentConfig, anp_agent, interface
from fastapi import FastAPI


@anp_agent(
    AgentConfig(
        name="My First ANP Agent",
        did="did:wba:example.com:learning-agent",
        prefix="/agent",
        description="用于学习 ANP 的最小 Agent",
    )
)
class LearningAgent:
    @interface
    async def hello(self, name: str) -> str:
        """向指定名称的用户返回一条中文问候语。"""
        return f"你好，{name}！这是来自 ANP Agent 的回答。"


app = FastAPI(title="ANP Learning")

app.include_router(LearningAgent.router())
