from typing import Any

from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from agents.reflective_agent import reflective_agent

from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


math_agent: Any = create_agent(
    model=model,
    tools=[add, multiply],
    name="sub-agent-math_expert",
    system_prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])

research_agent: Any = create_agent(
    model=model,
    tools=[web_search],
    name="sub-agent-research_expert",
    system_prompt="You are a world class researcher with access to web search. Do not do any math.",
).with_config(tags=["skip_stream"])


# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent, reflective_agent], # 将 reflective_agent 加入团队
    model=model,
    prompt=(
        "你是一个团队管理者。你有三名专家：\n"
        "1. research_agent: 负责网页搜索。\n"
        "2. math_expert: 负责数学计算。\n"
        "3. reflective_agent: 负责高质量的文案创作和深度评论，它会自动进行自我反思和纠错。\n"
        "请根据用户需求选择合适的专家。"
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
)

langgraph_supervisor_agent = workflow.compile()
