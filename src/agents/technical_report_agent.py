import asyncio
import logging
from datetime import datetime
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps

from agents.tools import database_search_func
from core import get_model, settings

logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    """State for the technical report writing agent."""

    remaining_steps: RemainingSteps
    kb_context: str
    retrieval_error: str


def _latest_user_query(state: AgentState) -> str:
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        return ""
    content: Any = human_messages[-1].content
    return content if isinstance(content, str) else str(content)


async def retrieve_report_context(state: AgentState, config: RunnableConfig) -> AgentState:
    """Retrieve supporting material from the imported URL/document knowledge base."""
    query = _latest_user_query(state)
    if not query.strip():
        return {"kb_context": "", "messages": []}

    try:
        context = await asyncio.to_thread(database_search_func, query)
        return {"kb_context": context.strip(), "messages": []}
    except Exception as exc:
        logger.exception("Failed to retrieve technical report context")
        return {
            "kb_context": "",
            "retrieval_error": str(exc),
            "messages": [],
        }


def _build_system_prompt(state: AgentState) -> str:
    current_date = datetime.now().strftime("%Y年%m月%d日")
    kb_context = state.get("kb_context", "").strip()
    retrieval_error = state.get("retrieval_error")

    source_block = (
        f"以下为从已导入网页 URL 或文档知识库检索到的支撑材料：\n\n{kb_context}"
        if kb_context
        else "本次未从知识库检索到可直接支撑报告正文的材料。"
    )
    if retrieval_error:
        source_block += f"\n\n知识库检索异常信息：{retrieval_error}"

    return f"""
你是“技术报告写作智能体”，专门基于已导入的网页 URL 或文档知识库材料，撰写面向政务、企事业单位、项目管理场景的技术报告。
今天日期是 {current_date}。

【知识库支撑要求】
1. 必须优先依据检索材料写作，关键事实、建设内容、进展成效、问题判断应能在材料中找到依据。
2. 涉及具体事实或判断时，在句末或段末用方括号标注来源，例如：[来源: Source 1 标题]。
3. 如果知识库材料不足，不得编造项目背景、进展、成效或数据；应先说明“知识库中暂未检索到足够支撑材料”，再给出可填写的规范化报告框架或建议补充材料清单。
4. 可以对材料进行归纳、提炼和公文式表达，但不得扩大事实边界。

【固定报告结构】
请按以下一级标题和顺序输出，标题文字必须完整保留：
一、工作背景与依据
二、总体目标
三、主要建设内容
四、工作进展及成效
五、存在问题
六、下一步工作计划
七、需要协调事项（如有）

【写作风格】
1. 政策合规、表达稳定、语气委婉、边界可控。
2. 多使用“贯彻落实”“统筹推进”“持续优化”“提升治理能力”“强化协同”等政策语言。
3. 避免过度技术化，不展开代码、算法、接口、工程细节；将技术动作转换为管理能力、业务支撑、数据共享、服务优化等表达。
4. 强调“成效 + 价值”，不仅说明完成事项，还要说明对管理、业务运行、协同效率和公众服务的积极意义。
5. 风险和问题表达要稳妥，优先使用“仍存在一定优化空间”“部分环节有待进一步完善”“协同机制需持续健全”等表述。
6. 如用户没有给出篇幅要求，输出一版完整但克制的正式报告草稿。

{source_block}
""".strip()


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    def create_system_message(state: AgentState):
        return [SystemMessage(content=_build_system_prompt(state))] + state["messages"]

    preprocessor = RunnableLambda(create_system_message, name="StateModifier")
    return RunnableSequence(preprocessor, model)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    model_runnable = wrap_model(get_model(model_name))
    response = await model_runnable.ainvoke(state, config)
    return {"messages": [response]}


agent = StateGraph(AgentState)
agent.add_node("retrieve_report_context", retrieve_report_context)
agent.add_node("model", acall_model)
agent.set_entry_point("retrieve_report_context")
agent.add_edge("retrieve_report_context", "model")
agent.add_edge("model", END)

technical_report_agent = agent.compile()
