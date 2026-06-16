from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.chinese_assistant import chinese_assistant
from agents.command_agent import command_agent
from agents.data_analyst_agent import data_analyst_agent
from agents.github_mcp_agent.github_mcp_agent import github_mcp_agent
from agents.interrupt_agent import interrupt_agent
from agents.knowledge_base_agent import kb_agent
from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.langgraph_supervisor_hierarchy_agent import langgraph_supervisor_hierarchy_agent
from agents.lazy_agent import LazyLoadingAgent
from agents.rag_assistant import rag_assistant
from agents.reflective_agent import reflective_agent
from agents.research_assistant import research_assistant
from agents.sql_assistant import sql_assistant
from agents.structured_output_agent import structured_output_agent
from agents.technical_report_agent import technical_report_agent
from schema import AgentInfo

DEFAULT_AGENT = "research-assistant"

# 用于处理不同 LangGraph 智能体模式的类型别名
# - @entrypoint 函数返回 Pregel
# - StateGraph().compile() returns CompiledStateGraph
AgentGraph = CompiledStateGraph | Pregel  # get_agent() 返回的内容（始终已加载）
AgentGraphLike = CompiledStateGraph | Pregel | LazyLoadingAgent  # 可以存储在注册表中的内容


@dataclass
class Agent:
    description: str
    graph_like: AgentGraphLike


agents: dict[str, Agent] = {
    "chatbot": Agent(description="一个简单的聊天机器人。", graph_like=chatbot),
    "research-assistant": Agent(
        description="一个带有网页搜索和计算器功能的研究助手。",
        graph_like=research_assistant,
    ),
    "rag-assistant": Agent(
        description="一个可以访问数据库信息的 RAG 助手。",
        graph_like=rag_assistant,
    ),
    "sql-assistant": Agent(
        description="一个专业的 SQL 查询助手，可以查询业务数据库。",
        graph_like=sql_assistant,
    ),
    "command-agent": Agent(description="一个命令智能体。", graph_like=command_agent),
    "bg-task-agent": Agent(description="一个后台任务智能体。", graph_like=bg_task_agent),
    "langgraph-supervisor-agent": Agent(
        description="一个 LangGraph 管理者智能体", graph_like=langgraph_supervisor_agent
    ),
    "langgraph-supervisor-hierarchy-agent": Agent(
        description="一个具有嵌套层级结构的 LangGraph 管理者智能体",
        graph_like=langgraph_supervisor_hierarchy_agent,
    ),
    "interrupt-agent": Agent(
        description="一个使用中断机制的智能体。", graph_like=interrupt_agent
    ),
    "knowledge-base-agent": Agent(
        description="一个使用 Amazon Bedrock 知识库的检索增强生成智能体",
        graph_like=kb_agent,
    ),
    "github-mcp-agent": Agent(
        description="一个带有 MCP 工具的 GitHub 智能体，用于仓库管理和开发工作流。",
        graph_like=github_mcp_agent,
    ),
    "chinese-assistant": Agent(
        description="一个简洁、务实的中文助手。", graph_like=chinese_assistant
    ),
    "structured-output-agent": Agent(
        description="一个演示如何输出结构化 JSON 的智能体。",
        graph_like=structured_output_agent
    ),
    "reflective-agent": Agent(
        description="一个会自动进行自我反思和纠错的写作专家。", 
        graph_like=reflective_agent
    ),
    "data-analyst-assistant": Agent(
        description="一个随身数据分析科学家，支持上传 CSV 文件进行自动清洗、Python 数据分析和图表绘制。",
        graph_like=data_analyst_agent,
    ),
    "technical-report-agent": Agent(
        description="一个基于已导入网页 URL 或文档知识库撰写政策合规型技术报告的智能体。",
        graph_like=technical_report_agent,
    ),
}


async def load_agent(agent_id: str) -> None:
    """如果需要，加载延迟加载的智能体。"""
    graph_like = agents[agent_id].graph_like
    if isinstance(graph_like, LazyLoadingAgent):
        await graph_like.load()


def get_agent(agent_id: str) -> AgentGraph:
    """获取智能体图，如果需要，加载延迟加载的智能体。"""
    agent_graph = agents[agent_id].graph_like

    # 如果是延迟加载智能体，确保它已加载并返回其图
    if isinstance(agent_graph, LazyLoadingAgent):
        if not agent_graph._loaded:
            raise RuntimeError(f"智能体 {agent_id} 未加载。请先调用 load()。")
        return agent_graph.get_graph()

    # 否则直接返回图
    return agent_graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
