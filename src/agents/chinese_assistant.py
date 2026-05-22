from typing import Literal
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode
from core.llm import get_model
from core.settings import settings
from langchain_core.runnables import RunnableConfig
from agents.tools import calculator

# 1. 定义工具列表
tools = [calculator]
tool_node = ToolNode(tools)

# 你的 Agent 的系统提示词 ← 改这里定义个性
SYSTEM_PROMPT = """你是一个简洁、务实的中文助手。
如果你需要进行数学计算，请务必使用计算器工具。
规则：
1. 只用中文回答
2. 回答要简短，不废话
3. 如果不知道就直说，不要编造
"""

async def call_model(state: MessagesState, config: RunnableConfig):
    # 从 config 中获取模型，如果没有则使用默认模型
    model_name = config.get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    # 2. 关键：将工具绑定到模型上
    model = get_model(model_name).bind_tools(tools)
    
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}

# 3. 定义路由逻辑：判断 AI 是否想调用工具
def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

# 定义图
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node) # 添加工具执行节点
builder.set_entry_point("call_model")

# 4. 设置条件边
builder.add_conditional_edges(
    "call_model",
    should_continue,
)

# 5. 工具执行完后，必须回到 call_model 让 AI 总结结果
builder.add_edge("tools", "call_model")

# 编译成 chinese_assistant
chinese_assistant = builder.compile()
