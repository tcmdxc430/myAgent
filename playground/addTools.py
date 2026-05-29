from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END


# ── 1. 定义一个工具 ──────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气"""
    data = {"北京": "晴天 22°C", "上海": "多云 18°C", "深圳": "小雨 25°C"}
    return data.get(city, f"{city} 暂无数据")


tools = [get_weather]
tools_by_name = {t.name: t for t in tools}


# ── 2. LLM 绑定工具 ──────────────────────────────────────────
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="your_key_here",
).bind_tools(tools)


# ── 3. 节点函数 ───────────────────────────────────────────────
def call_model(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


def call_tools(state: MessagesState):
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "call_tools"
    return END


# ── 4. 构建图 ─────────────────────────────────────────────────
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("call_tools", call_tools)
builder.set_entry_point("call_model")
builder.add_conditional_edges("call_model", should_continue)
builder.add_edge("call_tools", "call_model")
graph = builder.compile()


# ── 5. 运行 ───────────────────────────────────────────────────
if __name__ == "__main__":
    result = graph.invoke({"messages": [HumanMessage(content="北京天气怎么样？")]})
    print(result["messages"][-1].content)