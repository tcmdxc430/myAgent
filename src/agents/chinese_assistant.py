from typing import Literal
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode
from core.llm import get_model
from core.settings import settings
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from agents.tools import calculator, database_search

# --- 1. 定义工具 ---

@tool
def transfer_to_english_expert():
    """将任务移交给英文专家。当你发现用户需要翻译、写英文邮件或进行英语对话时调用此工具。"""
    return "正在转接英文专家..."

@tool
def transfer_to_translation_expert():
    """将任务移交给翻译专家。当你需要将英文内容翻译成地道的中文，或者需要中英双语对照时调用此工具。"""
    return "正在转接翻译专家..."

@tool
def transfer_back_to_chinese_host():
    """任务完成，返回中文接待员。当所有专家任务处理完毕或用户要求停止时调用此工具。"""
    return "任务完成，正在返回中文接待员..."

# 基础工具 + 移交工具
all_tools = [
    calculator, 
    database_search, 
    transfer_to_english_expert, 
    transfer_to_translation_expert,
    transfer_back_to_chinese_host
]
tool_node = ToolNode(all_tools)

# --- 2. 定义节点 (Agents) ---

async def chinese_host(state: MessagesState, config: RunnableConfig):
    model_name = config.get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    # 中文接待员绑定工具
    host_tools = [calculator, database_search, transfer_to_english_expert, transfer_to_translation_expert]
    model = get_model(model_name).bind_tools(host_tools)
    
    system_prompt = SystemMessage(content="""你是一个友好的中文接待员。
    1. 负责用中文回答用户。
    2. 如果需要算术或查手册，正常使用工具。
    3. 如果用户需要写英文内容，转交给 english_expert。
    4. 如果用户直接要求翻译一段文字，转交给 translation_expert。
    """)
    
    messages = [system_prompt] + state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def english_expert(state: MessagesState, config: RunnableConfig):
    model_name = config.get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    # 英文专家可以转回中文，也可以转给翻译
    expert_tools = [transfer_to_translation_expert, transfer_back_to_chinese_host]
    model = get_model(model_name).bind_tools(expert_tools)
    
    system_prompt = SystemMessage(content="""You are an English Language Expert.
    1. Respond to the user in fluent English.
    2. If you have written an English text (like an email) and think the user needs a Chinese translation for confirmation, call 'transfer_to_translation_expert'.
    3. If the user speaks Chinese, call 'transfer_back_to_chinese_host'.
    """)
    
    messages = [system_prompt] + state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def translation_expert(state: MessagesState, config: RunnableConfig):
    model_name = config.get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    # 翻译专家完成后转回中文接待员
    translation_tools = [transfer_back_to_chinese_host]
    model = get_model(model_name).bind_tools(translation_tools)
    
    system_prompt = SystemMessage(content="""你是一个精通中英双语的翻译专家。
    1. 你的任务是提供高质量、地道的翻译。
    2. 翻译完成后，务必调用 'transfer_back_to_chinese_host' 将控制权交还给接待员。
    """)
    
    messages = [system_prompt] + state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}

# --- 3. 路由逻辑 ---

def route_agent(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

def after_tools(state: MessagesState) -> Literal["chinese_host", "english_expert", "translation_expert"]:
    last_msg = state["messages"][-1]
    if "转接英文专家" in last_msg.content:
        return "english_expert"
    if "转接翻译专家" in last_msg.content:
        return "translation_expert"
    if "返回中文接待员" in last_msg.content:
        return "chinese_host"
    return "chinese_host"

# --- 4. 构建图 ---

builder = StateGraph(MessagesState)

builder.add_node("chinese_host", chinese_host)
builder.add_node("english_expert", english_expert)
builder.add_node("translation_expert", translation_expert)
builder.add_node("tools", tool_node)

builder.set_entry_point("chinese_host")

builder.add_conditional_edges("chinese_host", route_agent)
builder.add_conditional_edges("english_expert", route_agent)
builder.add_conditional_edges("translation_expert", route_agent)
builder.add_conditional_edges("tools", after_tools)

chinese_assistant = builder.compile()
