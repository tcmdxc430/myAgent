from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from core.llm import get_model
from core.settings import settings
from langchain_core.runnables import RunnableConfig

# --- 1. 定义我们想要的结构化数据模型 ---
class UserProfile(BaseModel):
    """用户信息模型"""
    name: str = Field(description="用户的姓名")
    age: int = Field(description="用户的年龄")
    interests: list[str] = Field(description="用户的兴趣爱好列表")
    summary: str = Field(description="对用户的一句话简评")

# --- 2. 定义节点 ---

async def info_extractor(state: MessagesState, config: RunnableConfig):
    model_name = config.get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    
    # 获取基础模型
    llm = get_model(model_name)
    
    # 【兼容性修复】改用工具调用方式来强制输出结构化数据
    # 这种方式比 with_structured_output 的默认方式兼容性更好，特别是在 DeepSeek 等模型上
    structured_llm = llm.with_structured_output(UserProfile, method="function_calling")
    
    system_prompt = SystemMessage(content="你是一个信息提取专家。请从用户的对话中提取姓名、年龄和兴趣爱好。")
    messages = [system_prompt] + state["messages"]
    
    # 调用模型，返回的是 UserProfile 对象
    profile = await structured_llm.ainvoke(messages)
    
    # 为了在聊天界面显示，我们将结构化数据转为字符串存入 AIMessage
    # 在实际业务中，你可以直接把 profile 对象存入数据库或进行后续逻辑处理
    content = f"已提取结构化数据：\n\n"
    content += f"👤 姓名: {profile.name}\n"
    content += f"🎂 年龄: {profile.age}\n"
    content += f"🎨 兴趣: {', '.join(profile.interests)}\n"
    content += f"📝 简评: {profile.summary}\n\n"
    content += f"原始 JSON: {profile.model_dump_json(indent=2)}"
    
    return {"messages": [AIMessage(content=content)]}

# --- 3. 构建图 ---

builder = StateGraph(MessagesState)
builder.add_node("extractor", info_extractor)
builder.set_entry_point("extractor")
builder.add_edge("extractor", END)

structured_output_agent = builder.compile()
