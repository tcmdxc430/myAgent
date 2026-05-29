from typing import Literal, Annotated
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from core.llm import get_model
from core.settings import settings
from langchain_core.runnables import RunnableConfig

# --- 1. 定义状态 ---
class ReflectiveState(MessagesState):
    # 记录反思次数，防止无限循环
    reflection_count: int

# --- 2. 定义节点 ---

async def generation_node(state: ReflectiveState, config: RunnableConfig):
    """生成初始回答或根据反馈进行修改"""
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    
    system_prompt = SystemMessage(content="""你是一个专业的文案写作专家。
    请根据用户的要求撰写内容。如果对话历史中有‘反思意见’，请务必参考该意见进行修改。""")
    
    messages = [system_prompt] + state["messages"]
    response = await model.ainvoke(messages)
    
    # 获取当前反思计数
    count = state.get("reflection_count", 0)
    return {"messages": [response], "reflection_count": count}

async def reflection_node(state: ReflectiveState, config: RunnableConfig):
    """作为批评者，对生成的内容进行严格审查"""
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    
    # 获取 AI 刚才生成的最后一条消息
    last_ai_message = state["messages"][-1].content
    
    reflection_prompt = [
        SystemMessage(content="""你是一名严苛的内容审查员。
        你的任务是检查 AI 生成的内容是否存在以下问题：
        1. 逻辑是否自洽？
        2. 是否包含事实错误？
        3. 语气是否合适？
        
        如果内容很完美，请只回复‘合格’。
        如果内容有待改进，请详细列出修改建议，并以‘建议：’开头。"""),
        HumanMessage(content=f"请审查以下内容：\n\n{last_ai_message}")
    ]
    
    reflection_res = await model.ainvoke(reflection_prompt)
    
    # 增加反思计数并返回反馈
    count = state.get("reflection_count", 0) + 1
    feedback = f"【自我反思 第 {count} 次】\n{reflection_res.content}"
    return {"messages": [AIMessage(content=feedback)], "reflection_count": count}

# --- 3. 定义路由逻辑 ---

def should_continue(state: ReflectiveState) -> Literal["reflect", END]:
    """决定是继续反思还是结束"""
    last_message = state["messages"][-1].content
    count = state.get("reflection_count", 0)
    
    # 如果 AI 说合格，或者反思次数超过 2 次（防止死循环），则结束
    if "合格" in last_message or count >= 2:
        return END
    
    return "reflect"

# --- 4. 构建图 ---

builder = StateGraph(ReflectiveState)

builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)

builder.set_entry_point("generate")

# generate -> 检查是否需要反思
builder.add_conditional_edges("generate", should_continue, {
    "reflect": "reflect",
    END: END
})

# reflect -> 必定回到 generate 进行修改
builder.add_edge("reflect", "generate")

reflective_agent = builder.compile(name="reflective_agent")
