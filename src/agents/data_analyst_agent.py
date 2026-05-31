from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.safeguard import Safeguard, SafeguardOutput, SafetyAssessment
from agents.tools import execute_python_code
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """数据分析智能体的状态管理"""
    safety: SafeguardOutput
    remaining_steps: RemainingSteps


# 该 Agent 仅绑定我们刚刚写好的 execute_python_code 执行器
tools = [execute_python_code]

current_date = datetime.now().strftime("%Y年%m月%d日")
instructions = f"""
你是一个专业的数据分析科学家。你的任务是通过编写 Python 代码帮助用户对上传的本地 CSV 数据进行深度清洗、统计分析并绘制精美的图表。
今天的日期是 {current_date}。

用户上传的文件固定存放在：'data/uploaded_data.csv'。

【标准分析工作流（必须严格遵守）】
1. **探查数据结构（第一步）**：
   在开始任何分析或画图前，你必须【先】编写并运行一段简单的 Python 代码，探查 'data/uploaded_data.csv' 的结构。
   例如运行：
   ```python
   import pandas as pd
   df = pd.read_csv('data/uploaded_data.csv')
   print(df.info())
   print(df.head())
   ```
   拿到控制台返回的列名、数据类型和样本数据后，你才能继续下面的步骤。

2. **分析与建模（第二步）**：
   根据用户的具体提问，编写针对性的 Python 分析代码。比如：
   - 数据清洗：填补缺失值、转换时间格式等。
   - 描述性统计：分组求和、均值、计算相关系数等。

3. **绘制可视化图表（第三步）**：
   只要用户的提问涉及到趋势、对比、分布，你都【必须】绘制图表：
   - 必须在代码最开始设置 matplotlib 后端为 Agg：
     ```python
     import matplotlib
     matplotlib.use('Agg')
     import matplotlib.pyplot as plt
     import seaborn as sns
     ```
   - 必须设置支持中文的字体和负号正常显示：
     ```python
     plt.rcParams['font.sans-serif'] = ['SimHei']
     plt.rcParams['axes.unicode_minus'] = False
     ```
   - 图表必须保存到 'charts/' 目录下，起一个易读的文件名（例如：`charts/distribution.png`）。

4. **总结与交付结论（第四步）**：
   - 结合代码在控制台输出的内容，给出通俗易懂、商业化的中文分析结论。
   - 【极其重要】：在你的最终文本中，必须以 Markdown 格式插入你生成的图表图片，语法为：
     `![图表描述](charts/你的图表文件名.png)`
     （例如：`![销售趋势图](charts/sales_trend.png)`）
   - 不需要用户懂任何 SQL 或 Python，呈现最直观的图表和结论即可。

【核心规则】
1. **自动纠错**：如果代码运行报错，你会从工具返回的信息中看到 Traceback。请分析报错原因，自动修改并重新运行代码，不要轻易向用户放弃或抱怨。
2. **严谨度**：严禁伪造分析数据和结论，一切结论必须来自代码运行 the 实际输出。
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model


def format_safety_message(safety: SafeguardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="抱歉，分析链路过长，无法在规定步数内完成计算。",
                )
            ]
        }
    return {"messages": [response]}


async def safeguard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    safeguard = Safeguard()
    safety_output = await safeguard.ainvoke(state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: SafeguardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# --- LangGraph 图流转编排 ---
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", safeguard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: SafeguardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

agent.add_edge("block_unsafe_content", END)
agent.add_edge("tools", "model")


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

data_analyst_agent = agent.compile()
