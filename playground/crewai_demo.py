import sys
import os

# 将 crewAI 源码路径加入 Python 路径，以便直接使用源码
sys.path.append(r"D:\Download\crewAI-main\crewAI-main\lib\crewai\src")
sys.path.append(r"D:\Download\crewAI-main\crewAI-main\lib\crewai-core\src")

from crewai import Agent, Task, Crew, Process, LLM

# 配置 LLM (使用你已有的 DeepSeek)
# 注意：需要确保环境变量 DEEPSEEK_API_KEY 已设置
my_llm = LLM(
    model="deepseek/deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

# 1. 定义智能体
manager = Agent(
    role='项目经理',
    goal='确保关于 {topic} 的调研报告质量精良且逻辑清晰',
    backstory='你是一名经验丰富的项目经理，擅长协调团队成员并对产出进行严格把关。',
    allow_delegation=True,
    verbose=True,
    llm=my_llm
)

researcher = Agent(
    role='资深研究员',
    goal='挖掘关于 {topic} 的深度见解',
    backstory='你是一名技术嗅觉敏锐的研究员，擅长从复杂的信息中提取核心结论。',
    allow_delegation=False,
    verbose=True,
    llm=my_llm
)

# 2. 定义任务
task1 = Task(
    description='调研 {topic} 在 2026 年的最新进展，并列出 3 个核心趋势。',
    expected_output='一份包含 3 个核心趋势的简要调研大纲。',
    agent=researcher # 指定由研究员执行
)

task2 = Task(
    description='审核研究员提供的调研大纲，并将其扩展为一份正式的汇报文稿。',
    expected_output='一份逻辑严密的汇报文稿，格式为 Markdown。',
    agent=manager # 指定由经理审核并收尾
)

# 3. 组装团队 (Crew)
my_crew = Crew(
    agents=[manager, researcher],
    tasks=[task1, task2],
    process=Process.sequential, # 顺序执行：研究员先做，经理后做
    verbose=True
)

# 4. 运行
print("### 开始运行 CrewAI 演示 ###")
result = my_crew.kickoff(inputs={'topic': 'Text-to-SQL 技术'})
print("\n\n########################")
print("## 最终产出结果 ##")
print("########################\n")
print(result)
