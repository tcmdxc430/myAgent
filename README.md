# 🧰 我的多智能体 (My Multi-Agent)

这是一个基于 LangGraph、FastAPI 和 Streamlit 构建的全栈 AI 智能体服务平台，支持多智能体协同 work。

## 🚀 功能介绍

本平台集成了一系列先进的 AI 智能体功能，旨在提供高效、智能的交互与数据分析服务：

1.  **多智能体协同 (Multi-Agent Collaboration)**：采用 `langgraph-supervisor` 模式，实现主智能体与多个专业子智能体（如研究助手、SQL 助手、RAG 助手）之间的无缝切换与任务分发。
2.  **专业中文助理**：针对中文语境优化的智能助手，能够理解复杂的指令并提供精准回复。
3.  **RAG 知识库检索**：集成 ChromaDB 向量数据库，支持对本地文档进行深度检索与问答。
4.  **SQL 业务数据分析**：内置专业 SQL 助手，能够安全地连接业务数据库，执行查询并生成分析结果。
5.  **先进的流式交互**：支持基于令牌 (Token) 和消息 (Message) 的混合流式输出，提供极致的响应速度与交互体验。
6.  **人机协作 (Human-in-the-loop)**：利用 LangGraph 的 `interrupt` 机制，在执行敏感操作时支持人工审批。
7.  **长期记忆与持久化**：支持 PostgreSQL 持久化存储，能够记住用户的偏好与历史对话上下文。
8.  **语音交互支持**：集成 STT (语音转文本) 和 TTS (文本转语音) 功能，支持更自然的交互方式。

## 🛠️ 启动方式

您可以选择使用 Docker 容器化部署，或者直接在本地 Python 环境中运行。

### 1. 准备工作

首先，克隆仓库并配置环境变量：

```sh
git clone <your-repo-url>
cd myAgent
cp .env.example .env
# 编辑 .env 文件，填入您的 LLM API Key (如 DEEPSEEK_API_KEY 或 OPENAI_API_KEY)
```

### 2. 使用 Docker 启动 (推荐)

这是最简单的启动方式，会自动配置数据库和所有依赖：

```sh
docker compose watch
```

启动后：
- **前端界面**: 访问 `http://localhost:8501`
- **后端 API**: 访问 `http://localhost:8080/docs` 查看 Swagger 文档

### 3. 本地 Python 启动

如果您希望在本地开发环境中运行：

**第一步：安装依赖并启动后端服务**

```sh
# 建议使用 uv 进行包管理
uv sync --frozen
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
python src/run_service.py
```

**第二步：启动 Streamlit 前端**

在另一个终端窗口中：

```sh
source .venv/bin/activate
streamlit run src/streamlit_app.py
```
