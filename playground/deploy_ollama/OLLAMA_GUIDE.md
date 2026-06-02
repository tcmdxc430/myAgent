# 🦙 Windows 10 本地极速部署 Ollama + Qwen2.5-1.5B 指南

在本地拥有 **GTX 1650 (4GB 显存)** 的 Windows 10 电脑上，使用 **Ollama** 部署 **Qwen2.5-1.5B-Instruct** 是最完美、最流畅的选择。

*   **极速响应**：1.5B 模型在 GTX 1650 上运行速度可达 **50 ~ 80 tokens/秒**，体验极其丝滑。
*   **零配置 GPU 加速**：Ollama Windows 客户端会自动检测并启用你的 NVIDIA 显卡进行加速，无需配置 WSL2 或复杂的 Docker 驱动。
*   **安全离线**：所有计算完全在本地进行，100% 物理隔离。

---

## 一、 📥 第一步：安装 Ollama 客户端

1.  **下载安装包**：
    *   点击官方链接直接下载 Windows 安装包：[OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe)
2.  **安装**：
    *   双击运行 `OllamaSetup.exe`，点击 **Install**。
    *   安装完成后，Windows 右下角系统托盘会出现一只可爱的“羊驼”图标，代表 Ollama 已在后台运行。
3.  **验证安装**：
    *   打开一个新的 PowerShell 或 CMD 窗口，运行：
        ```powershell
        ollama --version
        ```
        如果输出了版本号（如 `ollama version is 0.x.x`），说明安装成功！

---

## 二、 🚀 第二步：一键拉取并运行 Qwen2.5-1.5B

在命令行中直接运行以下命令：

```powershell
ollama run qwen2.5:1.5b
```

*   **自动下载**：Ollama 会自动从官方仓库下载 Qwen2.5-1.5B-Instruct 模型（约 980MB，下载极快）。
*   **直接对话**：下载完成后，你会直接进入命令行交互界面。你可以输入任何问题测试它，例如：
    ```text
    >>> 你好，请介绍一下你自己。
    ```
*   **退出交互**：输入 `/exit` 或按 `Ctrl + D` 即可退出命令行对话（Ollama 服务依然会在后台运行，提供 API 接口）。

---

## 三、 🔗 第三步：无缝接入 myAgent 项目

Ollama 启动后，默认会在本地 `http://localhost:11434` 暴露标准的 API 接口，并且**原生兼容 OpenAI 协议**。

我们只需要修改 `myAgent` 根目录下的 `.env` 文件，即可将项目大脑切换为本地的 Qwen2.5-1.5B。

### 1. 修改 `.env` 配置文件

打开 `D:\Download\myAgent\.env`，找到 LLM 配置部分，修改为以下内容：

```env
# 1. 启用 Ollama 供应商
DEFAULT_MODEL="ollama"

# 2. 配置 Ollama 模型名字和本地地址
OLLAMA_MODEL="qwen2.5:1.5b"
OLLAMA_BASE_URL="http://localhost:11434"
```

> 💡 **提示**：此时你不需要配置 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`，项目在启动时检测到 `DEFAULT_MODEL="ollama"`，会自动通过本地的 Ollama 引擎进行推理。

---

## 四、 🧪 第四步：本地连接测试

为了确保 `myAgent` 能够成功调用本地的 Ollama，我们在当前目录下提供了一个极简的测试脚本 `test_ollama.py`。

### 1. 运行测试脚本

在 `myAgent` 的虚拟环境下运行：

```powershell
python playground/deploy_ollama/test_ollama.py
```

如果看到控制台流畅地**流式输出** Qwen2.5 的回答，说明本地大模型大脑已经完美打通！

---

## 五、 🧠 关于向量模型 (Embeddings)

在 RAG（知识库检索）中，我们需要将文档切片并转化为向量。

*   **无需额外配置**：`myAgent` 项目在 `scripts/create_chroma_db.py` 和 `src/agents/tools.py` 中，默认已经配置了本地离线运行的 HuggingFace 向量模型 **`sentence-transformers/all-MiniLM-L6-v2`**。
*   它会在你第一次运行知识库切片或检索时，**自动下载到本地并在本地 CPU/GPU 上运行**，完全不需要依赖任何云端 API，也不需要占用 Ollama 的额外显存。
*   这套“本地 Ollama (LLM) + 本地 Sentence-Transformers (Embeddings)”的组合，是目前最经典、最轻量且完全免费的**纯本地离线 RAG 闭环方案**。
