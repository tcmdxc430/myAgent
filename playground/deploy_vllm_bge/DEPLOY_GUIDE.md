# 🚀 Qwen2.5-72B + BGE-Large-ZH 本地 vLLM 容器化部署指南

本指南详细介绍了如何在本地企业级 GPU 服务器上，通过 **vLLM** 推理框架与 **Docker Compose** 容器化一键部署 **Qwen2.5-72B-Instruct** (LLM) 和 **BGE-large-zh-v1.5** (Embedding) 模型的极速生产级服务。

---

## 一、 🖥️ 硬件配置与物理准备

部署 72B（720 亿参数）级别的模型，硬件显存（VRAM）是硬性门槛：

### 1. 方案 A：极致性价比方案（推荐，采用 AWQ 4-bit 量化版）
*   **模型**：`Qwen/Qwen2.5-72B-Instruct-AWQ`
*   **显存门槛**：**45 GB 以上** (加载模型需 ~38G，余下 7G 用于 KV Cache)。
*   **硬件推荐**：
    *   **2x RTX 4090 (24GB * 2 = 48GB)** (消费级神卡组合)
    *   **1x NVIDIA A100-80GB** 或 **1x H20-96GB**

### 2. 方案 B：工业级无损全精度方案（FP16 / BF16）
*   **模型**：`Qwen/Qwen2.5-72B-Instruct`
*   **显存门槛**：**160 GB 以上** (加载模型需 ~144G，高并发 KV Cache 需 20G+)。
*   **硬件推荐**：
    *   **8x RTX 4090 (24GB * 8 = 192GB)**
    *   **4x NVIDIA A100-80GB (320GB)** 
    *   **2x NVIDIA H800-80GB (160GB)**

---

## 二、 🐳 宿主机环境配置 (Linux)

在拉起 Docker 容器之前，宿主机必须安装好 NVIDIA 显卡驱动以及 **NVIDIA Container Toolkit**（使 Docker 容器能够调用宿主机显卡的关键桥梁）。

### 1. 安装 NVIDIA Container Toolkit (以 Ubuntu 为例)
在宿主机中执行以下命令：
```bash
# 1. 导入源
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. 安装 toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. 重启 Docker 引擎使之生效
sudo systemctl restart docker
```

---

## 三、 ⚙️ vLLM 核心启动参数详解

在 `docker-compose.yml` 中，我们为 vLLM 容器配置了以下关键启动参数：

*   `--model`：指定 HuggingFace/ModelScope 上的模型 ID。
*   `--quantization awq`：指定使用 AWQ 量化推理。如果是全精度 BF16/FP16，请删去此参数。
*   `--tensor-parallel-size 2`：**张量并行大小（TP）**。代表模型会被水平切分在 2 张 GPU 上协同计算。如果是单卡 A100，请改为 `1`；如果是 4 张卡，请改为 `4`。
*   `--max-model-len 16384`：限制最大上下文窗口为 16k。如果不限制，Qwen 默认会拉满到 128k，导致显存直接爆掉（OOM）。
*   `--gpu-memory-utilization 0.90`：分配 90% 的显卡空闲显存用于加载模型和缓存。
*   `--enable-prefix-caching`：**前缀缓存（极重要）**。开启后，RAG 中重复的系统提示词和上下文会被缓存在 GPU 中，使二次查询首字输出（TTFT）时间降低到毫秒级。
*   `--trust-remote-code`：信任远程代码。

---

## 四、 🚀 一键部署与启动

在当前包含 `docker-compose.yml` 的目录下，执行以下命令一键后台启动服务：

```bash
# 后台启动服务
docker compose up -d
```

### 观察容器启动日志（监控模型下载与加载进度）：
```bash
# 查看大模型 Qwen-72B 的实时加载日志
docker logs -f vllm-qwen-72b

# 查看 BGE 向量模型的实时加载日志
docker logs -f vllm-bge-embeddings
```

---

## 五、 🧪 接口调用与测试 (OpenAI 兼容协议)

当日志中出现 `Uvicorn running on http://0.0.0.0:8000` 时，代表服务已就绪！

### 1. 测试 Qwen-2.5-72B 生成大模型接口 (Port: 8000)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "messages": [
      {"role": "system", "content": "你是一个严谨的医疗器材合规专家。"},
      {"role": "user", "content": "请简述医疗器材在中国 NMPA 申报中，一类、二类、三类器械的核心区别。"}
    ],
    "temperature": 0.2
  }'
```

### 2. 测试 BGE-Large-ZH 向量提取接口 (Port: 8001)
```bash
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-large-zh-v1.5",
    "input": "智能体安全防线 Safeguard 拦截机制"
  }'
```

---

## 六、 🔗 无缝接入 myAgent 项目配置

部署完成后，你只需要打开 `myAgent` 根目录下的 `.env` 文件，进行如下修改，就能将项目大脑无缝替换为你本地部署的 72B 顶级国产大模型：

```env
# 1. 替换大模型大脑 API 地址和模型名字
OPENAI_API_KEY="vllm" # 随便填，vLLM 无需验证
OPENAI_API_BASE="http://你的服务器IP:8000/v1"
DEFAULT_MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"

# 2. 替换向量模型 API 地址
# (由于 langchain-chroma 可以直接接收兼容 OpenAI 格式的向量 API，这省去了本地运行 Embedding 包的开销)
# 你可以在 myAgent 的工具链中将 HuggingFaceEmbeddings 替换为：
# ChatOpenAI(openai_api_base="http://你的服务器IP:8001/v1", model="BAAI/bge-large-zh-v1.5")
```
