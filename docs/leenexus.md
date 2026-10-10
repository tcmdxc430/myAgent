# 本机 LeeNexus 接入

myAgent 提供 `POST /a2a/myagent`，LeeNexus 中注册名为 `myagent-git-status`。该入口运行 `leenexus-git-agent`，Agent 通过 LeeNexus 的 `/rpc` 调用 `git-mcp-local-git-status`，返回当前 myAgent 仓库状态。此 Agent 不需要模型 API Key。

## 配置

运行 myAgent 的环境需要：

```text
LEENEXUS_URL=http://127.0.0.1:8000
LEENEXUS_TOKEN=<LeeNexus Bearer token>
AUTH_SECRET=<myAgent A2A 入口的 Bearer token>
```

注册脚本还读取 `MYAGENT_AUTH_SECRET`（与 `AUTH_SECRET` 相同）和可选的 `MYAGENT_A2A_URL`（默认 `http://127.0.0.1:8080/a2a/myagent`）。它会在 LeeNexus 创建或更新注册项，并配置 LeeNexus 调用 myAgent 时使用的认证头。

```powershell
python scripts/register_leenexus_a2a.py
```

## 本机整链验收

LeeNexus 本机仓库及其服务应已启动，且已注册 `git-mcp-local-git-status`。在项目根目录运行：

```powershell
$env:LEENEXUS_REPO='C:\Users\39703\Documents\leenexus-platform'
.\.venv\Scripts\python.exe scripts\verify_leenexus_integration.py
```

脚本优先读取 myAgent `.env` 中的 `LEENEXUS_TOKEN` 作为运行令牌；未配置时生成一天有效的测试令牌。注册 A2A 项使用单独的临时管理令牌。脚本从本机 LeeNexus 的 JWT 密钥派生固定的 myAgent 鉴权密钥，重复运行时前后端可以使用同一密钥。脚本用临时 SQLite 数据库启动 myAgent，经 LeeNexus 调用并检查返回的 Git 状态。成功后 myAgent 服务继续运行。重新验收前先停止当前占用 8080 端口的 myAgent 进程。日志位于系统临时目录的 `myagent-leenexus-smoke.log`。

外部调用格式：

```json
{"parameters":{"query":"查询 myAgent 仓库 Git 状态"}}
```

调用地址：`POST http://127.0.0.1:8000/a2a/myagent-git-status/invoke`，使用 LeeNexus Bearer token。
