# 小红书链接导入 RAG 知识库功能说明

项目：myAgent  
日期：2026-06-15  
版本：XHS Ingestion v1

## 1. 功能概览

当前版本已经完成从 Streamlit 左侧栏输入小红书分享链接，到后端抓取正文、图片 OCR、PostgreSQL 原文入库、Chroma 向量索引，以及 `rag-assistant` 检索溯源的基础闭环。

登录采用用户本地授权会话：用户首次点击“登录小红书”后，在本机 Chrome 中手动完成登录，后续导入复用本地 profile。系统不绕过验证码、不抓取用户无权访问的内容。

## 2. 已实现功能

- Streamlit 左侧栏新增“小红书导入”入口。
- 支持填写小红书分享 URL、强制刷新、导入、登录小红书和导入状态提示。
- 新增后端接口：
  - `POST /ingest/xhs`：导入小红书笔记。
  - `POST /ingest/xhs/login`：打开本地小红书登录窗口。
- 支持小红书 URL 规范化与 `note_id` 提取。
- 使用 Playwright 驱动本机 Chrome 访问小红书页面。
- 使用独立 Chrome profile 保存小红书登录态。
- 抽取标题、正文、作者、发布时间、标签、图片 URL、原链接和抓取时间。
- 下载图片并调用 Gemini 2.5 Flash 进行 OCR。
- PostgreSQL 保存文章原文、图片资产、OCR 文本和 chunk 明细。
- Chroma 保存 chunk 向量索引。
- `rag-assistant` 检索时可回查文章标题和来源链接。
- 重复导入同一篇笔记时，更新 PostgreSQL 并删除旧 Chroma chunks 后重建索引。
- 图片 OCR 部分失败不会阻断正文入库。

## 3. 技术架构

整体采用混合 RAG 架构：

- PostgreSQL：保存原文、结构化元数据、图片 OCR、chunk 明细。
- Chroma：保存 chunk 向量索引，用于语义检索。

核心链路：

```text
Streamlit 侧栏
  -> AgentClient
  -> FastAPI /ingest/xhs
  -> XhsImporter
  -> Playwright + Chrome 抓取页面
  -> Gemini 2.5 Flash 图片 OCR
  -> PostgreSQL 写入原文和 chunk
  -> Chroma 写入向量索引
  -> rag-assistant 检索与回答
```

## 4. 核心组件

| 层级 | 组件 | 职责 |
| --- | --- | --- |
| 前端 | Streamlit sidebar | 输入 URL、触发导入/登录、展示状态 |
| 客户端 | AgentClient | 封装导入 API 调用 |
| API | FastAPI service | 暴露 `/ingest/xhs` 和 `/ingest/xhs/login` |
| 抓取 | Playwright + system Chrome | 复用本地登录态，打开并解析小红书页面 |
| OCR | Gemini 2.5 Flash | 识别图片中的文字 |
| 原文存储 | PostgreSQL | 保存文章、图片资产、chunk 记录 |
| 向量库 | Chroma | 保存 chunk embedding 和检索 metadata |
| 问答 | rag-assistant | 基于 Chroma 检索，并回查 PostgreSQL 来源信息 |

## 5. 数据模型

### `rag_articles`

保存文章主记录：

- 平台
- `note_id`
- canonical URL
- 原始 URL
- 标题
- 作者
- 正文
- OCR 合并文本
- 状态
- 错误原因
- 创建/更新时间

### `rag_article_assets`

保存图片资产：

- 图片 URL
- 本地缓存路径
- OCR 文本
- OCR 状态
- 所属文章 ID

### `rag_chunks`

保存 chunk 明细：

- chunk 文本
- chunk 顺序
- Chroma document ID
- 所属文章 ID

Chroma metadata 包含：

- `article_id`
- `chunk_id`
- `source_platform=xiaohongshu`
- `source_url`

## 6. 当前本地部署约定

| 项目 | 当前值 |
| --- | --- |
| PostgreSQL | Docker 容器端口映射为 `5433 -> 5432` |
| Chroma | `D:\仓库\myAgent\chroma_db\chroma.sqlite3` |
| 小红书登录 profile | `C:\Users\39703\AppData\Local\myAgent\xhs-playwright-profile` |
| 小红书图片缓存 | `C:\Users\39703\AppData\Local\myAgent\xhs-assets` |
| 已验证样例 | 真实小红书链接首次导入成功，生成 7 个 chunks、7 个 assets |

## 7. 导入流程

1. 用户在 Streamlit 左侧栏输入小红书分享链接。
2. 用户点击“导入”。
3. 后端规范化 URL，并提取 `note_id`。
4. Playwright 使用本地 Chrome profile 打开页面。
5. 如果需要登录，接口返回 `login_required`，前端提示用户点击“登录小红书”。
6. 如果页面可访问，系统抽取正文、标题、作者、标签、发布时间和图片 URL。
7. 图片被下载到本地缓存目录。
8. Gemini 2.5 Flash 对图片执行 OCR。
9. PostgreSQL 写入文章、图片资产和 chunk 记录。
10. Chroma 删除旧 chunk 向量并写入新的向量索引。
11. 用户切换到 `rag-assistant` 后，可以基于导入内容提问。

## 8. 重点难点

### 登录与风控

小红书内容经常需要登录态或触发风控。当前方案只复用用户本地手动登录后的授权 session，不做验证码绕过，也不访问用户无权限内容。

### Windows 路径编码

项目路径中包含中文目录，Chrome/Playwright 在某些启动场景下会出现路径编码问题。因此登录 profile 和图片缓存迁移到了 ASCII 友好的 AppData 路径。

### Playwright 浏览器安装

Playwright Python 包可以安装，但浏览器二进制下载不稳定。当前实现改用系统已安装的 Chrome，降低本地安装阻力。

### 事件循环兼容

服务入口使用的 Windows event loop 策略会影响 Playwright 子进程启动。当前在抓取线程内临时使用 Proactor event loop 策略，规避子进程启动失败。

### 内容抽取稳定性

小红书页面结构可能变化。当前版本先基于页面 DOM 和可见文本抽取，再用图片 OCR 补充内容。后续可以增强结构化 JSON 提取能力。

### OCR 部分失败

图片可能下载失败、被防盗链拦截或模型识别失败。当前策略是不阻断正文入库，只记录部分图片 OCR 失败状态。

### 重复导入一致性

同一篇笔记重复导入时，需要避免 PostgreSQL 和 Chroma 出现旧数据残留。当前流程会更新文章记录，并删除旧 Chroma chunks 后重建。

## 9. 测试与验证

已覆盖的测试方向：

- URL 规范化。
- `note_id` 提取。
- 重复导入更新。
- chunk metadata 生成。
- 导入成功。
- 需要登录。
- 无效链接。
- OCR 部分失败。

已执行测试命令：

```powershell
uv run pytest tests\ingestion\test_xhs.py tests\service\test_xhs_ingest.py
```

结果：

```text
5 passed
```

真实链接验证结果：

- 小红书真实链接首次导入成功。
- PostgreSQL 中可查到文章、图片资产和 chunk 记录。
- Chroma 中生成对应向量索引。
- `rag-assistant` 可通过检索链路使用导入内容。

## 10. 未来可扩展功能

### 导入成果查看页

在前端新增“知识库导入记录”页面，展示：

- 已导入文章列表
- 文章标题
- 来源链接
- 导入状态
- chunk 数量
- 图片数量
- OCR 状态
- 失败原因
- 删除/重新导入按钮

### 后台任务队列

将导入流程改造成异步 job，支持：

- 进度展示
- 日志查看
- 失败重试
- 长任务不阻塞前端
- 并发导入控制

### 更稳定的内容抽取

增强小红书页面解析能力：

- 优先提取页面中的结构化 JSON。
- 保留 DOM fallback。
- 建立真实样例回归测试集。
- 针对页面结构变化做自动告警。

### 向量存储演进

可以考虑引入 `pgvector`，将结构化数据和向量数据统一放入 PostgreSQL，减少 PostgreSQL + Chroma 双存储的一致性维护成本。

### OCR 抽象层

为 OCR 增加 provider 抽象：

- Gemini 2.5 Flash
- 本地 OCR
- 云 OCR
- 多 provider fallback

### 多平台导入

将当前 ingestion 能力扩展到更多来源：

- 微信公众号文章
- 知乎
- 普通网页
- PDF
- 视频字幕
- 本地文档

### 权限与审计

如果未来支持多人使用，需要增加：

- 用户级数据隔离
- 来源授权记录
- 删除追踪
- 敏感内容策略
- 数据导出与清理能力

### RAG 引用体验增强

在回答中展示更清晰的引用信息：

- 来源标题
- 原文链接
- 命中的 chunk
- 正文命中还是图片 OCR 命中
- 图片预览或资产引用

## 11. 建议下一步

1. 优先补一个“导入成果查看”页面，让用户不用查数据库也能看到文章、图片和 chunks。
2. 将导入流程改造成后台任务，提升真实网页抓取和 OCR 时的交互体验。
3. 为小红书抽取逻辑增加真实样例回归测试，降低页面结构变化带来的维护成本。
4. 在 RAG 回答中突出来源引用，区分正文命中和图片 OCR 命中。
