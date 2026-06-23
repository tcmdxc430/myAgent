# 小红书图文拉取 Tool 技术方案

## 1. 功能定位

小红书图文拉取功能是一个可被前端、后端服务或外部系统调用的内容导入 tool。

它负责将小红书笔记链接中的页面文本、主图/轮播图文字、标签和来源信息提取出来，写入 PostgreSQL，并生成 Chroma 向量索引，供 `rag-assistant` 后续检索和问答使用。

当前该能力也被包含在通用图文导入接口中：

```http
POST /ingest/article
```

如果输入 URL 被识别为小红书链接，系统会自动走小红书专项抓取逻辑。

## 2. 输入

### 通用图文导入接口

```http
POST /ingest/article
Content-Type: application/json
```

请求体：

```json
{
  "url": "https://www.xiaohongshu.com/explore/...",
  "force_refresh": true
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `url` | string | 小红书分享链接、笔记链接，或其他图文网页链接 |
| `force_refresh` | boolean | 是否强制刷新已导入内容，重新抓取、OCR、切 chunk、重建向量 |

### 小红书专项接口

```http
POST /ingest/xhs
Content-Type: application/json
```

请求体同上：

```json
{
  "url": "https://www.xiaohongshu.com/explore/...",
  "force_refresh": true
}
```

### 小红书登录接口

```http
POST /ingest/xhs/login
```

用途：打开本机 Chrome 小红书登录窗口，让用户手动完成登录。登录态保存在本机独立 profile 中。

## 3. 处理流程

1. 接收 URL。
2. 规范化 URL，并判断是否为小红书链接。
3. 使用 Playwright 启动本机 Chrome。
4. 打开小红书页面。
5. 如果出现登录弹窗，优先尝试点击右上角 `X` 关闭。
6. 如果关闭后仍无法访问内容，返回 `login_required`。
7. 抽取页面标题、正文、标签、作者、发布时间和原始链接。
8. 定位左侧主图/轮播图区域。
9. 对主图区域截图，最多处理多张轮播图。
10. 将截图保存到本地缓存目录。
11. 调用百度 OCR 识别图片文字。
12. 合并页面文本和 OCR 文本，生成 `combined_text`。
13. 写入 PostgreSQL：
    - 文章主记录
    - 图片资产记录
    - chunk 明细记录
14. 将 `combined_text` 切分成 RAG chunks。
15. 删除同一文章旧 Chroma chunks。
16. 写入新的 Chroma 向量索引。
17. 返回导入状态、文章 ID、chunk 数、图片数和 OCR 失败数。

## 4. 输出

### 成功

```json
{
  "status": "success",
  "message": "Imported Xiaohongshu note.",
  "article_id": "f666f63f-c074-47a0-8a8d-fe86bac80859",
  "title": "文件系统撑不住多agent并发，但数据库又太 - 小红书",
  "source_url": "https://www.xiaohongshu.com/explore/...",
  "chunk_count": 2,
  "asset_count": 7,
  "ocr_failed_count": 0,
  "needs_login": false
}
```

### 部分成功

当正文或部分图片 OCR 成功，但部分图片识别失败时：

```json
{
  "status": "partial_success",
  "message": "Imported note, but some image OCR tasks failed.",
  "article_id": "...",
  "title": "...",
  "source_url": "https://www.xiaohongshu.com/explore/...",
  "chunk_count": 2,
  "asset_count": 7,
  "ocr_failed_count": 3,
  "needs_login": false
}
```

### 需要登录

```json
{
  "status": "login_required",
  "message": "Xiaohongshu login is required. Open the login window and retry.",
  "source_url": "https://www.xiaohongshu.com/explore/...",
  "chunk_count": 0,
  "asset_count": 0,
  "ocr_failed_count": 0,
  "needs_login": true
}
```

### 失败

```json
{
  "status": "failed",
  "message": "No readable note content was found.",
  "source_url": "https://www.xiaohongshu.com/explore/...",
  "chunk_count": 0,
  "asset_count": 0,
  "ocr_failed_count": 0,
  "needs_login": false
}
```

## 5. 存储结构

### PostgreSQL

#### `rag_articles`

保存文章主记录。

| 字段 | 说明 |
| --- | --- |
| `article_id` | 文章唯一 ID |
| `source_platform` | 来源平台，例如 `xiaohongshu` |
| `note_key` | 小红书 note id 或 URL hash |
| `source_url` | 原始输入 URL |
| `canonical_url` | 规范化后的来源 URL |
| `title` | 文章标题 |
| `author` | 作者 |
| `published_at` | 发布时间 |
| `body_text` | 页面文本内容 |
| `ocr_text` | 图片 OCR 合并文本 |
| `combined_text` | 最终用于切 chunk 的文本 |
| `tags` | 标签 |
| `status` | 导入状态 |
| `fetched_at` | 抓取时间 |

#### `rag_article_assets`

保存图片、截图和 OCR 结果。

| 字段 | 说明 |
| --- | --- |
| `asset_id` | 图片资产 ID |
| `article_id` | 所属文章 ID |
| `image_url` | 原图 URL 或截图标识 |
| `local_path` | 本地缓存路径 |
| `ocr_text` | 该图片识别出的文字 |
| `ocr_status` | `success` 或 `failed` |
| `ocr_error` | OCR 失败原因 |

#### `rag_chunks`

保存 RAG 文本切片。

| 字段 | 说明 |
| --- | --- |
| `chunk_id` | chunk ID |
| `article_id` | 所属文章 ID |
| `chunk_index` | chunk 顺序 |
| `chroma_document_id` | Chroma 向量文档 ID |
| `chunk_text` | 实际用于检索的文本 |

### Chroma

Chroma 保存每个 chunk 的向量索引。

包含：

```text
article_id
chunk_id
note_key
title
source_platform
source_url
chunk_index
```


## 6. OCR 方案

OCR 使用百度 OCR。

当前支持两种百度鉴权方式：

1. `Authorization: Bearer <API Key>`
2. `access_token`

处理策略：

- 优先识别主图截图。
- OCR 失败不阻断整篇文章入库。
- 图片失败会记录到 `rag_article_assets.ocr_status` 和 `ocr_error`。
- 成功识别的 OCR 文本会合并进 `rag_articles.ocr_text` 和 `combined_text`。

## 7. RAG 检索方案

导入完成后，RAG 检索使用两层召回：

1. PostgreSQL 关键词检索兜底：
   - 标题
   - `combined_text`
   - `rag_chunks.chunk_text`
2. Chroma 向量检索：
   - 基于 chunk embedding 做语义召回

最终会合并去重，返回带来源标题和 URL 的上下文给 `rag-assistant`。

## 8. 外部系统调用方式

如果服务部署在 `http://127.0.0.1:8080`：

```bash
curl -X POST http://127.0.0.1:8080/ingest/article \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"https://www.xiaohongshu.com/explore/...\",\"force_refresh\":true}"
```

如果配置了 `AUTH_SECRET`，需要加：

```bash
-H "Authorization: Bearer <AUTH_SECRET>"
```

## 9. 失败处理

| 场景 | 处理方式 |
| --- | --- |
| 登录弹窗可关闭 | 点击 `X` 后继续抓取 |
| 仍需登录 | 返回 `login_required` |
| 部分图片 OCR 失败 | 返回 `partial_success`，正文和成功 OCR 仍入库 |
| 链接无效 | 返回 `failed` |
| 重复导入 | 更新 PostgreSQL，删除旧 Chroma chunks 后重建 |
| 百度 OCR 限流 | 标记对应图片 OCR 失败，不阻断文章入库 |


