import io
import math
import os
import re
import sqlite3
import sys
import traceback

import numexpr
import psycopg
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import HuggingFaceEmbeddings

from core import settings
from memory.postgres import get_postgres_connection_string


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents with source metadata
def format_contexts(docs):
    context_items = []
    article_ids = [doc.metadata.get("article_id") for doc in docs if doc.metadata.get("article_id")]
    articles = _load_rag_articles(article_ids)
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Source")
        source_url = doc.metadata.get("source_url")
        title = doc.metadata.get("title")
        article_id = doc.metadata.get("article_id")
        page_number = doc.metadata.get("page_number")
        section_title = doc.metadata.get("section_title")
        if not page_number:
            page_match = re.search(r"【PDF 第 (\d+) 页】", doc.page_content)
            if page_match:
                page_number = int(page_match.group(1))
        if not section_title:
            section_match = re.search(r"【章节：([^】]+)】", doc.page_content)
            if section_match:
                section_title = section_match.group(1)
        if article_id in articles:
            article = articles[article_id]
            title = article.get("title") or title
            source_url = article.get("canonical_url") or source_url
        # Extract just the filename if it's a full path
        source_name = source.split("\\")[-1].split("/")[-1]
        header = f"--- Source {i+1}: {title or source_name} ---"
        if page_number:
            header += f"\nPage: {page_number}"
        if section_title:
            header += f"\nSection: {section_title}"
        if source_url:
            header += f"\nURL: {source_url}"
        item = f"{header}\n{doc.page_content}"
        context_items.append(item)
    return "\n\n".join(context_items)


def _load_rag_articles(article_ids: list[str]) -> dict[str, dict[str, str]]:
    if not article_ids or settings.DATABASE_TYPE != "postgres":
        return {}
    try:
        with psycopg.connect(get_postgres_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT article_id, title, canonical_url
                    FROM rag_articles
                    WHERE article_id = ANY(%s)
                    """,
                    (list(set(article_ids)),),
                )
                return {
                    row[0]: {"title": row[1], "canonical_url": row[2]}
                    for row in cur.fetchall()
                }
    except Exception:
        return {}


def load_chroma_db(k: int = 12):
    # 使用 HuggingFace 的本地向量模型
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize HuggingFaceEmbeddings: {e}"
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    return retriever


def database_search_func(query: str) -> str:
    """Searches the imported article knowledge base for relevant information."""
    # Get the chroma retriever
    retriever = load_chroma_db()

    # Search the database for relevant documents
    keyword_documents = _load_keyword_rag_documents(query)
    vector_documents = retriever.invoke(query)
    documents = _merge_retrieved_documents(
        keyword_documents + vector_documents,
        limit=12,
        diversify_articles=_is_catalog_query(query),
    )

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


def _load_keyword_rag_documents(query: str, limit: int | None = None) -> list[Document]:
    if settings.DATABASE_TYPE != "postgres":
        return []
    terms = _query_terms(query)
    if not terms:
        return []
    catalog_query = _is_catalog_query(query)
    if limit is None:
        limit = 12 if catalog_query else 8

    conditions = []
    score_expressions = []
    score_params: list[str] = []
    where_params: list[str] = []
    for term in terms:
        like = f"%{term}%"
        score_expressions.append(
            "(CASE WHEN a.title ILIKE %s THEN 12 ELSE 0 END + "
            "CASE WHEN c.chunk_text ILIKE %s THEN 4 ELSE 0 END + "
            "CASE WHEN a.combined_text ILIKE %s THEN 1 ELSE 0 END)"
        )
        score_params.extend([like, like, like])
        conditions.extend(
            [
                "a.title ILIKE %s",
                "a.combined_text ILIKE %s",
                "c.chunk_text ILIKE %s",
            ]
        )
        where_params.extend([like, like, like])
    relevance_sql = " + ".join(score_expressions)

    try:
        with psycopg.connect(get_postgres_connection_string()) as conn:
            with conn.cursor() as cur:
                order_sql = (
                    "article_rank ASC, relevance DESC, updated_at DESC, chunk_index ASC"
                    if catalog_query
                    else "relevance DESC, updated_at DESC, chunk_index ASC"
                )
                cur.execute(
                    f"""
                    WITH scored AS (
                        SELECT
                        c.chunk_text,
                        c.chunk_id,
                        c.chunk_index,
                        a.article_id,
                        a.title,
                        a.canonical_url,
                        a.source_platform,
                        a.note_key,
                        a.updated_at,
                        {relevance_sql} AS relevance
                        FROM rag_chunks c
                        JOIN rag_articles a ON a.article_id = c.article_id
                        WHERE {" OR ".join(conditions)}
                    ), ranked AS (
                        SELECT *, ROW_NUMBER() OVER (
                            PARTITION BY article_id
                            ORDER BY relevance DESC, chunk_index ASC
                        ) AS article_rank
                        FROM scored
                    )
                    SELECT chunk_text, chunk_id, chunk_index, article_id, title,
                           canonical_url, source_platform, note_key
                    FROM ranked
                    ORDER BY {order_sql}
                    LIMIT %s
                    """,
                    [*score_params, *where_params, limit],
                )
                return [
                    Document(
                        page_content=row[0],
                        metadata={
                            "chunk_id": row[1],
                            "chunk_index": row[2],
                            "article_id": row[3],
                            "title": row[4],
                            "source": row[5],
                            "source_url": row[5],
                            "source_platform": row[6],
                            "note_key": row[7],
                        },
                    )
                    for row in cur.fetchall()
                ]
    except Exception:
        return []


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    stripped = re.sub(r"[？?！!，,。；;：:]", " ", query).strip()
    if len(stripped) >= 2:
        terms.append(stripped)
    cleaned = _clean_search_query(stripped)
    if len(cleaned) >= 2:
        terms.append(cleaned)
    blocks = re.findall(r"[A-Za-z0-9_#\-\u4e00-\u9fff]{2,}", query)
    terms.extend(blocks)

    split_pattern = r"(?:是什么|有哪些|为什么|怎么样|如何|怎么|怎样|以及|并且|其中|请问|的|和|与|及)"
    segments = [segment for segment in re.split(split_pattern, stripped) if len(segment) >= 2]
    terms.extend(segments)

    generic_prefixes = ("请介绍", "请说明", "介绍", "说明", "关于", "人工智能", "智能体")
    generic_suffixes = (
        "你了解什么",
        "了解什么",
        "介绍一下",
        "说明一下",
        "标准化架构",
        "架构",
        "流程",
        "机制",
        "要求",
        "规范",
        "定义",
        "内容",
        "步骤",
        "原则",
        "国标",
    )
    for segment in [*segments, cleaned]:
        for prefix in generic_prefixes:
            if segment.startswith(prefix) and len(segment) - len(prefix) >= 2:
                terms.append(segment[len(prefix) :])
        for suffix in generic_suffixes:
            if segment.endswith(suffix) and len(segment) - len(suffix) >= 2:
                terms.append(segment[: -len(suffix)])
    if "智能体互联" in stripped:
        terms.append("智能体互联")
    standard_code = re.search(r"GB\s*/?\s*Z\s*\d+(?:\s*\.\s*\d+)?", query, re.IGNORECASE)
    if standard_code:
        terms.append(re.sub(r"\s+", " ", standard_code.group(0)).upper())
    deduped = []
    seen = set()
    for term in terms:
        term = term.strip()
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        deduped.append(term[:80])
    return deduped[:16]


def _clean_search_query(query: str) -> str:
    cleaned = re.sub(r"^(?:请问|请|麻烦|帮我|我想知道|关于)+", "", query).strip()
    cleaned = re.sub(
        r"(?:你了解什么|了解什么|是什么|有哪些|介绍一下|说明一下|说说|概述一下|总结一下)$",
        "",
        cleaned,
    ).strip()
    return cleaned


def _is_catalog_query(query: str) -> bool:
    compact = re.sub(r"\s+", "", query)
    if re.search(r"(?:标准系列|系列标准|全部|完整|总览|概述|有哪些|了解什么|都包括|包含哪些)", compact):
        return True
    specific_parts = (
        "总体架构",
        "身份码",
        "身份管理",
        "智能体描述",
        "智能体发现",
        "智能体交互",
        "工具调用",
    )
    asks_about_series = "智能体互联" in compact and ("国标" in compact or "标准" in compact)
    return asks_about_series and not any(part in compact for part in specific_parts)


def _merge_retrieved_documents(
    documents: list[Document],
    *,
    limit: int = 12,
    diversify_articles: bool = False,
) -> list[Document]:
    merged = []
    seen = set()
    for doc in documents:
        key = doc.metadata.get("chunk_id") or doc.metadata.get("source_url") or doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    if not diversify_articles:
        return merged[:limit]

    diversified = []
    remaining = []
    seen_articles = set()
    for doc in merged:
        article_key = doc.metadata.get("article_id") or doc.metadata.get("source_url")
        if article_key and article_key not in seen_articles:
            seen_articles.add(article_key)
            diversified.append(doc)
        else:
            remaining.append(doc)
    return (diversified + remaining)[:limit]


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database


def get_db_connection():
    """Get a connection to the business database."""
    if not settings.BUSINESS_DB_URL:
        # Fallback to a local sqlite for demo if no URL provided
        return sqlite3.connect("business.db")
    
    db_url = settings.BUSINESS_DB_URL.get_secret_value()
    if db_url.startswith("postgresql"):
        return psycopg.connect(db_url)
    elif db_url.startswith("sqlite"):
        # Extract path from sqlite:///path/to/db
        path = db_url.replace("sqlite:///", "")
        return sqlite3.connect(path)
    else:
        raise ValueError(f"Unsupported database type in BUSINESS_DB_URL: {db_url}")


@tool
def list_tables() -> str:
    """List all available tables in the business database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if isinstance(conn, sqlite3.Connection):
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        else:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return "Available tables: " + ", ".join(tables)
    except Exception as e:
        return f"Error listing tables: {e}"


@tool
def get_table_schema(table_name: str) -> str:
    """Get the schema (columns and types) for a specific table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if isinstance(conn, sqlite3.Connection):
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
        else:
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s;
            """, (table_name,))
            columns = [f"{row[0]} ({row[1]})" for row in cursor.fetchall()]
            
        conn.close()
        if not columns:
            return f"Table '{table_name}' not found or has no columns."
        return f"Schema for {table_name}: " + ", ".join(columns)
    except Exception as e:
        return f"Error getting schema for {table_name}: {e}"


@tool
def execute_sql(sql_query: str) -> str:
    """
    Execute a SQL SELECT query on the business database and return the results.
    ONLY SELECT statements are allowed for security.
    """
    # Simple security check
    clean_query = sql_query.strip().lower()
    if not clean_query.startswith("select"):
        return "Error: Only SELECT queries are allowed for security reasons."
    
    # Block potentially dangerous keywords
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "grant", "revoke"]
    for word in forbidden:
        if re.search(r'\b' + word + r'\b', clean_query):
            return f"Error: Forbidden keyword '{word}' found in query."

    try:
        conn = get_db_connection()
        # Use dict cursor if possible for better output
        if not isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
        else:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
        cursor.execute(sql_query)
        
        # Fetch results
        rows = cursor.fetchmany(100) # Limit to 100 rows for safety
        
        if not rows:
            conn.close()
            return "Query executed successfully, but returned no results."
            
        # Format results
        if isinstance(conn, sqlite3.Connection):
            col_names = rows[0].keys()
            result_list = [dict(row) for row in rows]
        else:
            col_names = [desc[0] for desc in cursor.description]
            result_list = [dict(zip(col_names, row)) for row in rows]
            
        conn.close()
        
        import json
        return json.dumps(result_list, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error executing SQL: {e}"

# 创建一个专门存放生成图表的目录
# 这个目录会被 Streamlit 访问，用来向用户展示生成的图表
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

@tool
def execute_python_code(code: str) -> str:
    """
    执行一段用于数据分析的 Python 代码，并返回代码在控制台的输出结果。
    你可以使用 pandas, numpy, matplotlib 和 seaborn。
    
    【重要绘图规则】
    1. 如果你需要绘制图表，请【必须】将图表保存到 'charts/' 目录下（例如：plt.savefig('charts/sales_trend.png')）。
    2. 为了防止绘图时控制台卡死，代码最开始必须引入：
       import matplotlib
       matplotlib.use('Agg')
    3. 为了让图表正常显示中文，请设置：
       plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统推荐
       plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
    
    Args:
        code (str): 需要运行的、完整的、合法的 Python 代码字符串。
        
    Returns:
        str: 控制台标准输出(STDOUT)和标准错误(STDERR)的内容，或执行报错时的异常堆栈。
    """
    # 强制在执行代码前注入 Agg 后端，双重保险
    if "matplotlib" in code and "matplotlib.use" not in code:
        code = "import matplotlib; matplotlib.use('Agg')\n" + code

    # 重定向控制台输出
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = sys.stdout = io.StringIO()
    redirected_error = sys.stderr = io.StringIO()

    try:
        # 执行代码
        # 创建一个沙箱字典作为局部变量域，避免污染全局变量
        exec_globals = {}
        exec(code, exec_globals)
        
        stdout_val = redirected_output.getvalue()
        stderr_val = redirected_error.getvalue()
        
        output = []
        if stdout_val:
            output.append(f"【控制台输出】:\n{stdout_val}")
        if stderr_val:
            output.append(f"【控制台警告/错误】:\n{stderr_val}")
        if not stdout_val and not stderr_val:
            output.append("代码执行成功，但控制台没有产生任何输出。")
            
        return "\n".join(output)
        
    except Exception:
        # 捕获运行代码时的任何异常，并格式化成易读的报错信息返回给 Agent，方便它自我纠错
        return f"【代码执行失败，报错堆栈如下】:\n{traceback.format_exc()}"
        
    finally:
        # 恢复控制台默认输出
        sys.stdout = old_stdout
        sys.stderr = old_stderr
