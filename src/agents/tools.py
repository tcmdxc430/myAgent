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
        if article_id in articles:
            article = articles[article_id]
            title = article.get("title") or title
            source_url = article.get("canonical_url") or source_url
        # Extract just the filename if it's a full path
        source_name = source.split("\\")[-1].split("/")[-1]
        header = f"--- Source {i+1}: {title or source_name} ---"
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


def load_chroma_db():
    # 使用 HuggingFace 的本地向量模型
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize HuggingFaceEmbeddings: {e}"
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 8})
    return retriever


def database_search_func(query: str) -> str:
    """Searches the imported article knowledge base for relevant information."""
    # Get the chroma retriever
    retriever = load_chroma_db()

    # Search the database for relevant documents
    keyword_documents = _load_keyword_rag_documents(query)
    vector_documents = retriever.invoke(query)
    documents = _merge_retrieved_documents(keyword_documents + vector_documents)

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


def _load_keyword_rag_documents(query: str, limit: int = 5) -> list[Document]:
    if settings.DATABASE_TYPE != "postgres":
        return []
    terms = _query_terms(query)
    if not terms:
        return []

    conditions = []
    params: list[str | int] = []
    for term in terms:
        like = f"%{term}%"
        conditions.extend(
            [
                "a.title ILIKE %s",
                "a.combined_text ILIKE %s",
                "c.chunk_text ILIKE %s",
            ]
        )
        params.extend([like, like, like])
    params.append(limit)

    try:
        with psycopg.connect(get_postgres_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        c.chunk_text,
                        c.chunk_id,
                        c.chunk_index,
                        a.article_id,
                        a.title,
                        a.canonical_url,
                        a.source_platform,
                        a.note_key
                    FROM rag_chunks c
                    JOIN rag_articles a ON a.article_id = c.article_id
                    WHERE {" OR ".join(conditions)}
                    ORDER BY a.updated_at DESC, c.chunk_index ASC
                    LIMIT %s
                    """,
                    params,
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
    terms = []
    stripped = query.strip()
    if len(stripped) >= 2:
        terms.append(stripped)
    terms.extend(re.findall(r"[A-Za-z0-9_#\-\u4e00-\u9fff]{2,}", query))
    deduped = []
    seen = set()
    for term in terms:
        term = term.strip()
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        deduped.append(term[:80])
    return deduped[:8]


def _merge_retrieved_documents(documents: list[Document]) -> list[Document]:
    merged = []
    seen = set()
    for doc in documents:
        key = doc.metadata.get("chunk_id") or doc.metadata.get("source_url") or doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged[:8]


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
