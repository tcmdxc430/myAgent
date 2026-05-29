import math
import re
import sqlite3
from typing import Any

import numexpr
import psycopg
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import HuggingFaceEmbeddings

from core import settings


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
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Source")
        # Extract just the filename if it's a full path
        source_name = source.split("\\")[-1].split("/")[-1]
        item = f"--- Source {i+1}: {source_name} ---\n{doc.page_content}"
        context_items.append(item)
    return "\n\n".join(context_items)


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
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    # Get the chroma retriever
    retriever = load_chroma_db()

    # Search the database for relevant documents
    documents = retriever.invoke(query)

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


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
