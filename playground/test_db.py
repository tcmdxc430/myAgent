# test_db.py
import psycopg
try:
    conn = psycopg.connect("postgresql://postgres:password123@localhost:5432/agent_db")
    print("✅ 数据库连接成功！")
    conn.close()
except Exception as e:
    print(f"❌ 连接失败: {e}")