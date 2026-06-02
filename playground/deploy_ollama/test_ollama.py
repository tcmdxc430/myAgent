import sys
import os

# 解决 Windows 控制台中文乱码问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# 将项目根目录添加到 python 路径，以便能正确导入 core
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
    print("✅ 成功导入 langchain_ollama 依赖库！")
except ImportError:
    print("❌ 导入 langchain_ollama 失败，请确保在虚拟环境下运行此脚本。")
    sys.exit(1)

def test_local_ollama_stream():
    print("\n🔄 正在尝试连接本地 Ollama 服务 (http://localhost:11434)...")
    print("📢 请确保你已经运行了: ollama run qwen2.5:1.5b\n")
    
    try:
        # 初始化 Ollama 客户端，指定 qwen2.5:1.5b 模型
        llm = ChatOllama(
            model="qwen2.5:1.5b",
            temperature=0.5,
            base_url="http://localhost:11434"
        )
        
        # 准备测试消息
        messages = [
            HumanMessage(content="你好！请用一句话证明你已经成功在本地启动，并告诉我你当前的参数量大小。")
        ]
        
        print("🤖 [Qwen2.5-1.5B 正在流式回答]:")
        print("-" * 50)
        
        # 流式调用并打印
        for chunk in llm.stream(messages):
            # 打印流式输出的文本内容
            print(chunk.content, end="", flush=True)
            
        print("\n" + "-" * 50)
        print("🎉 测试成功！本地 Ollama + Qwen2.5-1.5B 已经完美打通！")
        
    except Exception as e:
        print(f"\n❌ 连接 Ollama 失败。报错信息: {e}")
        print("\n💡 排查建议:")
        print("1. 请检查 Windows 右下角托盘，确保 Ollama 客户端正在运行。")
        print("2. 请在命令行中运行 `ollama list`，检查 `qwen2.5:1.5b` 是否在列表中。")
        print("3. 如果模型不在列表中，请运行 `ollama pull qwen2.5:1.5b` 下载模型。")

if __name__ == "__main__":
    test_local_ollama_stream()
