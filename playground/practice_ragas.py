import os
import sys
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# 解决 Windows 控制台中文打印乱码
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# 1. 载入 .env 环境变量（获取 DeepSeek 密钥）
load_dotenv()
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("未在 .env 文件中检测到 DEEPSEEK_API_KEY，请确保其已配置。")

# 2. 初始化 LangChain 的 ChatOpenAI 实例连接 DeepSeek
from langchain_openai import ChatOpenAI
openai_llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    temperature=0.0  # 评测需要严谨，温度设为 0
)

# 3. 初始化你本地已经存在的 HuggingFace 向量模型（完全免网、离线运行！）
from langchain_huggingface import HuggingFaceEmbeddings
local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. 使用 Ragas 包装类同时包装大模型和向量模型
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

evaluator_llm = LangchainLLMWrapper(langchain_llm=openai_llm, bypass_n=True)
evaluator_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

# 5. 准备评测数据集 (Evaluation Dataset)
eval_data = {
    "question": [
        "凡在公司连续工作满 1 年的正式员工，每年可享受多少天的带薪年假？该怎么申请？",
        "凡在公司连续工作满 1 年的正式员工，每年可享受多少天的带薪年假？该怎么申请？"
    ],
    "contexts": [
        # 系统检索出的真实公司手册上下文（Contexts）
        [
            "凡在本公司连续工作满 1 年的正式员工，每年可享受 10 天的带薪年休假。工作满 5 年以上的员工，每年可享受 15 天的带薪年休假。年假不可跨年度累计，须在当年 12 月 31 日前使用完毕。员工请假应提前在公司 OA 系统中提交申请，由部门主管和 HR 审批。"
        ],
        [
            "凡在本公司连续工作满 1 年的正式员工，每年可享受 10 天的带薪年休假。工作满 5 年以上的员工，每年可享受 15 天的带薪年休假。年假不可跨年度累计，须在当年 12 月 31 日前使用完毕。员工请假应提前在公司 OA 系统中提交申请，由部门主管和 HR 审批。"
        ]
    ],
    "answer": [
        # 答案 A：一个优秀的、完全不编造的 RAG 系统回答
        "根据公司规定，连续工作满 1 年的正式员工每年可享受 10 天带薪年假。请假时，员工需要提前在公司的 OA 系统中提交申请。",
        
        # 答案 B：一个带有严重幻觉（胡编乱造）的坏 RAG 回答（编造了 30 天，以及找财务请假）
        "您好！连续工作满 1 年的员工，每年可以享受多达 30 天的带薪超长年假。请假非常简单，您不需要走系统，直接向财务部的前台口头说明并登记就可以了。"
    ],
    "system_name": [
        "优秀无幻觉 RAG 助手",
        "幻觉满天飞 RAG 助手"
    ]
}

# 转换成 Ragas 要求的 Hugging Face Dataset 格式
dataset = Dataset.from_dict({
    "question": eval_data["question"],
    "contexts": eval_data["contexts"],
    "answer": eval_data["answer"]
})

# 6. 导入评测模块
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy

# 明确为两个 Metrics 绑定评测 LLM 和本地向量模型
faithfulness_metric = Faithfulness(llm=evaluator_llm)
answer_relevancy_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

# 强制将 AnswerRelevancy 的 strictness 设为 1，防止多样本冲突
answer_relevancy_metric.strictness = 1

selected_metrics = [faithfulness_metric, answer_relevancy_metric]

def main():
    print("⏳ 正在调用 DeepSeek (已绑定本地离线 Embeddings) 进行 Ragas 自动化评分，请稍候...")
    
    # 7. 启动评测
    result = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    # 8. 转换成 Pandas DataFrame 方便精美打印
    df_result = result.to_pandas()
    
    # 拼装我们自定义的系统名称，方便阅读对比
    df_result["评估对象"] = eval_data["system_name"]
    df_result["AI生成回答"] = eval_data["answer"]
    
    # 整理列顺序
    cols = ["评估对象", "AI生成回答", "faithfulness", "answer_relevancy"]
    df_display = df_result[cols]
    
    print("\n🎉 Ragas 自动化评测结果出炉：")
    print("=" * 100)
    for idx, row in df_display.iterrows():
        print(f"🤖 评估对象: {row['评估对象']}")
        print(f"📝 它的回答: {row['AI生成回答']}")
        print(f"📊 【Faithfulness 忠实度（无幻觉得分）】: {row['faithfulness']:.4f}  (分值越接近 1 越没有幻觉)")
        print(f"🎯 【Answer Relevancy 回答相关性得分】 : {row['answer_relevancy']:.4f} (分值越接近 1 答得越切题)")
        print("-" * 100)
    print("=" * 100)

if __name__ == "__main__":
    main()