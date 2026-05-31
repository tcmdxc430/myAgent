import sys

# 解决 Windows 控制台中文打印乱码
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# 模拟 1：向量检索（Dense Search）返回的 Top 5 结果
# 向量检索更关注语义，比如“Security”和“Safeguard”、“防线”在语义上非常接近
dense_results = [
    "Doc_A: 智能体安全防线 Safeguard 的安全合规策略与输入拦截机制",  # 第一名 (Rank 1)
    "Doc_B: 基于 RAG 架构的公司核心资产数据向量化存储与隐私保护",  # 第二名
    "Doc_C: Agent 长期记忆体中的 Vector DB 语义检索与多轮会话上下文保持",  # 第三名
    "Doc_D: 大语言模型 Fine-tuning 微调过程中的有害数据清洗指南",  # 第四名
    "Doc_E: 基于 LangGraph 的反思智能体 Reflective Agent 架构解析"   # 第五名
]

# 模拟 2：关键词/BM25检索（Sparse Search）返回的 Top 5 结果
# 关键词检索擅长精准匹配字面，这里精准匹配了“Safeguard”、“Agent”等关键词
sparse_results = [
    "Doc_F: 深度解析 LlamaGuard 在智能体防护中的作用与安全边界",    # 第一名 (Rank 1)
    "Doc_A: 智能体安全防线 Safeguard 的安全合规策略与输入拦截机制",  # 第二名 (包含关键词 Safeguard)
    "Doc_E: 基于 LangGraph 的反思智能体 Reflective Agent 架构解析",   # 第三名 (包含关键词 Agent)
    "Doc_G: 详解如何使用 BM25 稀疏检索加速敏感词与违禁专有词的字面匹配", # 第四名
    "Doc_C: Agent 长期记忆体中的 Vector DB 语义检索与多轮会话上下文保持"  # 第五名 (包含关键词 Agent)
]

def main():
    print("=== 1. 原始检索结果 ===")
    print("向量检索（Dense）结果:")
    for i, doc in enumerate(dense_results, 1):
        print(f"  Rank {i}: {doc}")
        
    print("\n关键词检索（Sparse）结果:")
    for i, doc in enumerate(sparse_results, 1):
        print(f"  Rank {i}: {doc}")

    # 调用你的 RRF 融合算法
    # 这里我们设置平滑常数 k = 60
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

    print("\n=== 2. RRF 混合检索融合后的终极排名 ===")
    for rank, (doc, score) in enumerate(fused_results, 1):
        print(f"  终极 Rank {rank} (分数: {score:.5f}): {doc}")

def reciprocal_rank_fusion(dense_list: list, sparse_list: list, k: int = 60) -> list:
    """
    RRF 倒数排名融合算法实现
    
    Args:
        dense_list: 密集/向量检索返回的文档列表
        sparse_list: 稀疏/关键词检索返回的文档列表
        k: 排名平滑常数，默认 60
        
    Returns:
        排序后的最终融合列表，元素格式为：(文档名称, RRF总得分)
    """
    # 创建一个字典来存储每个文档累加的 RRF 得分
    # 格式为 { "Doc_A": score, "Doc_B": score }
    rrf_scores = {}
    
    # 1. 计算向量检索列表（dense_list）中每个文档的倒数排名得分
    for rank, doc in enumerate(dense_list, 1):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        # 核心公式：1 / (k + rank)
        rrf_scores[doc] += 1.0 / (k + rank)
        
    # 2. 计算关键词检索列表（sparse_list）中每个文档的倒数排名得分
    for rank, doc in enumerate(sparse_list, 1):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        # 累加得分。如果一个文档在两个列表都出现了，它的得分会变得很高！
        rrf_scores[doc] += 1.0 / (k + rank)
        
    # 3. 对字典按照得分从大到小进行排序
    sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_results

if __name__ == "__main__":
    main()

