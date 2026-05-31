import os
import sys

# 解决 Windows 控制台编码问题，确保能正常打印 UTF-8 字符和 Emoji
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 定义一段模拟的公司员工手册文本进行测试
sample_text = """
一、工作时间与考勤规定
1. 标准工作时间
本公司实行每周5天、每天8小时的标准工时制。具体每日工作时间为：上午 9:00 至 下午 18:00，其中中午 12:00 至 13:00 为午餐与休息时间。
2. 考勤管理
所有员工必须按照规定在每日上下班时进行指纹或人脸识别打卡。迟到或早退超过 15 分钟且未办理请假手续者，视为旷工半天。

二、休假福利与申请流程
1. 年假规定
凡在本公司连续工作满 1 年的正式员工，每年可享受 10 天的带薪年休假。工作满 5 年以上的员工，每年可享受 15 天的带薪年休假。年假不可跨年度累计，须在当年 12 月 31 日前使用完毕，逾期自动作废。
2. 请假流程
员工因病或因私请假，应提前在公司 OA 系统中提交申请。请病假须提供正规医院出具 of 诊断证明。请假在 3 天（含）以内的由部门主管审批，3 天以上的须报送 HR 部门及总经理审批。

三、报销与差旅标准
1. 差旅住宿标准
员工因公出差，在一线城市（北京、上海、广州、深圳）的每日住宿报销上限为 500 元；在二线及其他城市每日住宿报销上限为 350 元。超出标准部分需自行承担。
2. 报销提交时效
所有因公消费的发票及收据，必须在消费发生之日起 30 天内通过财务系统完成报销申请提交。逾期财务部门有权拒绝受理。
"""

def print_chunks(chunks, title):
    print("=" * 60)
    print(f"★ {title}")
    print(f"-> 总共切分出 {len(chunks)} 个分块")
    print("=" * 60)
    for i, chunk in enumerate(chunks, 1):
        print(f"[*] 【Chunk {i}】 (长度: {len(chunk.page_content)} 字):")
        print(f"   内容: {chunk.page_content.strip()}")
        print("-" * 60)
    print("\n")

def main():
    print("=== 欢迎来到 langchain 文本切片（Chunking）实践课堂！ ===")
    print("我们将使用 RecursiveCharacterTextSplitter 对上面那段员工手册进行切片。")
    print("这个拆分器会优先按照段落 '\\n\\n'、换行 '\\n'、空格 ' ' 的顺序尝试拆分，以尽可能保持语义完整性。\n")

    # 实验 1: 较小的块大小，无重叠 (Chunk Size = 100, Overlap = 0)
    # 这会导致语义在边界被硬生生切断
    splitter_1 = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        length_function=len
    )
    chunks_1 = splitter_1.create_documents([sample_text])
    print_chunks(chunks_1, "实验 1: 块大小=100, 重叠度=0 (无重叠容易导致边界语义断裂)")

    # 实验 2: 较小的块大小，有重叠 (Chunk Size = 100, Overlap = 30)
    # 重叠部分可以帮助保留上下文
    splitter_2 = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30,
        length_function=len
    )
    chunks_2 = splitter_2.create_documents([sample_text])
    print_chunks(chunks_2, "实验 2: 块大小=100, 重叠度=30 (设置重叠可以连接上下文)")

    # 实验 3: 较大的块大小 (Chunk Size = 300, Overlap = 50)
    # 适合用来包含更完整的段落/规定
    splitter_3 = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks_3 = splitter_3.create_documents([sample_text])
    print_chunks(chunks_3, "实验 3: 块大小=300, 重叠度=50 (适合保留完整的规章段落)")

if __name__ == "__main__":
    main()
