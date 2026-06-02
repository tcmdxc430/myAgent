import os
import sys
import torch

# 既然你连接了国外的 VPN，我们可以直接使用 Hugging Face 官方源，不需要走国内镜像了！
# 这样可以完全避免镜像站与 VPN 代理之间的冲突。
try:
    from modelscope import snapshot_download
    USE_MODELSCOPE = True
except ImportError:
    USE_MODELSCOPE = False

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, PeftModel

# 解决 Windows 终端中文显示问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def run_minimal_finetune():
    print("🎯 --- 欢迎来到大模型微调极简课堂 --- 🎯")
    print("本脚本专为学习设计：使用超轻量 0.5B 模型，无需复杂配置，保证在本地 100% 跑通！\n")
    
    # ==========================================
    # 步骤 1: 路径与超轻量模型定义
    # ==========================================
    # 我们使用 Qwen2.5-0.5B-Instruct (仅 5亿参数，权重约 900MB)
    # 它体积小、下载极快、显存占用极低，但微调方法与 72B 完全一模一样！
    if USE_MODELSCOPE:
        # ModelScope 上的模型 ID
        model_id = "qwen/Qwen2.5-0.5B-Instruct"
    else:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        
    dataset_path = "playground/finetune_qwen/financial_dataset.json"
    output_dir = "playground/finetune_qwen/minimal_saved_weights"
    
    # 检测设备：优先 GPU，显存不够或没有则优雅降级到 CPU
    # 0.5B 模型在 CPU 上训练 3 条数据也只需要几秒钟，绝对不会卡死！
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/6] 💻 正在检测训练设备: {device.upper()}")
    
    # ==========================================
    # 步骤 2: 加载分词器与模型
    # ==========================================
    if USE_MODELSCOPE:
        print("[2/6] 🚀 检测到已安装 ModelScope，正在通过国内极速引擎下载并加载模型...")
        # snapshot_download 会自动把模型下载到本地，并返回本地绝对路径
        local_model_dir = snapshot_download(model_id)
        print(f"   👉 模型已成功下载至本地缓存: {local_model_dir}")
        model_load_path = local_model_dir
    else:
        print(f"[2/6] ⏳ 正在通过 HuggingFace 镜像源加载模型 ({model_id})...")
        model_load_path = model_id

    tokenizer = AutoTokenizer.from_pretrained(model_load_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 以半精度 (Float16) 加载模型，显存占用仅约 1GB
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # ==========================================
    # 步骤 3: 绑定 LoRA (低秩适应)
    # ==========================================
    print("[3/6] ⚙️ 正在构建 LoRA 旁路网络...")
    # LoRA 的核心思想：冻结大模型原本的参数，只在旁边贴上一个小“补丁”进行训练。
    # r=8 (秩) 代表补丁的宽度，target_modules 指定我们要把补丁贴在模型的哪些注意力投影层上。
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # 最核心的 Query 和 Value 投影层
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # 将 LoRA 绑定到我们的基座模型上
    model = get_peft_model(model, lora_config)
    print("   👉 LoRA 绑定成功！可训练参数占比：")
    model.print_trainable_parameters()
    
    # ==========================================
    # 步骤 4: 载入并处理 SFT 数据集
    # ==========================================
    print("[4/6] 📊 正在载入并格式化 SFT 数据集...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 核心：将我们的问答对拼接成大模型能听懂的 ChatML 格式
    def process_data(example):
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        # 拼接符合 Qwen 规范的 Prompt 模板
        prompt = f"<|im_start|>system\n你是一个专业、严谨的金融专家。请给出预期、专业的行业回答。<|im_end|>\n<|im_start|>user\n{instruction} {input_text}<|im_end|>\n<|im_start|>assistant\n"
        response = f"{output}<|im_end|>"
        
        # 将文本转化为 Token ID 数组
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        
        input_ids = prompt_ids + response_ids
        # 训练技巧：Prompt 部分的 Label 设为 -100。
        # PyTorch 在计算交叉熵损失 (CrossEntropyLoss) 时会自动忽略 -100，
        # 这样大模型就只会对 Assistant 回答的每一个字进行学习和梯度更新，不会去死记硬背问题。
        labels = [-100] * len(prompt_ids) + response_ids
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }
        
    tokenized_dataset = dataset.map(process_data, remove_columns=dataset.column_names)
    
    # ==========================================
    # 步骤 5: 配置训练参数并启动训练 (Trainer)
    # ==========================================
    print("[5/6] 🔥 正在配置极简训练参数并启动训练...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,       # 每次只喂 1 条数据
        gradient_accumulation_steps=1,       # 立即更新梯度
        learning_rate=1e-4,                  # 学习率
        num_train_epochs=3,                  # 迭代 3 轮
        logging_steps=1,                     # 每一步都打印 Loss 变化
        fp16=(device == "cuda"),             # 如果是 GPU 则开启混合精度
        save_strategy="no",                  # 极简学习，不保存中间临时文件
        report_to="none"                     # 完全本地离线，不上传任何日志
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    
    # 启动训练循环！你会在这里看到 Loss（损失值）一步步下降
    trainer.train()
    
    # 训练完成，保存我们的 LoRA “补丁”权重
    print(f"\n💾 训练完成！LoRA 补丁权重已成功保存至: {output_dir}")
    model.save_pretrained(output_dir)
    
    # ==========================================
    # 步骤 6: 效果对比测试（微调前 vs 微调后）
    # ==========================================
    print("\n[6/6] 🧪 正在进行微调效果对比测试...")
    test_prompt = "请分析一下美联储加息对国内 A 股市场的传导路径。"
    formatted_prompt = f"<|im_start|>system\n你是一个专业、严谨的金融专家。请给出预期、专业的行业回答。<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 准备输入 Token
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # 1. 测试微调后的效果
    print("\n🔔 【微调后模型（Base + LoRA 补丁）的回答】:")
    print("-" * 50)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, eos_token_id=tokenizer.eos_token_id)
        # 仅截取 Assistant 生成的内容
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(response.strip())
    print("-" * 50)
    
    # 2. 卸载 LoRA 补丁，看看原本基座模型的回答（对比）
    print("\n🔔 【卸载 LoRA 补丁后，原始基座模型的回答】:")
    print("-" * 50)
    # 卸载 LoRA 恢复原始模型
    base_model = model.unload()
    with torch.no_grad():
        outputs = base_model.generate(**inputs, max_new_tokens=150, eos_token_id=tokenizer.eos_token_id)
        raw_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(raw_response.strip())
    print("-" * 50)
    
    print("\n🎉 恭喜你！你已经完整走通了大模型 SFT 微调的全部生命周期！")
    print("你学到了：1.数据 ChatML 格式化 -> 2.Labels 掩码设计 (-100) -> 3.LoRA 旁路构建 -> 4.Trainer 训练 -> 5.权重保存与加载。")

if __name__ == "__main__":
    run_minimal_finetune()
