import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 解决 Windows 终端中文显示问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def train():
    print("🚀 开始初始化 QLoRA 金融微调流程...")
    
    # 1. 定义模型和数据路径
    # 这里我们使用 Hugging Face 的原版 Qwen2.5-1.5B-Instruct 模型作为基座进行微调
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "playground/finetune_qwen/financial_dataset.json"
    output_dir = "playground/finetune_qwen/saved_weights"
    
    # 2. 检查显卡支持
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 当前训练设备: {device.upper()}")
    if device == "cpu":
        print("⚠️ 警告: 未检测到 GPU，CPU 训练极其缓慢！强烈建议使用 Google Colab 或租用 GPU 服务器。")
        
    # 3. 极速 4-bit 量化配置（压榨 4GB 显存的关键）
    # bitsandbytes 可以将 1.5B 模型压缩到 ~900MB 加载，腾出宝贵显存用于梯度计算
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 4. 加载分词器和模型
    print("⏳ 正在下载/加载基座模型分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("⏳ 正在加载 4-bit 量化基座模型 (这可能需要一些时间，请耐心等待)...")
    # 如果本地显存只有 4GB，建议设置 device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 5. 准备 PEFT 训练
    # 这一步会冻结基座模型的全部参数，只训练极少数的 LoRA 旁路参数
    model = prepare_model_for_kbit_training(model)
    
    # 6. 配置 LoRA (Low-Rank Adaptation)
    # r=8, alpha=32 代表旁路参数的低秩大小，既保证了拟合能力又极大节省了显存
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("✅ LoRA 旁路网络绑定成功！可训练参数占比通常低于 1%！")
    model.print_trainable_parameters()
    
    # 7. 数据集处理与 Token 话
    # 将 Instruction-Input-Output 拼接成 Qwen 专用的 Chat 格式
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def process_func(example):
        MAX_LEN = 256  # 限制最长上下文为 256，严防 4GB 显卡 OOM！
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]
        
        # 拼接 Prompt
        prompt = f"<|im_start|>system\n你是一个专业、严谨的金融专家。请给出预期、专业的行业回答。<|im_end|>\n<|im_start|>user\n{instruction} {input_text}<|im_end|>\n<|im_start|>assistant\n"
        response = f"{output}<|im_end|>"
        
        # 转换为 input_ids
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids  # 将 Prompt 部分设为 -100，训练时不计算它们的 Loss
        
        # 截断
        if len(input_ids) > MAX_LEN:
            input_ids = input_ids[:MAX_LEN]
            labels = labels[:MAX_LEN]
            
        return {
            "input_ids": input_ids,
            "labels": labels
        }
        
    print("📊 正在处理数据集并转换为 Token 格式...")
    tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)
    
    # 8. 极限优化训练参数（专为低显存设计）
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,        # 极限 Batch Size = 1
        gradient_accumulation_steps=4,       # 通过梯度累加等效达到 Batch Size = 4 的效果
        learning_rate=2e-4,                  # LoRA 经典学习率
        num_train_epochs=5,                  # 样本少，可以多迭代几轮确保过拟合金融风格
        logging_steps=1,
        fp16=True,                           # 开启半精度混合训练
        save_strategy="no",                  # 训练中不保存临时 Checkpoint，节省硬盘和显存
        gradient_checkpointing=True,         # 开启梯度检查点！用时间换空间，显存直降 50%
        report_to="none"                     # 禁用外部追踪，完全离线训练
    )
    
    # 9. 启动训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    
    print("\n🔥 开始微调训练！正在向显卡注入参数...")
    print("⚠️ 提示：如果本地显存依然溢出，代表 4GB 物理硬件已达极限。请无缝将此代码及数据集上传至 AutoDL/Google Colab 运行，代码完全通用！\n")
    
    trainer.train()
    
    # 10. 保存权重
    print(f"🎉 训练完成！金融 LoRA 权重已保存至: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train()
