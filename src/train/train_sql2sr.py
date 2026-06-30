"""
train_sql2sr.py - SQL-to-SR (Semantic Representation) LoRA SFT 微调脚本

基于 SHARE-main 的 BAM 训练方式，使用 HuggingFace transformers + PEFT + TRL
对 Meta-Llama-3.1-8B-Instruct 进行 LoRA SFT，让模型将 SQL 转换为 SR 轨迹。

训练参数与 SHARE-main/scripts/example_train.sh 保持一致。
"""
import os
import sys
import json
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============ 提示词模板（与 SHARE-main 训练 BAM 保持一致）============
SYSTEM_PROMPT = "You are an expert about text-to-SQL and pandas code."

SQL2SR_USER_PROMPT = """SR is a piece of pandas-like code, which is a intermediate representation between the natural language and SQL. I will provide you:
1. Schema: A python list and each element is a `table_name`.`column_name` string. It indicates that the table and column you could use in the SR.
2. SQL: The SQL that needed to be converted to SR
 
Your task is to generate valid SR which reflect the accurate logic in the SQL. Later, the SR will be converted to SQL.
Please pay attention that SR ignore 'join' action. Do not generate 'join' action.

schema = {schema}
sql = "{sql}"

Now generate the valid SR that display the reasoning process of generating SQL that can accurately answer the question:
```SR
[Your Answer]
```"""

ASSISTANT_PROMPT = """```SR
{sr}
```"""

# ============ Llama3 Chat Template ============
# LLaMA-3.1-8B-Instruct uses a specific chat template
# We format as: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>...
LLAMA3_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


def load_json(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_training_data(
    brid_path: str,
    spider_path: str,
) -> list:
    """
    加载 brid_trajectory 和 spider_trajectory 数据集，
    转换为 instruction/output/system 格式。
    """
    all_samples = []
    for subset_name, json_path in [
        ("brid_trajectory", brid_path),
        ("spider_trajectory", spider_path),
    ]:
        if not os.path.exists(json_path):
            logger.warning(f"数据集不存在: {json_path}")
            continue
        logger.info(f"加载 {subset_name}: {json_path}")
        data = load_json(json_path)
        valid_count = 0
        for item in data:
            trajectory = item.get("trajectory", "").strip()
            gold_sql = item.get("gold_sql", "").strip()
            schema_str = item.get("schema_str", "")
            if not trajectory or not gold_sql:
                continue
            instruction = SQL2SR_USER_PROMPT.format(
                schema=schema_str, sql=gold_sql
            )
            output = ASSISTANT_PROMPT.format(sr=trajectory)
            all_samples.append({
                "instruction": instruction,
                "output": output,
                "system": SYSTEM_PROMPT,
            })
            valid_count += 1
        logger.info(f"  {subset_name}: 有效 {valid_count} 条")
    logger.info(f"总计 {len(all_samples)} 条训练样本")
    return all_samples


def formatting_func(sample: dict) -> str:
    """
    将样本格式化为 Llama3 chat template 格式的字符串。
    用于 SFTTrainer 的 formatting_func。
    """
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    # 使用 tokenizer 的 chat template（如果已设置）
    # 这里手动构造以确保格式正确
    parts = []
    # System
    parts.append(
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{sample['system']}<|eot_id|>"
    )
    # User
    parts.append(
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{sample['instruction']}<|eot_id|>"
    )
    # Assistant
    parts.append(
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{sample['output']}<|eot_id|>"
    )
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="SQL2SR LoRA SFT 训练（基于 SHARE-main BAM 配置）"
    )
    # 路径参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/user4/XiaZY/xia-zhenyu/dep/model/Meta-Llama-3___1-8B-Instruct",
        help="预训练模型路径",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/user4/XiaZY/xia-zhenyu/dat",
        help="数据集根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user4/XiaZY/xia-zhenyu/exp/outputs/sql2sr_lora",
        help="模型输出目录",
    )
    parser.add_argument(
        "--save_data_path",
        type=str,
        default="/home/user4/XiaZY/xia-zhenyu/exp/ft_data/sql2sr_train_data.json",
        help="处理后训练数据保存路径",
    )

    # 训练超参数（与 SHARE-main 保持一致）
    parser.add_argument("--num_train_epochs", type=float, default=6.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # LoRA 参数（与 SHARE-main 保持一致）
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_4bit", action="store_true", default=False,
                        help="是否使用 4-bit 量化（显存不足时开启）")
    parser.add_argument("--use_llamafactory", action="store_true", default=False,
                        help="使用 LLaMA-Factory 训练（需安装 llamafactory）")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed 配置文件路径（可选）")

    args = parser.parse_args()

    # ============ 创建输出目录 ============
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_data_path), exist_ok=True)

    # ============ 准备训练数据 ============
    logger.info("=" * 60)
    logger.info("Step 1: 准备训练数据")
    logger.info("=" * 60)

    brid_path = os.path.join(args.data_dir, "brid_trajectory", "train_trajectory.json")
    spider_path = os.path.join(args.data_dir, "spider_trajectory", "train_trajectory.json")
    train_samples = prepare_training_data(brid_path, spider_path)

    # 保存处理后的数据（供 LLaMA-Factory 使用）
    with open(args.save_data_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    logger.info(f"处理后数据保存到: {args.save_data_path}")

    # 保存 LLaMA-Factory 的 dataset_info.json
    dataset_info_path = os.path.join(os.path.dirname(args.save_data_path), "dataset_info.json")
    dataset_info = {
        "sql2sr_train_data": {
            "file_name": "sql2sr_train_data.json",
            "columns": {
                "prompt": "instruction",
                "response": "output",
                "system": "system",
            },
        }
    }
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    logger.info(f"dataset_info 保存到: {dataset_info_path}")

    # ============ 如果使用 LLaMA-Factory，则启动对应训练 ============
    if args.use_llamafactory:
        logger.info("=" * 60)
        logger.info("使用 LLaMA-Factory 启动训练")
        logger.info("=" * 60)
        import subprocess
        cmd = [
            "llamafactory-cli", "train",
            "--stage", "sft",
            "--do_train", "True",
            "--model_name_or_path", args.model_name_or_path,
            "--preprocessing_num_workers", "16",
            "--finetuning_type", "lora",
            "--template", "llama3",
            "--flash_attn", "auto",
            "--dataset_dir", os.path.dirname(args.save_data_path),
            "--dataset", "sql2sr_train_data",
            "--cutoff_len", str(args.max_seq_length),
            "--learning_rate", str(args.learning_rate),
            "--num_train_epochs", str(args.num_train_epochs),
            "--max_samples", "100000",
            "--per_device_train_batch_size", str(args.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--lr_scheduler_type", args.lr_scheduler_type,
            "--max_grad_norm", str(args.max_grad_norm),
            "--logging_steps", str(args.logging_steps),
            "--save_steps", str(args.save_steps),
            "--warmup_steps", str(args.warmup_steps),
            "--packing", "False",
            "--report_to", "none",
            "--output_dir", args.output_dir,
            "--bf16", str(args.bf16),
            "--plot_loss", "True",
            "--trust_remote_code", "True",
            "--ddp_timeout", "180000000",
            "--include_num_input_tokens_seen", "True",
            "--optim", "adamw_torch",
            "--lora_rank", str(args.lora_rank),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout),
            "--lora_target", "all",
        ]
        logger.info(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("LLaMA-Factory 训练完成！")
        return

    # ============ 使用 TRL SFTTrainer 训练 ============
    logger.info("=" * 60)
    logger.info("Step 2: 加载模型和分词器")
    logger.info("=" * 60)

    set_seed(args.seed)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 设置 chat_template
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "device_map": "auto",
    }
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("使用 4-bit 量化加载模型")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    logger.info(f"模型加载完成: {type(model).__name__}")

    # ============ 配置 LoRA ============
    logger.info("=" * 60)
    logger.info("Step 3: 配置 LoRA")
    logger.info("=" * 60)

    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ============ 创建 Dataset ============
    logger.info("=" * 60)
    logger.info("Step 4: 构建数据集")
    logger.info("=" * 60)

    hf_dataset = Dataset.from_list(train_samples)

    # ============ 配置训练参数 ============
    logger.info("=" * 60)
    logger.info("Step 5: 开始训练")
    logger.info("=" * 60)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        report_to="none",
        logging_dir=os.path.join(args.output_dir, "logs"),
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
    )

    # 创建 DataCollator（只对 assistant 回复计算 loss）
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        data_collator=collator,
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    # 开始训练
    trainer.train()

    # ============ 保存模型 ============
    logger.info("=" * 60)
    logger.info("Step 6: 保存模型")
    logger.info("=" * 60)

    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"模型已保存到: {final_model_path}")

    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
