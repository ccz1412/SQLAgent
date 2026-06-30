#!/bin/bash
# ============================================================
# SQL2SR LoRA SFT 训练脚本
# 基于 SHARE-main 的 example_train.sh，使用 LLaMA-Factory
# 任务: 将 SQL 语句转换为 SR (Semantic Representation) 路径
# ============================================================
#
# 使用说明:
#   1. 先运行 python prepare_data.py 准备数据
#   2. 然后运行 bash train_sql2sr.sh 开始训练
# ============================================================

set -e

# ============ 路径配置 ============
PROJECT_DIR="/home/user4/XiaZY/xia-zhenyu"
BASE_MODEL="/home/user4/XiaZY/xia-zhenyu/dep/model/Meta-Llama-3___1-8B-Instruct"
DATASET_DIR="${PROJECT_DIR}/exp/ft_data"
OUTPUT_DIR="${PROJECT_DIR}/exp/outputs/sql2sr_lora"
DATASET_NAME="sql2sr_train_data"

# ============ 环境配置 ============
# 使用 py11 conda 环境（已安装 llamafactory, torch, transformers）
CONDA_ENV="/home/user4/miniconda3/envs/py11"
export PATH="${CONDA_ENV}/bin:$PATH"

# 使用 GPU 0,2,3（GPU 1 被占用）
export CUDA_VISIBLE_DEVICES="0,2,3"

# ============ 创建输出目录 ============
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATASET_DIR}"

echo "========================================"
echo " SQL2SR LoRA SFT 训练"
echo "========================================"
echo "模型:     ${BASE_MODEL}"
echo "数据集:   ${DATASET_DIR}/${DATASET_NAME}.json"
echo "输出:     ${OUTPUT_DIR}"
echo "GPU:      ${CUDA_VISIBLE_DEVICES}"
echo "========================================"
echo ""

# ============ 第一步: 准备训练数据 ============
echo "[Step 1/2] 准备训练数据..."
python "${PROJECT_DIR}/src/train/prepare_data.py" \
    --data_dir "${PROJECT_DIR}/dat" \
    --output_dir "${DATASET_DIR}" \
    --output_name "${DATASET_NAME}"

echo ""
echo "[Step 2/2] 开始 LLaMA-Factory LoRA 训练..."
echo ""

# ============ 第二步: 启动训练 ============
# 训练参数参考 SHARE-main/scripts/example_train.sh
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "${BASE_MODEL}" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir "${DATASET_DIR}" \
    --dataset "${DATASET_NAME}" \
    --cutoff_len 4096 \
    --learning_rate 5e-05 \
    --num_train_epochs 6.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 500 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir "${OUTPUT_DIR}" \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

echo ""
echo "========================================"
echo " 训练完成！"
echo " 模型保存路径: ${OUTPUT_DIR}"
echo "========================================"
