#!/bin/bash
# 两卡 DDP DPO（RL）冒烟脚本（Qwen3-0.6B + orca_dpo_pairs 官方数据集）
# v5: 换官方标准 DPO 数据集，避免 zhihu 长文本问题
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B

mkdir -p "$BASE/logs"

$SWIFT rlhf \
    --rlhf_type dpo \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset 'swift/orca_dpo_pairs#500' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 100 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 1024 \
    --output_dir "$BASE/output/rl_dpo" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2
