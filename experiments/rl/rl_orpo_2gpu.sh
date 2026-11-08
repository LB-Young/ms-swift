#!/bin/bash
# 用途：两卡 DDP ORPO 冒烟（Qwen3-0.6B + orca_dpo_pairs）—— 单阶段偏好对齐，不需要 ref model
# 要点：chosen/rejected 拼在一条序列里直接优化，无需参考模型，显存比 DPO 省一半
# 运行：cd /root/ms-swift && bash experiments/rl/rl_orpo_2gpu.sh
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B

mkdir -p "$BASE/logs"

$SWIFT rlhf \
    --rlhf_type orpo \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset 'swift/orca_dpo_pairs#400' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 20 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 1024 \
    --output_dir "$BASE/output/rl_orpo" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2
