#!/bin/bash
# 用途：两卡 DDP SimPO 冒烟（Qwen3-0.6B + orca_dpo_pairs）—— 无参考模型的偏好对齐，用长度归一化 + reward margin
# 要点：beta 默认 2.0（比 DPO 的 0.1 大得多），simpo_gamma 是 chosen/rejected 之间要拉开的间隔，论文建议 0.5~1.5
# 运行：cd /root/ms-swift && bash experiments/rl/rl_simpo_2gpu.sh
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B

mkdir -p "$BASE/logs"

$SWIFT rlhf \
    --rlhf_type simpo \
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
    --beta 2.0 \
    --simpo_gamma 1.0 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 1024 \
    --output_dir "$BASE/output/rl_simpo" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2
