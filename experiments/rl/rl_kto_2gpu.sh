#!/bin/bash
# 用途：两卡 DDP KTO 冒烟（Qwen3-0.6B + ultrafeedback KTO 二进制标签集）—— 只用「好/坏」标签，不需要成对比较
# 要点：数据每行带 label=true/false（不需要 chosen/rejected 双栏）；desirable/undesirable_weight 用来平衡正负样本数
# 运行：cd /root/ms-swift && bash experiments/rl/rl_kto_2gpu.sh
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B

mkdir -p "$BASE/logs"

$SWIFT rlhf \
    --rlhf_type kto \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset 'AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto#400' \
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
    --desirable_weight 1.0 \
    --undesirable_weight 1.0 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 1024 \
    --output_dir "$BASE/output/rl_kto" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2
