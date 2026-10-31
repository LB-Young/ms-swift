#!/bin/bash
# 两卡 DDP GRPO 冒烟脚本（Qwen3-0.6B + gsm8k 数学题 + accuracy reward）
# native rollout（--use_vllm false，无需安装 vllm）；如需 vllm 加速改为 --use_vllm true
# 列映射：gsm8k 的 question→query（模板输入）、answer→solution（accuracy reward 所需）
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B

mkdir -p "$BASE/logs"

$SWIFT rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --tuner_type lora \
    --use_vllm false \
    --dataset 'modelscope/gsm8k#200' \
    --disable_auto_column_mapping true \
    --columns '{"question": "query", "answer": "solution"}' \
    --reward_funcs accuracy \
    --enable_thinking false \
    --torch_dtype bfloat16 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --max_steps 20 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 1024 \
    --max_completion_length 256 \
    --output_dir "$BASE/output/rl_grpo" \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 2
