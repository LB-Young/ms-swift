#!/bin/bash
# 两卡 DDP SFT 冒烟脚本（Qwen3-0.6B + alpaca-gpt4-data-zh）
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B

mkdir -p "$BASE/logs"

$SWIFT sft \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset "$BASE/datasets/alpaca_gpt4_data_zh/train.csv#2000" \
    --val_dataset "$BASE/datasets/self_cognition/self_cognition.jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 100 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 1024 \
    --output_dir "$BASE/output/sft" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
