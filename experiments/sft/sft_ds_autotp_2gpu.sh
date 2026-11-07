#!/bin/bash
# 两卡 DeepSpeed AutoTP（张量并行）SFT 冒烟脚本（Qwen3-0.6B + alpaca-zh，full 微调）
# 用法：--deepspeed zero2 + --deepspeed_autotp_size 2（AutoTP 自动把层内权重切分到 2 卡）
# 注意：AutoTP 仅支持 full 参数微调（--tuner_type full）
# 运行：cd /root/ms-swift && bash experiments/sft/sft_ds_autotp_2gpu.sh
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
    --tuner_type full \
    --dataset "$BASE/datasets/alpaca_gpt4_data_zh/train.csv#2000" \
    --val_dataset "$BASE/datasets/self_cognition/self_cognition.jsonl" \
    --deepspeed zero2 \
    --deepspeed_autotp_size 2 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 25 \
    --save_steps 25 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 1024 \
    --output_dir "$BASE/output/sft_ds_autotp" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
