#!/bin/bash
# 两卡 DeepSpeed ZeRO-2：Qwen3-4B + LoRA（v2 输出目录）
# 运行：cd /root/ms-swift && bash experiments/sft/sft_ds_zero2_qwen4b_lora_v2.sh
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-4B

mkdir -p "$BASE/logs"

$SWIFT sft \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset "$BASE/datasets/alpaca_gpt4_data_zh/train.csv#2000" \
    --val_dataset "$BASE/datasets/self_cognition/self_cognition.jsonl" \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --max_length 512 \
    --output_dir "$BASE/output/sft_ds_zero2_qwen4b_lora_v2" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
