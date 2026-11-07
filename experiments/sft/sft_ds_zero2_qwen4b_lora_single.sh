#!/bin/bash
# 单卡 DeepSpeed ZeRO-2：Qwen3-4B + LoRA（v4 单卡版）
# 背景：WSL VM 内存仅 15GB，双卡 4B 训练两进程各 7.5GB RSS → 内核 global_oom 杀进程（有 /var/log/syslog 铁证）。
#      单卡单进程 RSS ~7.5GB < 15GB，显存 8GB 权重 + LoRA < 16GB，可跑。
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0
export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=4

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-4B

mkdir -p "$BASE/logs"

$SWIFT sft \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset "$BASE/datasets/alpaca_gpt4_data_zh/train.csv#500" \
    --val_dataset "$BASE/datasets/self_cognition/self_cognition.jsonl" \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 10 \
    --save_steps 5 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --max_length 256 \
    --output_dir "$BASE/output/sft_ds_zero2_qwen4b_lora_single" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
