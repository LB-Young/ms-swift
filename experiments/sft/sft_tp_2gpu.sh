#!/bin/bash
# 两卡模型并行（device_map 自动切分）SFT 冒烟脚本（Qwen3-0.6B + alpaca-zh）
# 说明：这是 ms-swift 的"开箱即用"多卡模型并行方式——单进程，accelerate device_map
#      自动把模型各层分配到 2 张卡上（模型并行），数据不再切分（与 DDP 的数据并行互补）。
#      适用于模型权重超单卡显存、但每层能放下的场景（官方示例：Qwen2.5-VL-72B 2×76GiB）。
# 注意：0.6B 模型本身单卡能装下，此脚本仅用于验证 TP 流程可跑通；
#      真正需要 TP 时请换成大模型，并配合 --max_memory 控制每卡分配。
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B

mkdir -p "$BASE/logs"

$SWIFT sft \
    --model "$MODEL" \
    --tuner_type lora \
    --device_map auto \
    --dataset "$BASE/datasets/alpaca_gpt4_data_zh/train.csv#500" \
    --val_dataset "$BASE/datasets/self_cognition/self_cognition.jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 25 \
    --save_steps 25 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 1024 \
    --output_dir "$BASE/output/sft_tp" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2
