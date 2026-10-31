#!/bin/bash
# 4B + LoRA：zero3_offload 单测（预期：CPU 优化器需要 nvcc 编译 DeepSpeedCPUAdam，可能失败——验证结论）
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-4B

$SWIFT sft \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset "$BASE/datasets/alpaca_gpt4_data_zh/train.csv#500" \
    --val_dataset "$BASE/datasets/self_cognition/self_cognition.jsonl" \
    --deepspeed zero3_offload \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 5 \
    --save_steps 5 \
    --save_total_limit 1 \
    --logging_steps 2 \
    --max_length 256 \
    --output_dir "$BASE/output/sft_4b_ds_zero3_offload" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
echo Z3O_EXIT=$?
