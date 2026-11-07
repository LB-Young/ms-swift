#!/bin/bash
# 两卡 DeepSpeed ZeRO-2 GRPO 冒烟脚本（Qwen3-0.6B + gsm8k + accuracy reward）
# 在 rl_grpo_2gpu.sh 基础上新增 --deepspeed zero2（DeepSpeed 管理梯度/优化器分片）
# 运行：cd /root/ms-swift && bash experiments/rl/rl_grpo_ds_2gpu.sh
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
    --deepspeed zero2 \
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
    --max_steps 20 \
    --save_steps 10 \
    --eval_steps 20 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --max_length 1024 \
    --output_dir "$BASE/output/rl_grpo_ds" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
