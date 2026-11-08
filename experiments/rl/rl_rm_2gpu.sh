#!/bin/bash
# 用途：两卡 DDP Reward Model（RM）冒烟（Qwen3-0.6B + orca_dpo_pairs）—— 训一个给「回答」打分的模型，产物是打分器不是对话模型
# 要点：同一份 chosen/rejected 数据；center_rewards_coefficient 让输出奖励均值靠 0，防止整体漂移
# 产出：experiments/output/rl_rm（内含 value head，供 PPO 的 --reward_model 使用）
# 运行：cd /root/ms-swift && bash experiments/rl/rl_rm_2gpu.sh
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B

mkdir -p "$BASE/logs"

$SWIFT rlhf \
    --rlhf_type rm \
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
    --center_rewards_coefficient 0.01 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 1024 \
    --output_dir "$BASE/output/rl_rm" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2
