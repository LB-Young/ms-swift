#!/bin/bash
# 4B + LoRA：ZeRO 策略矩阵冒烟（无 DS / zero0 / zero1 / zero2 / zero3 / zero2_offload / zero3_offload）
# 统一配置：LoRA rank8、max_length 256、500 条、10 步、batch1、grad_accum4
# 运行：cd /root/ms-swift && bash experiments/sft/sft_4b_zero_matrix.sh
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export MALLOC_ARENA_MAX=1

SWIFT=/opt/miniconda3/envs/ms-swift/bin/swift
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-4B

run_case() {
  local name="$1"; shift
  local extra=("$@")
  echo "===== RUN $name $(date +%H:%M:%S) ====="
  $SWIFT sft \
    --model "$MODEL" \
    --tuner_type lora \
    --dataset "$BASE/datasets/alpaca_gpt4_data_zh/train.csv#500" \
    --val_dataset "$BASE/datasets/self_cognition/self_cognition.jsonl" \
    "${extra[@]}" \
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
    --save_steps 10 \
    --save_total_limit 1 \
    --logging_steps 2 \
    --max_length 256 \
    --output_dir "$BASE/output/sft_4b_${name}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  echo "===== DONE $name exit=$? ====="
}

echo "########## 4B ZERO MATRIX START $(date) ##########"
run_case nods
run_case ds_zero0 --deepspeed zero0
run_case ds_zero1 --deepspeed zero1
run_case ds_zero2 --deepspeed zero2
run_case ds_zero3 --deepspeed zero3
run_case ds_zero2_offload --deepspeed zero2_offload
run_case ds_zero3_offload --deepspeed zero3_offload
echo "########## 4B ZERO MATRIX END $(date) ##########"
