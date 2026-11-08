#!/bin/bash
# 用途：Gemma4-E2B 双卡 DDP 多模态图文 CPT（mock 图文集；LoRA r8/alpha16；冻结 ViT+aligner）
# 要点：swift pt + use_chat_template true + loss_scale all（全 token 算 loss）
# 注意：不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True——Gemma4 双卡会在加载阶段报 device not ready
# 输出：experiments/output/cpt-gemma4-2gpu
# 运行：cd /root/ms-swift && bash experiments/cpt/cpt_gemma4_2gpu.sh
set -e

export HOME=/root
export CONDA_PREFIX=/opt/miniconda3/envs/ms-swift
export PATH=$CONDA_PREFIX/bin:$PATH
export MODELSCOPE_CACHE=/root/.cache/modelscope

BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/model/gemma-4-E2B-it
SWIFT=$CONDA_PREFIX/bin/swift
DS=$BASE/datasets/mock_vl_cpt
LOG=$BASE/logs/cpt_gemma4_2gpu.log
OUT=$BASE/output/cpt-gemma4-2gpu

mkdir -p "$OUT" "$BASE/logs"

echo "[CPT-Gemma4] model : $MODEL"
echo "[CPT-Gemma4] data  : $DS/cpt_train.jsonl (+ cpt_val.jsonl)"
echo "[CPT-Gemma4] log   : $LOG"
echo "[CPT-Gemma4] out   : $OUT"

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
$SWIFT pt \
  --model "$MODEL" \
  --template gemma4_nothinking \
  --dataset "$DS/cpt_train.jsonl" \
  --val_dataset "$DS/cpt_val.jsonl" \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --max_length 1024 \
  --freeze_vit true \
  --freeze_aligner true \
  --use_chat_template true \
  --loss_scale all \
  --logging_steps 1 \
  --save_steps 200 \
  --eval_steps 100 \
  --save_total_limit 2 \
  --dataloader_num_workers 0 --dataloader_pin_memory false \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"
