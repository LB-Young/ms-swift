#!/bin/bash
# CPT (Continued Pre-Training) on a multimodal (image-text) dataset with ms-swift.
# ms-swift has no `cpt` subcommand; CPT == `swift pt` (SwiftPretrain inherits SwiftSft,
# differing only in use_chat_template=False + loss_scale='all' -> every token contributes to loss).
# 2x RTX 5060 Ti (sm_120), LoRA, Qwen2-VL-2B-Instruct.
set -e

export HOME=/root
export CONDA_PREFIX=/opt/miniconda3/envs/ms-swift
export PATH=$CONDA_PREFIX/bin:$PATH
export MODELSCOPE_CACHE=/root/.cache/modelscope

BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/qwen/Qwen2-VL-2B-Instruct
SWIFT=$CONDA_PREFIX/bin/swift
DS=$BASE/datasets/mock_vl_cpt
LOG=$BASE/logs/cpt_vl_2gpu.log
OUT=$BASE/output/cpt-vl-2gpu

mkdir -p "$OUT" "$BASE/logs"

echo "[CPT-VL] model : $MODEL"
echo "[CPT-VL] data  : $DS/cpt_train.jsonl (+ $DS/cpt_val.jsonl)"
echo "[CPT-VL] log   : $LOG"
echo "[CPT-VL] out   : $OUT"

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$SWIFT pt \
  --model "$MODEL" \
  --dataset "$DS/cpt_train.jsonl" \
  --val_dataset "$DS/cpt_val.jsonl" \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_length 2048 \
  --freeze_vit true \
  --freeze_aligner true \
  --use_chat_template false \
  --loss_scale all \
  --logging_steps 1 \
  --save_steps 200 \
  --eval_steps 100 \
  --save_total_limit 2 \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"
