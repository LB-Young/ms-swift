#!/bin/bash
set -e
export HOME=/root
export CONDA_PREFIX=/opt/miniconda3/envs/ms-swift
export PATH=$CONDA_PREFIX/bin:$PATH

BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/model/gemma-4-E2B-it
CKPT=$(ls -d $BASE/output/cpt-gemma4-1gpu/v*/checkpoint-16 | tail -1) || {
  CKPT=$(ls -d $BASE/output/cpt-gemma4-2gpu/v*/checkpoint-* 2>/dev/null | tail -1); }
LOG=$BASE/logs/cpt_gemma4_infer.log

echo "[infer] adapter=$CKPT"
swift infer \
    --model "$MODEL" \
    --adapters "$CKPT" \
    --template gemma4_nothinking \
    --val_dataset "$BASE/datasets/mock_vl_cpt/cpt_val.jsonl" \
    --max_new_tokens 256 \
    --temperature 0 \
    --max_batch_size 1 \
    --result_path "$BASE/output/cpt-gemma4-1gpu/infer_results.jsonl" \
    2>&1 | tee "$LOG"

echo "EXIT_OK"
