#!/bin/bash
# 用途：用 Qwen2-VL-2B CPT 产出的 LoRA adapter 对验证集推理（swift infer，自动取 checkpoint-16）
# 输出：experiments/output/cpt-vl-2gpu/infer_results.jsonl；日志 experiments/logs/cpt_vl_infer.log
# 运行：cd /root/ms-swift && bash experiments/cpt/cpt_vl_infer.sh
set -e
export HOME=/root
export CONDA_PREFIX=/opt/miniconda3/envs/ms-swift
export PATH=$CONDA_PREFIX/bin:$PATH

BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/qwen/Qwen2-VL-2B-Instruct
CKPT=$(ls -d $BASE/output/cpt-vl-2gpu/v1-*/checkpoint-16 | tail -1)
LOG=$BASE/logs/cpt_vl_infer.log

echo "[infer] adapter=$CKPT"
swift infer \
    --model "$MODEL" \
    --adapters "$CKPT" \
    --val_dataset "$BASE/datasets/mock_vl_cpt/cpt_val.jsonl" \
    --max_new_tokens 256 \
    --temperature 0 \
    --max_batch_size 2 \
    --result_path "$BASE/output/cpt-vl-2gpu/infer_results.jsonl" \
    2>&1 | tee "$LOG"

echo "EXIT_OK"
