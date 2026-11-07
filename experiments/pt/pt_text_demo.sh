#!/bin/bash
# 用途：纯文本 PT（预训练）最小可跑 demo —— Qwen3-0.6B + 10 条 mock 语料
# 要点：数据列是 text；use_chat_template false（不套模板）；loss_scale all（每个 token 都算 loss）
# 输出：experiments/output/pt-text-demo
# 运行：cd /root/ms-swift && bash experiments/pt/pt_text_demo.sh
export HOME=/root
export PATH=/opt/miniconda3/envs/ms-swift/bin:$PATH
BASE=/root/ms-swift/experiments
DS=$BASE/datasets/demo_pt_cpt
OUT=$BASE/output/pt-text-demo

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift pt \
  --model /mnt/e/models/models/Qwen/Qwen3-0___6B \
  --dataset $DS/pt_train.jsonl \
  --val_dataset $DS/pt_val.jsonl \
  --tuner_type lora --lora_rank 8 --lora_alpha 16 \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_length 1024 \
  --use_chat_template false --loss_scale all \
  --logging_steps 1 \
  --save_steps 100 --eval_steps 10 \
  --dataloader_num_workers 0 --dataloader_pin_memory false \
  --output_dir $OUT
