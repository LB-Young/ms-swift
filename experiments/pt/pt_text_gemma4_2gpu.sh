#!/bin/bash
# 用途：纯文本 PT demo —— Gemma4-E2B-it 双卡 DDP，数据每行只有 {"text": "..."}
# 要点：和图文版的唯一差别是数据没有 images 字段；参数完全一致（无模板标记 + 全 token loss）
# 运行：cd /root/ms-swift && bash experiments/pt/pt_text_gemma4_2gpu.sh
export HOME=/root
export PATH=/opt/miniconda3/envs/ms-swift/bin:$PATH
BASE=/root/ms-swift/experiments
DS=$BASE/datasets/demo_pt_cpt/structured
OUT=$BASE/output/pt_text_gemma4

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift pt \
  --model /mnt/e/models/model/gemma-4-E2B-it \
  --template gemma4_nothinking \
  --dataset $DS/pt_text_train.jsonl \
  --val_dataset $DS/pt_text_val.jsonl \
  --tuner_type lora --lora_rank 8 --lora_alpha 16 \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_length 1024 \
  --use_chat_template false --loss_scale all \
  --logging_steps 1 \
  --save_steps 100 --eval_steps 5 \
  --dataloader_num_workers 0 --dataloader_pin_memory false \
  --output_dir $OUT
