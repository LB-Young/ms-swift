#!/bin/bash
# 用途：纯文本 PT demo —— Qwen3.5-0.8B-Instruct 单卡（比 Qwen3-0.6B 更新的一版小模型）
# 数据：每行只有 {"text": "..."}
# 要点：Qwen3.5 的 model_type / template 自动匹配都有歧义（qwen3_5 vs ovis_ocr2 / qwen3_8），两者都必须显式指定
#      不加 role 标记、全 token 算 loss
# 运行：cd /root/ms-swift && bash experiments/pt/pt_text_qwen35_1gpu.sh
export HOME=/root
export PATH=/opt/miniconda3/envs/ms-swift/bin:$PATH
BASE=/root/ms-swift/experiments
DS=$BASE/datasets/demo_pt_cpt/structured
OUT=$BASE/output/pt_text_qwen35

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift pt \
  --model /mnt/e/models/qwen/Qwen3.5-0.8B-Instruct \
  --model_type qwen3_5 \
  --template qwen3_5 \
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
