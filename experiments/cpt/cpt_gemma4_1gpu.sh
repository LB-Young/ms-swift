#!/bin/bash
# 用途：Gemma4-E2B 单卡多模态图文 CPT（0 号卡；mock 图文集；LoRA r8/alpha16；冻结 ViT+aligner）
# 要点：swift pt + use_chat_template true + loss_scale all（全 token 算 loss）
# 输出：experiments/output/cpt-gemma4-1gpu
# 运行：cd /root/ms-swift && bash experiments/cpt/cpt_gemma4_1gpu.sh
export HOME=/root
export PATH=/opt/miniconda3/envs/ms-swift/bin:$PATH
BASE=/root/ms-swift/experiments
DS=$BASE/datasets/mock_vl_cpt
OUT=$BASE/output/cpt-gemma4-1gpu

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift pt \
  --model /mnt/e/models/model/gemma-4-E2B-it \
  --template gemma4_nothinking \
  --dataset $DS/cpt_train.jsonl \
  --val_dataset $DS/cpt_val.jsonl \
  --tuner_type lora --lora_rank 8 --lora_alpha 16 \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_length 1024 \
  --freeze_vit true --freeze_aligner true \
  --use_chat_template true --loss_scale all \
  --logging_steps 1 \
  --save_steps 200 --eval_steps 100 \
  --dataloader_num_workers 0 --dataloader_pin_memory false \
  --output_dir $OUT
