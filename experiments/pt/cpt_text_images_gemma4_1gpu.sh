#!/bin/bash
# 用途：图文 CPT demo（text/images 结构，无 role）—— Gemma4-E2B-it 单卡
# 数据：{"text": "图片描述", "images": ["/abs/x.png"]}，text 里不带 <image> 占位符
# 要点：--use_chat_template false 让模板不加 role 标记；--loss_scale all 全 token 算 loss
# 运行：cd /root/ms-swift && bash experiments/pt/cpt_text_images_gemma4_1gpu.sh
export HOME=/root
export PATH=/opt/miniconda3/envs/ms-swift/bin:$PATH
BASE=/root/ms-swift/experiments
DS=$BASE/datasets/demo_pt_cpt/structured
OUT=$BASE/output/cpt_text_images_gemma4

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift pt \
  --model /mnt/e/models/model/gemma-4-E2B-it \
  --template gemma4_nothinking \
  --dataset $DS/cpt_text_images_train.jsonl \
  --val_dataset $DS/cpt_text_images_val.jsonl \
  --tuner_type lora --lora_rank 8 --lora_alpha 16 \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --max_length 1024 \
  --freeze_vit true --freeze_aligner true \
  --use_chat_template false --loss_scale all \
  --logging_steps 1 \
  --save_steps 100 --eval_steps 4 \
  --dataloader_num_workers 0 --dataloader_pin_memory false \
  --output_dir $OUT
