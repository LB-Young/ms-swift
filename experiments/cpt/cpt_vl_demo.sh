#!/bin/bash
# 用途：多模态图文 CPT（继续预训练）最小可跑 demo —— Gemma4-E2B 双卡 DDP + 8 条 mock 图文
# 要点：数据是 messages + images，content 里用标准 <image> 占位；swift pt + 套模板 + 全 token loss
# 注意：不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True——Gemma4 双卡会在加载阶段报 device not ready
# 输出：experiments/output/cpt-vl-demo
# 运行：cd /root/ms-swift && bash experiments/cpt/cpt_vl_demo.sh
export HOME=/root
export PATH=/opt/miniconda3/envs/ms-swift/bin:$PATH
BASE=/root/ms-swift/experiments
DS=$BASE/datasets/demo_pt_cpt
OUT=$BASE/output/cpt-vl-demo

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift pt \
  --model /mnt/e/models/model/gemma-4-E2B-it \
  --template gemma4_nothinking \
  --dataset $DS/cpt_train.jsonl \
  --val_dataset $DS/cpt_val.jsonl \
  --tuner_type lora --lora_rank 8 --lora_alpha 16 \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_length 1024 \
  --freeze_vit true --freeze_aligner true \
  --use_chat_template true --loss_scale all \
  --logging_steps 1 \
  --save_steps 100 --eval_steps 4 \
  --dataloader_num_workers 0 --dataloader_pin_memory false \
  --output_dir $OUT
