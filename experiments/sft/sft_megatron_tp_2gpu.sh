#!/bin/bash
# 两卡【真正张量并行 TP】SFT 脚本（Megatron-LM 后端，参考样例）
# ✅ 已跑通（2026-10-19）：依赖与补丁均已就位（见 ms-swift安装到训练全流程记录.md 第十章）
# 说明：标准 TP——把每一层内部权重（注意力头/FFN 列）切分到多卡，前向/反向 all-reduce 拼合。
# 用法：megatron sft --tensor_model_parallel_size 2（TP 切分大小 = 显卡数）
# 运行：cd /root/ms-swift && bash experiments/sft/sft_megatron_tp_2gpu.sh
set -e
export HOME=/root
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2        # Megatron TP：进程数 = tensor_model_parallel_size（每个进程一张卡，权重在卡间切分）
export TRITON_PTXAS_PATH=/opt/miniconda3/envs/ms-swift/bin/ptxas        # Blackwell(sm_120) 编译 kernel 需要 ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/opt/miniconda3/envs/ms-swift/bin/ptxas

BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B

mkdir -p "$BASE/logs"

/opt/miniconda3/envs/ms-swift/bin/megatron sft \
    --model "$MODEL" \
    --save_safetensors true \
    --dataset "$BASE/datasets/alpaca_gpt4_data_zh/train.csv#500" \
              "$BASE/datasets/self_cognition/self_cognition.jsonl" \
    --tensor_model_parallel_size 2 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --num_train_epochs 1 \
    --output_dir "$BASE/output/sft_megatron_tp" \
    --save_steps 25 \
    --max_length 1024 \
    --dataloader_num_workers 2 \
    --no_save_optim true \
    --no_save_rng true \
    --padding_free false \
    --masked_softmax_fusion false \
    --dataset_num_proc 4
