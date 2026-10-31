# ms-swift 两卡训练 · 快速上手（只写正确流程，照抄即可跑通）

> 环境：WSL Ubuntu-24.04 + 2×RTX 5060 Ti（sm_120）｜conda env：`ms-swift`｜模型：`/mnt/e/models/models/Qwen/Qwen3-0___6B`
> 本文只列可执行命令与预期结果；踩坑细节见《ms-swift安装到训练全流程记录.md》。

---

## 第 1 步：创建 conda 环境

```bash
conda create -n ms-swift python=3.12 -y
conda activate ms-swift
```

## 第 2 步：安装 pytorch + flash-attn（conda 预编译，不源码编译）

```bash
conda install -n ms-swift -y pytorch=2.14.* flash-attn=2.8.3 \
    --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
```

预期：torch 2.14.0 (cuda134) + flash_attn 2.8.3 + triton 3.8.0。

## 第 3 步：安装 ms-swift（源码可编辑安装）

```bash
cd /root/ms-swift
pip install -e .
```

预期：装出 ms_swift 4.7.0.dev0 及 transformers/peft/trl/accelerate 全套依赖。

## 第 4 步：验证 GPU

```bash
python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

预期：`2.14.0 True 2` + `NVIDIA GeForce RTX 5060 Ti (12, 0)`。

## 第 5 步：下载数据集（ModelScope，一次性）

```bash
modelscope download --repo-type dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --local-dir /root/ms-swift/experiments/datasets/alpaca_gpt4_data_zh
modelscope download --repo-type dataset swift/self-cognition \
    --local-dir /root/ms-swift/experiments/datasets/self_cognition
modelscope download --repo-type dataset AI-ModelScope/zhihu_rlhf_3k \
    --local-dir /root/ms-swift/experiments/datasets/zhihu_rlhf_3k
modelscope download --repo-type dataset modelscope/gsm8k \
    --local-dir /root/ms-swift/experiments/datasets/gsm8k
```

（模型已在本机 `/mnt/e/models/models/Qwen/Qwen3-0___6B`，无需下载。）

---

## 第 6 步：跑两卡 SFT（DDP）

```bash
cd /root/ms-swift
bash experiments/sft_2gpu.sh
```

- 日志：`experiments/logs/sft.log`　产物：`experiments/output/sft/`
- 预期：100/100 步，train_loss ~1.80，约 3m28s，checkpoint-50/100 已存

## 第 7 步：跑两卡 DPO（RL）

```bash
bash experiments/rl_dpo_2gpu.sh
```

- 日志：`experiments/logs/rl_dpo.log`　产物：`experiments/output/rl_dpo/`
- 预期：100/100 步，train_loss 0.69→0.13，约 10m20s，checkpoint-50/100 已存

## 第 8 步：跑两卡 GRPO（RL）

```bash
bash experiments/rl_grpo_2gpu.sh
```

- 日志：`experiments/logs/rl_grpo.log`　产物：`experiments/output/rl_grpo/`
- 预期：20/20 步，MathAccuracy reward 0→0.25，约 4m39s，checkpoint-10/20 已存

## 第 9 步：跑两卡模型并行 SFT（device_map，单进程多卡）

```bash
bash experiments/sft_tp_2gpu.sh
```

- 日志：`experiments/logs/sft_tp.log`　产物：`experiments/output/sft_tp/`
- 预期：50/50 步，train_loss 1.84，1m15s；模型切分 layers0-19→卡0、layers20-27→卡1

## 第 10 步：跑两卡真张量并行 SFT（Megatron TP）

```bash
# ① 装依赖（预编译，无源码编译）
pip install 'megatron-core>=0.16' 'mcore-bridge>=1.6.3'

# ② Blackwell 编译 kernel 需要 cuda 头与 ptxas
conda install -n ms-swift -y -c nvidia cuda-cudart-dev cuda-nvcc
ln -sf /opt/miniconda3/envs/ms-swift/include/cuda.h /opt/miniconda3/envs/ms-swift/targets/x86_64-linux/include/cuda.h

# ③ 跑训练
bash experiments/sft_megatron_tp_2gpu.sh
```

- 日志：`experiments/logs/sft_megatron_tp.log`　产物：`experiments/output/sft_megatron_tp/`
- 预期：38/38 步，train_loss 1.7~2.0，44s，6.31 GiB/卡；`TP: 2, PP: 1`；checkpoint-25/38 已存
- 说明：ms-swift 的 Megatron 后端依赖 transformer-engine（源码编译）与若干 TE 特性。本环境已按《全流程记录》10.6 卸载 TE 走 `transformer_impl='local'` 并打完 13 处补丁；**若重建环境，需按 10.6 重新打补丁**（脚本本身无需改动）。

---

## 第 11 步：跑 DeepSpeed（ZeRO-2 / AutoTP，可选）

```bash
# ① 安装（纯 Python wheel，ZeRO 核心不依赖 nvcc 编译）
pip install deepspeed

# ② ZeRO-2（优化器状态/梯度分片，LoRA 或 full 皆可）
bash experiments/sft_ds_zero2_2gpu.sh        # SFT + zero2：100 步，loss 1.80，2m58s，8.45GB
bash experiments/rl_grpo_ds_2gpu.sh          # GRPO + zero2：20 步，5m20s，2.53GB

# ③ AutoTP 张量并行（deepspeed 自动切分层内权重，仅 full 微调）
bash experiments/sft_ds_autotp_2gpu.sh       # SFT + AutoTP(2卡)：50 步，1m43s，10.97GB
```

- 日志：`experiments/logs/sft_ds_zero2.log` / `rl_grpo_ds.log` / `sft_ds_autotp.log`
- 产物：`experiments/output/sft_ds_zero2/` / `rl_grpo_ds/` / `sft_ds_autotp/`
- 说明：`--deepspeed` 可写预设名 `zero0/zero1/zero2/zero3/zero2_offload/zero3_offload` 或自定义 json 路径；AutoTP 需同时加 `--deepspeed_autotp_size 2 --tuner_type full`。deepspeed 与 device_map 不兼容。

## 第 12 步：更大模型 Qwen3-4B（可选，实测结论）

```bash
# ① 下载（E 盘 GGUF 大模型不能训练，需 HF safetensors 版）
/opt/miniconda3/envs/ms-swift/bin/modelscope download --model Qwen/Qwen3-4B --local_dir /mnt/e/models/models/Qwen/Qwen3-4B

# ② 双卡 4B + LoRA：当前 WSL 内存（15GB=物理 50%）不够，会被内核 OOM 杀进程（见全流程记录 12.3）
# ③ 单卡 4B + LoRA + ZeRO-2：可跑（先装 mpi4py：conda install -p /opt/miniconda3/envs/ms-swift -c conda-forge mpi4py）
bash experiments/sft_ds_zero2_qwen4b_lora_single.sh   # 10/10 步，train_loss 2.12，8.5GB 显存

# ④ 双卡 4B：已给 WSL 扩容（.wslconfig memory=24GB，已重启生效），直接跑双卡脚本：
bash experiments/sft_ds_zero2_qwen4b_lora_v2.sh   # 双卡 20/20 步，train_loss 1.725，9.53GB/卡
```

## 第 13 步：4B ZeRO 策略对比（实测结论，重要）

```bash
# 一键跑全矩阵（7 个策略，~15 分钟）
bash experiments/sft_4b_zero_matrix.sh        # 日志 logs/sft_4b_zero_matrix.log
```

结论（4B + LoRA 双卡，10 步实测）：
- **最快 = 纯 DDP**（16.9s），zero0/1/2 约 31~47s，**zero3 最慢 152s**（allgather 通信开销）
- 显存全部持平 8.5~8.8GB（LoRA 优化器状态太小，ZeRO 分片无收益）→ **LoRA 直接用 DDP，别上 deepspeed**
- **zero2_offload / zero3_offload 不可用**：CPU 优化器 DeepSpeedCPUAdam 需要 nvcc 编译，本环境无 CUDA_HOME（MissingCUDAException）

## Windows PowerShell 后台启动（推荐，防 WSL 杀进程）

```powershell
wsl -d Ubuntu-24.04 -- bash -c "export HOME=/root; cd /root/ms-swift && bash experiments/sft_2gpu.sh > experiments/logs/sft.log 2>&1"
```

- 查看进度：`wsl -d Ubuntu-24.04 -- bash -c "tail -f /root/ms-swift/experiments/logs/sft.log"`
- 换脚本时改 `sft_2gpu.sh` 与日志名即可（sft / rl_dpo / rl_grpo / sft_tp / sft_megatron_tp）。

## 关键环境变量（所有脚本内已内置，无需手动设置）

| 变量 | 值 | 作用 |
|---|---|---|
| `NPROC_PER_NODE` | 2（DDP/DPO/GRPO/Megatron）；不设（device_map） | 进程数 |
| `CUDA_VISIBLE_DEVICES` | 0,1 | 使用哪两张卡 |
| `TRITON_PTXAS_PATH` / `TRITON_PTXAS_BLACKWELL_PATH` | env/bin/ptxas | Blackwell 编译 kernel（Megatron 脚本内） |
