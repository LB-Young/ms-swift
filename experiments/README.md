# ms-swift WSL 两卡训练实验（experiments）

> 2026-10-18 搭建与验证记录。目的：验证 ms-swift 在 WSL2 双卡环境下 SFT、RL（DPO）、RL（GRPO）流程可端到端训练。

## 1. 环境

| 项 | 值 |
|---|---|
| WSL 发行版 | Ubuntu-24.04（WSL2，`D:\WSL\Ubuntu-24.04\ext4.vhdx`） |
| conda | `/opt/miniconda3`（conda 26.7.1） |
| 环境 | `ms-swift`（Python 3.12.15） |
| torch | 2.14.0（cuda134，conda-forge 预编译，支持 sm_120） |
| flash-attn | 2.8.3（pkgs/main 预编译） |
| ms-swift | 4.7.0.dev0（`pip install -e /root/ms-swift`） |
| math_verify | 0.9.0（GRPO accuracy reward 依赖） |
| GPU | 2× RTX 5060 Ti 16GB（sm_120） |
| 模型 | `/mnt/e/models/models/Qwen/Qwen3-0___6B`（本地） |

## 2. 数据集（experiments/datasets/，ModelScope 下载）

| 用途 | 数据 | 说明 |
|---|---|---|
| SFT 训练 | `alpaca_gpt4_data_zh/train.csv` | 中文 alpaca，34 万行，脚本取 #2000 |
| SFT 验证 | `self_cognition/self_cognition.jsonl` | 中文身份认知，108 条 |
| RL 原始 | `zhihu_rlhf_3k/zhihu_3k_rlfh.tsv` | 中文知乎 preference，3772 行 |
| RL 转换 | `zhihu_rlhf_3k/zhihu_3k_rlfh_dpo.jsonl` | tsv→jsonl（messages/response/rejected_response 标准 DPO 格式） |
| RL 数学 | `gsm8k/` | GRPO 数学数据集（modelscope 加载脚本 + ms-swift 云端拉取） |

> ⚠️ zhihu_rlhf_3k 的 chosen/rejected 回答超长，直接用于 DPO 会出现 `logits/chosen=nan` 且 OOM；**改用 ms-swift 注册表官方数据集 `swift/orca_dpo_pairs` 后训练完全正常**。zhihu 数据保留作参考。

## 3. 脚本

- `sft_2gpu.sh`：两卡 DDP SFT（LoRA，Qwen3-0.6B + alpaca-zh）
- `rl_dpo_2gpu.sh`：两卡 DDP DPO（LoRA，`swift/orca_dpo_pairs#500`）
- `rl_grpo_2gpu.sh`：两卡 DDP GRPO（LoRA，gsm8k + accuracy reward，native rollout）
- `sft_tp_2gpu.sh`：两卡**模型并行** SFT（`--device_map auto` 自动切分，单进程多卡）
- `sft_megatron_tp_2gpu.sh`：两卡**真张量并行 TP** SFT（Megatron 后端，**已跑通**，见全流程记录第十章）
- `sft_ds_zero2_2gpu.sh`：两卡 **DeepSpeed ZeRO-2** SFT（`--deepspeed zero2`，LoRA）
- `rl_grpo_ds_2gpu.sh`：两卡 **DeepSpeed ZeRO-2** GRPO
- `sft_ds_autotp_2gpu.sh`：两卡 **DeepSpeed AutoTP 张量并行** SFT（`--deepspeed_autotp_size 2`，full 微调）

多卡写法：`NPROC_PER_NODE=2` + `CUDA_VISIBLE_DEVICES=0,1` 环境变量 + `swift sft / swift rlhf`（DDP）；不设 NPROC_PER_NODE + `--device_map auto`（模型并行）。

## 4. 验证结果

| 流程 | 结果 |
|---|---|
| SFT | ✅ 100/100 步，train_loss 1.80，token_acc 0.58，3m28s；checkpoint-50/100 已存 |
| RL DPO | ✅ 100/100 步，train_loss 0.69→**0.13**，10m20s；checkpoint-50/100 已存 |
| RL GRPO | ✅ 20/20 步，MathAccuracy reward 0→**0.25**，4m39s，显存 2.5GB/卡；checkpoint-10/20 已存 |
| SFT TP（device_map） | ✅ 50/50 步，train_loss 1.84，1m15s；模型切分：layers0-19→卡0、layers20-27→卡1；checkpoint-25/50 已存 |
| SFT Megatron TP | ✅ 38/38 步，train_loss 1.7~2.0，44s，6.31GB/卡，TP:2 PP:1；checkpoint-25/38 已存 |
| SFT DeepSpeed ZeRO-2 | ✅ 100/100 步，train_loss 1.80，2m58s，8.45GB；checkpoint-50/100 已存 |
| GRPO DeepSpeed ZeRO-2 | ✅ 20/20 步，train_loss 0.0012，5m20s，2.53GB；checkpoint-10/20 已存 |
| SFT DeepSpeed AutoTP(2) | ✅ 50/50 步，train_loss 3.72（full 收敛慢正常），1m43s，10.97GB；checkpoint-25/50 已存 |
| SFT 4B 单卡 DS ZeRO-2 (LoRA) | ✅ 10/10 步，train_loss 2.12，8.54GB；checkpoint-10 已存 |
| SFT 4B 双卡 DS ZeRO-2 (LoRA) | ✅ 20/20 步，train_loss 1.725，9.53GB/卡；checkpoint-20 已存（WSL 扩容 24GB 后跑通） |
| 4B ZeRO 矩阵（nods/zero0/1/2/3） | ✅ 5 策略全跑通 10/10 步，loss 2.18~2.22，8.5~8.8GB；纯 DDP 最快 16.9s，zero3 最慢 152s；offload×2 因 CPUAdam 需 nvcc 失败 |

产物目录：`experiments/output/sft/`、`experiments/output/rl_dpo/`（v5 为最终成功版本）、`experiments/output/rl_grpo/`（v2 为最终成功版本）、`experiments/output/sft_tp/`。

## 5. 注意事项（踩坑记录）

1. **HOME 问题**：通过非交互 `wsl -d ... -- bash -c` 调命令时 `HOME` 被透传为 Windows 路径，conda 读不到 `/root/.condarc`。解决：`export HOME=/root`。
2. **后台任务**：`nohup ... &` 在 wsl 命令退出后会被 WSL 发行版自动关闭杀掉。解决：让 wsl 进程常驻（如 PowerShell 后台任务持有）。
3. **tsv 数据集**：ms-swift 本地加载不支持 `.tsv` 扩展名（报 `Couldn't reach 'tsv' on the Hub`）。转 jsonl。
4. **preference 列名**：ms-swift DPO 数据不识别裸 `rejected` 列，标准格式为 `rejected_response`（或 `rejected_messages`）。
5. **DPO 显存**：16GB 卡上 DPO 双序列 + batch2 + max_length1024 会偶发 OOM 警告（训练可继续，PyTorch 自动重分配）；建议 batch=1 或 max_length=512。
6. **GRPO 依赖**：`--deepspeed` 需要 deepspeed 包（未装且可能触发 nvcc 编译）→ 0.6B 小模型去掉即可；GRPO trainer 需要 `msgspec` 包（`pip install msgspec`）。
7. **GRPO 列映射**：`MathAccuracy` reward 强制要求数据的 `solution` 列。gsm8k 原始列是 question/answer，需 `--disable_auto_column_mapping true --columns '{"question": "query", "answer": "solution"}'`。
8. **`swift --help`** 有 KeyError 小 bug，不影响子命令（`swift sft --help` 等正常）。
9. **nvcc 冲突**：本路线全用预编译 wheel（torch/flash-attn/vllm），无需 Linux nvcc，规避了 verl 时代的多卡编译冲突。

## 6. 日常使用

```bash
wsl -d Ubuntu-24.04
conda activate ms-swift
cd /root/ms-swift
bash experiments/sft_2gpu.sh      # SFT
bash experiments/rl_dpo_2gpu.sh   # RL DPO
bash experiments/rl_grpo_2gpu.sh  # RL GRPO
```
