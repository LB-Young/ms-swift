# ms-swift WSL 两卡训练：从环境搭建到 SFT / DPO / GRPO 全部跑通（完整记录）

> 📁 目录说明（2026-11-07）：本文档已移至 `experiments/docs/`。文中提到的 `experiments/*.sh`、`experiments/*.py` 现已归类到 `experiments/sft|rl|cpt|infer/` 子目录，路径映射见 `experiments/README.md`。

> 日期：2026-10-18
> 机器：2× NVIDIA GeForce RTX 5060 Ti 16GB（Blackwell，sm_120）+ WSL2（Ubuntu-24.04）
> 目的：在 WSL 中为 ms-swift 搭建 conda 环境，从 ModelScope 下载数据集，编写并跑通 **SFT、DPO（RL）、GRPO（RL）** 三个两卡训练脚本。
> 本文记录每一步的命令、验证结果，以及踩过的每一个坑和最终解决方案，可按此文档完整复现。

---

## 一、总体结论（先看这个）

- 三个流程**全部跑通**：SFT ✅、RL/DPO ✅、RL/GRPO ✅，且全程**双卡 DDP 正常，没有任何 nvcc/底层编译冲突**。
- 核心思路：**全走预编译 wheel**（torch / flash-attn / ms-swift 依赖），不进行任何源码编译 → 不需要 Linux 版 nvcc → 规避了此前 verl 时代的多卡编译冲突问题。
- 硬件上 0.6B 模型非常轻松：SFT 峰值 8.4GB/卡，DPO 峰值 14.9GB/卡（偶发 OOM 警告），GRPO 峰值仅 2.5GB/卡。

| 流程 | 脚本 | 数据集 | 结果 |
|---|---|---|---|
| SFT | `sft_2gpu.sh` | alpaca-gpt4-data-zh#2000 + self-cognition(eval) | 100/100 步，train_loss 1.80，3m28s |
| RL/DPO | `rl_dpo_2gpu.sh` | swift/orca_dpo_pairs#500 | 100/100 步，train_loss 0.69→0.13，10m20s |
| RL/GRPO | `rl_grpo_2gpu.sh` | modelscope/gsm8k#200 | 20/20 步，MathAccuracy reward 0→0.25，4m39s |

---

## 二、环境搭建（从零开始）

### 2.1 前置：WSL 与网络

- WSL 发行版：`Ubuntu-24.04`（WSL2），虚拟磁盘在 `D:\WSL\Ubuntu-24.04\ext4.vhdx`。
- 网络：`C:\Users\LBY\.wslconfig` 配置 `networkingMode=mirrored`，可直连清华镜像 / 走 `127.0.0.1:7897` 代理。
- 显卡穿透：`nvidia-smi` 在 WSL 内正常显示 2 张 5060 Ti，驱动 610.57.01 / CUDA UMD 13.3。

### 2.2 conda（/opt/miniconda3）

本机 conda 位于 `/opt/miniconda3`（conda 26.7.1）。**若需要重装**：

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
```

### 2.3 配置清华镜像（关键！）

**坑 1：conda 访问官方源 repo.anaconda.com 报 SSL 错误 / 超时**

现象：`CondaSSLError ... SSLEOFError ... Max retries exceeded with url: /pkgs/main/linux-64/repodata.json.zst`。

排查过程：写好了 `/root/.condarc` 依然失败 → 用 `conda config --show-sources` 诊断 → 发现 **`HOME` 被透传成了 Windows 路径（`C:UsersLBY`）**，conda 根本读不到 `/root/.condarc`！

根因：通过 `wsl -d Ubuntu-24.04 -- bash -c "..."`（非交互调用）时，Windows 的 `HOME` 环境变量被继承，覆盖了 `/root`。

解决方案（两条都做）：
1. 每次非交互调用前显式 `export HOME=/root`；
2. 配置 `/root/.condarc`（清华镜像 + conda-forge）：

```yaml
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
channel_priority: flexible
```

同时配置 pip 清华源 `/root/.pip/pip.conf`：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

若 `.condarc` 尚未生效，可临时用 `--override-channels` 强制指定镜像通道（本次环境创建即用了此招兜底）。

### 2.4 创建 ms-swift conda 环境

```bash
conda create -n ms-swift python=3.12 -y
```

### 2.5 安装 pytorch + flash-attn（预编译路线，这是本方案的核心）

**为什么用 conda 预编译包而不是 pip？**
- RTX 5060 Ti 是 Blackwell（sm_120），**必须 CUDA ≥ 12.8 的构建才能识别**；
- 源码编译 flash-attn 需要 nvcc + 对 Blackwell 支持差的工具链，极易踩坑（此前 verl 就是死在这里）；
- conda-forge 的 `pytorch`（cuda134 构建）与 Anaconda pkgs/main 的 `flash-attn` 预编译包 ABI 匹配，三者由 conda 求解器统一解决。

```bash
conda install -n ms-swift -y pytorch=2.14.* flash-attn=2.8.3 \
    --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
# 下载约 3GB，装好后：torch 2.14.0 (cuda134) + flash_attn 2.8.3 + triton 3.8.0
```

**坑 2：torch 报了 `cuda.h: No such file or directory`（本机此前遇到过）**
- 原因：conda 环境只有 CUDA 运行时库，没有开发头文件。
- 解决：`conda install -n ms-swift -c conda-forge cuda-cudart-dev`，并把 `export CPATH=/opt/miniconda3/envs/ms-swift/include` 写入 `~/.bashrc`。
- 本次全程预编译未再触发；仅记录备用。

### 2.6 安装 ms-swift（源码可编辑安装）

```bash
cd /root/ms-swift
pip install -e .
# 装出 ms_swift 4.7.0.dev0，以及 transformers 5.19.0 / peft 0.20.0 / trl 0.29.1 / accelerate 1.15.0 等
```

**坑 3：`swift --help` 报 KeyError**
- 现象：`KeyError: '--help'`（cli/main.py 的 route_mapping 没有处理 `--help`）。
- 结论：不影响子命令，`swift sft --help` / `swift rlhf --help` / `swift infer --help` 均正常。忽略即可。

### 2.7 GPU 环境验证

```python
import torch, flash_attn
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
```
预期输出：`2.14.0 True 2` + `NVIDIA GeForce RTX 5060 Ti (12, 0)`（sm_120）✅

---

## 三、数据集准备（ModelScope 下载）

### 3.1 下载命令（modelscope CLI）

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

### 3.2 下载结果与格式

| 数据集 | 实际文件 | 格式 | 用途 |
|---|---|---|---|
| alpaca-gpt4-data-zh | `train.csv`（34 万行）+ 元数据 | instruction,input,output | SFT 训练（#2000） |
| self-cognition | `self_cognition.jsonl`（108 条） | query,response,tag | SFT 验证集 |
| zhihu_rlhf_3k | `zhihu_3k_rlfh.tsv`（3772 行） | prompt,chosen,rejected | RL 原始数据 |
| gsm8k | `gsm8k.py` + dataset_infos | question,answer（云端拉取） | GRPO 数学题 |

**坑 4：`$MS` 变量被 PowerShell 吃掉**
- 现象：`MS=/path/to/modelscope; $MS download ...` 报 `download: command not found`。
- 原因：PowerShell 双引号内 `$MS` 被 PowerShell 自身展开为空。
- 解决：写完整路径，或转义 `\$MS`。

**坑 5：tsv 转 jsonl**
- 现象：DPO 训练 `--dataset xxx.tsv` 报 `Couldn't reach 'tsv' on the Hub`。
- 原因：ms-swift 本地加载不支持 `.tsv` 扩展名（会把它当 Hub 数据集名）。
- 解决：用 Python 转成 jsonl（3460 行有效数据）。

```python
import csv, json
src = '.../zhihu_3k_rlfh.tsv'; dst = '.../zhihu_3k_rlfh.jsonl'
with open(src, encoding='utf-8') as f, open(dst, 'w', encoding='utf-8') as g:
    for row in csv.DictReader(f, delimiter='\t'):
        g.write(json.dumps({'prompt': row['prompt'], 'chosen': row['chosen'],
                            'rejected': row['rejected']}, ensure_ascii=False) + '\n')
```

---

## 四、SFT 脚本：`sft_2gpu.sh`

### 4.1 脚本内容要点

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
swift sft \
    --model /mnt/e/models/models/Qwen/Qwen3-0___6B \
    --tuner_type lora \
    --dataset '.../alpaca_gpt4_data_zh/train.csv#2000' \
    --val_dataset '.../self_cognition/self_cognition.jsonl' \
    --torch_dtype bfloat16 --max_steps 100 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
    --max_length 1024 --learning_rate 1e-4 \
    --output_dir '.../output/sft'
```

> 多卡核心：`NPROC_PER_NODE` 与 `CUDA_VISIBLE_DEVICES` 两个环境变量，ms-swift 自动拉起 torchrun 多进程 DDP。

### 4.2 运行方式（启动命令）

```bash
# 方式 A（最简单）：WSL 交互式会话内直接运行
conda activate ms-swift
cd /root/ms-swift
bash experiments/sft_2gpu.sh

# 方式 B（推荐，可稳定后台跑）：Windows PowerShell 后台任务持有 wsl 进程
wsl -d Ubuntu-24.04 -- bash -c "export HOME=/root; cd /root/ms-swift && \
    bash experiments/sft_2gpu.sh > experiments/logs/sft.log 2>&1"

# 查看进度：
wsl -d Ubuntu-24.04 -- bash -c "tail -f /root/ms-swift/experiments/logs/sft.log"

# 方式 C（不推荐）：nohup
# nohup bash experiments/sft_2gpu.sh > logs/sft.log 2>&1 &
```

**坑 6：`nohup ... &` 后台进程被杀**
- 现象：wsl 命令一退出，训练进程就没了，日志文件都消失。
- 根因：WSL 发行版在最后一个进程退出后会自动关闭，nohup 的进程随之被杀。
- 解决：**让 wsl 进程本身保持运行**——用 PowerShell 后台任务（run_in_background）持有整个 `wsl ... bash -c "训练脚本前台运行"` 命令；或保持一个交互式 wsl 会话打开。

**坑 7：日志目录未建导致重定向失败**
- 现象：`> experiments/logs/sft.log` 报 No such file。
- 原因：脚本内部的 `mkdir -p logs` 晚于 shell 重定向执行。
- 解决：运行前先 `mkdir -p experiments/logs`。

### 4.3 验证结果

```
Train: 100%|██████████| 100/100 [03:27]
{'loss': '1.775', 'token_acc': '0.5775', 'memory(GiB)': '8.39', ...}
{'eval_loss': '2.288', 'eval_token_acc': '0.5553'}
[INFO:swift] last_model_checkpoint: .../output/sft/v0-20261018-210950/checkpoint-100
```
- 双卡 DDP 正常（rank0/rank1 日志齐全），loss 1.8 附近波动，eval 27 条完成，checkpoint 已保存。✅

---

## 五、RL/DPO 脚本：`rl_dpo_2gpu.sh`

### 5.1 脚本内容要点

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
swift rlhf \
    --rlhf_type dpo \
    --model /mnt/e/models/models/Qwen/Qwen3-0___6B \
    --tuner_type lora \
    --dataset 'swift/orca_dpo_pairs#500' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 --max_steps 100 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 --max_length 1024 --learning_rate 1e-5 \
    --output_dir '.../output/rl_dpo'
```

### 5.2 运行方式（启动命令）

```bash
# WSL 交互式会话内（最简单）：
conda activate ms-swift
cd /root/ms-swift
bash experiments/rl_dpo_2gpu.sh

# Windows PowerShell 后台运行（推荐，防 WSL 发行版自动关闭杀进程）：
#   wsl -d Ubuntu-24.04 -- bash -c "export HOME=/root; cd /root/ms-swift && bash experiments/rl_dpo_2gpu.sh > experiments/logs/rl_dpo.log 2>&1"
# 查看进度：
#   wsl -d Ubuntu-24.04 -- bash -c "tail -f /root/ms-swift/experiments/logs/rl_dpo.log"
```

### 5.3 踩坑全记录（DPO 是本次最曲折的一环）

**坑 8：DPO 数据不识别裸 `rejected` 列**
- 现象：`ValueError: inputs.rejected is None`。
- 根因：查 `swift/template/base.py` 的 `from_dict`：ms-swift 只认 `rejected_` 前缀的字段（`rejected_response` / `rejected_messages`），**不认裸 `rejected` 列**。
- 解决：转换为标准格式——`messages + response + rejected_response`：

```json
{"messages": [{"role": "user", "content": "问题"}],
 "response": "chosen 回答",
 "rejected_response": "rejected 回答"}
```

**坑 9：zhihu_rlhf_3k 导致 chosen 分支 nan + OOM**
- 现象：训练能跑，但 `logits/chosen: nan`、`logps/chosen: 0`、`rewards/chosen: 0`，且显存频繁 OOM（free 一度为 0）。
- 根因：zhihu 数据 chosen/rejected 都是**超长知乎回答**（数千字），DPO 双序列 + 截断策略下 chosen 分支异常；batch 降到 1、max_length 降到 512 依然 nan → 数据本身问题而非显存。
- 解决：**换 ms-swift 注册表官方 DPO 数据集 `swift/orca_dpo_pairs#500`**（短文本、标准格式），一次通过，loss 正常下降。zhihu 数据保留在本地作参考。

**坑 10：DPO 双序列显存吃紧**
- 现象：orca 数据下训练正常，但 batch=2 + max_length=1024 仍偶发 `CUDACachingAllocator OOM` 警告（PyTorch 自动重分配，训练不中断）。
- 建议：正式跑长文本 DPO 时 `per_device_train_batch_size=1` 或 `max_length=512`。

### 5.4 验证结果

```
Train: 100%|██████████| 100/100 [10:20]
{'train_loss': '0.1327', 'train_runtime': '620.3', 'memory(GiB)': '14.89'}
[INFO:swift] last_model_checkpoint: .../output/rl_dpo/v5-20261018-212521/checkpoint-100
```
- loss 从 0.6931 一路降到 **0.1327**（DPO 有效学习），checkpoint 已保存。✅

---

## 六、RL/GRPO 脚本：`rl_grpo_2gpu.sh`

### 6.1 脚本内容要点

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/e/models/models/Qwen/Qwen3-0___6B \
    --tuner_type lora \
    --use_vllm false \
    --dataset 'modelscope/gsm8k#200' \
    --disable_auto_column_mapping true \
    --columns '{"question": "query", "answer": "solution"}' \
    --reward_funcs accuracy \
    --enable_thinking false \
    --num_generations 2 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 --max_length 1024 --learning_rate 5e-5 \
    --output_dir '.../output/rl_grpo'
```

### 6.2 运行方式（启动命令）

```bash
# WSL 交互式会话内（最简单）：
conda activate ms-swift
cd /root/ms-swift
bash experiments/rl_grpo_2gpu.sh

# Windows PowerShell 后台运行（推荐，防 WSL 发行版自动关闭杀进程）：
#   wsl -d Ubuntu-24.04 -- bash -c "export HOME=/root; cd /root/ms-swift && bash experiments/rl_grpo_2gpu.sh > experiments/logs/rl_grpo.log 2>&1"
# 查看进度：
#   wsl -d Ubuntu-24.04 -- bash -c "tail -f /root/ms-swift/experiments/logs/rl_grpo.log"
```

### 6.3 踩坑全记录

**坑 11：`--deepspeed zero2` 报缺 deepspeed 包**
- 现象：`PackageNotFoundError: No package metadata was found for deepspeed`。
- 说明：ms-swift 基础安装不带 deepspeed；且 deepspeed 的 CUDA 算子可能触发 nvcc 编译（正是想规避的）。
- 解决：0.6B 小模型默认 DDP 足够，**去掉 `--deepspeed`**。

**坑 12：GRPO trainer 缺 msgspec**
- 现象：`ModuleNotFoundError: No module named 'msgspec'`（trainer_factory 加载 GRPO trainer 时）。
- 解决：`pip install msgspec`。

**坑 13：`MathAccuracy` reward 报缺 `solution` 参数**
- 现象：`TypeError: MathAccuracy.__call__() missing 1 required positional argument: 'solution'`。
- 根因：accuracy reward 强制要求数据集带 `solution` 列（最终答案）；gsm8k 原始列是 `question/answer`，自动映射把 answer 变成 response，导致没有 solution。
- 解决：**关闭自动列映射 + 手动映射**：
  ```bash
  --disable_auto_column_mapping true \
  --columns '{"question": "query", "answer": "solution"}'
  ```
  `question→query`（模板输入），`answer→solution`（reward 输入）。

**坑 14（可选）：vllm 后端**
- 当前脚本用 `--use_vllm false`（native 纯 torch rollout），无需装 vllm。
- 若要 vllm 加速：`pip install vllm`（0.8.3+ 支持 sm_120，预编译 wheel 不会引入 nvcc 问题），脚本改为 `--use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.5`。

### 6.4 验证结果

```
Train: 100%|██████████| 20/20 [04:39]
{'loss': '0.0005263', 'reward': '0.25', 'rewards/MathAccuracy/mean': '0.25',
 'completions/mean_length': '233.6', 'memory(GiB)': '2.48'}
[INFO:swift] last_model_checkpoint: .../output/rl_grpo/v2-20261018-222340/checkpoint-20
```
- MathAccuracy reward 从 0 提升到 **0.25**（模型学会答对部分 gsm8k 题），rollout→reward→policy update 全链路正常，显存仅 2.5GB/卡。✅

---

## 七、踩坑总表（速查）

| # | 现象 | 根因 | 解决方案 |
|---|---|---|---|
| 1 | conda SSL 错误访问官方源 | HOME 被透传为 Windows 路径，.condarc 失效 | `export HOME=/root` + 清华镜像 .condarc |
| 2 | cuda.h not found | 缺开发头文件 | conda 装 `cuda-cudart-dev` + CPATH |
| 3 | swift --help KeyError | CLI 路由 bug | 忽略，子命令正常 |
| 4 | $MS download not found | PowerShell 展开变量 | 写完整路径/转义 |
| 5 | tsv 无法加载 | ms-swift 不支持 .tsv | 转 jsonl |
| 6 | nohup 进程被杀 | WSL 发行版自动关闭 | wsl 进程常驻（PowerShell 后台任务） |
| 7 | 日志重定向失败 | logs 目录未先建 | 先 mkdir -p logs |
| 8 | inputs.rejected is None | 不识别裸 rejected 列 | 转 messages/response/rejected_response |
| 9 | chosen nan + OOM | zhihu 长文本数据问题 | 换 orca_dpo_pairs |
| 10 | DPO 偶发 OOM | 双序列显存吃紧 | batch=1 或 max_length=512 |
| 11 | 缺 deepspeed | 未安装 | 去掉 --deepspeed |
| 12 | 缺 msgspec | GRPO 依赖 | pip install msgspec |
| 13 | MathAccuracy 缺 solution | gsm8k 无 solution 列 | --columns 映射 answer→solution |
| 14 | vllm（可选） | 需加速 rollout | pip install vllm + --use_vllm true |
| 15 | megatron --help traceback | megatron-core 未装 | pip install megatron-core（TP 真版前置） |
| 16 | 小模型 TP 收益为负 | TP 通信开销 > 并行收益 | 大模型才用 TP，小模型优先 DDP |

---

## 八、日常使用

```bash
wsl -d Ubuntu-24.04
conda activate ms-swift
cd /root/ms-swift
bash experiments/sft_2gpu.sh          # SFT（DDP）
bash experiments/rl_dpo_2gpu.sh       # RL DPO（DDP）
bash experiments/rl_grpo_2gpu.sh      # RL GRPO（DDP）
bash experiments/sft_tp_2gpu.sh       # SFT（device_map 模型并行）
bash experiments/sft_megatron_tp_2gpu.sh  # SFT（Megatron 真 TP，需先装 megatron-core）
```

产物：`experiments/output/{sft,rl_dpo,rl_grpo,sft_tp}/`（checkpoint 的 LoRA adapter）。
日志：`experiments/logs/*.log`。

## 九、后续建议

1. 正式训练用更大模型（7B/8B）时：SFT 直接可用；DPO 建议 batch=1；GRPO 建议装 vllm 加速 rollout。
2. 需要更复杂 reward 时，ms-swift 支持 `--reward_funcs` 组合（accuracy/format/code 等）及外部 reward 插件。
3. 清理调参中间产物：可删 `output/rl_dpo/v0~v4`、`output/rl_grpo/v0~v1`（各保留最终成功版本）。

---

## 十、TP（张量并行）脚本：构建过程与验证（2026-10-19 追加）

在 DDP 全部跑通后，追加"张量并行（Tensor Parallelism）"脚本的需求。以下是完整的探查、方案选择、脚本构建与验证过程。

### 10.1 先弄清 ms-swift 的 TP 支持方式（探查）

TP 与 DDP 的本质区别：DDP 是**数据并行**（每卡一份完整模型，数据切分）；TP 是**模型并行**（把模型本身切开放到多卡）。写脚本前先探查 ms-swift 提供了哪几条 TP 路线：

```bash
# ① 官方多卡示例目录
ls /root/ms-swift/examples/train/multi-gpu/   # ddp / ddp_device_map / device_map / deepspeed / fsdp2_lora ...

# ② device_map 示例（官方 72B 双卡写法）
head /root/ms-swift/examples/train/multi-gpu/device_map/train.sh
# → CUDA_VISIBLE_DEVICES=0,1 + 不设 NPROC_PER_NODE + swift sft ...（模型自动切分到两卡）

# ③ megatron 真 TP 示例
head /root/ms-swift/examples/megatron/sft.sh
# → megatron sft --tensor_model_parallel_size 2 --sequence_parallel true ...（层内权重切分）

# ④ TP 相关参数定义
grep -rn 'device_groups\|tensor_parallel\|device_map' /root/ms-swift/swift/arguments/
# → sft_args.py: deepspeed_autotp_size（DeepSpeed AutoTP，需 deepspeed）
# → model_args.py: device_map='auto' 参数（accelerate 自动切分）
# → infer/deploy_args: vllm_tensor_parallel_size / sglang_tp_size（仅推理服务，与训练无关）
```

探查结论：ms-swift 训练侧可用的 TP 路线有 **3 条**：

| 路线 | 机制 | 是否可选 |
|---|---|---|
| `--device_map auto` | accelerate 自动按层切分模型到多卡（单进程） | ✅ 无需新依赖 |
| `megatron sft --tensor_model_parallel_size N` | 真 TP：层内权重切分 + all-reduce 通信 | ⚠️ 需装 megatron-core |
| `--deepspeed ... --deepspeed_autotp_size N` | DeepSpeed AutoTP | ❌ 需 deepspeed（可能触发 nvcc 编译，本路线刻意规避） |

同时确认：当前环境 **megatron-core 未安装**（`pip list | grep megatron` 为空，`megatron --help` 直接 traceback 到 `swift/cli/_megatron/main.py`），vllm/sglang 的 TP 参数只用于推理不用于训练。

### 10.2 方案决策

- **主脚本（可跑通）**：`sft_tp_2gpu.sh` —— `--device_map auto` 模型并行。理由：零新依赖、官方多卡示例同款写法（Qwen2.5-VL-72B 2×76GiB 即此方案）、当前环境立即可验证。
- **参考脚本**：`sft_megatron_tp_2gpu.sh` —— Megatron 真 TP。理由：是"教科书级"标准 TP，供将来装好 megatron-core 后用；脚本内注明依赖前置，不强行安装（避免大依赖 + 编译风险）。
- DeepSpeed AutoTP 路线直接放弃（与"全预编译、避开 nvcc"的总原则冲突）。

### 10.3 脚本一：`sft_tp_2gpu.sh`（device_map 模型并行）

```bash
#!/bin/bash
# 两卡模型并行（device_map 自动切分）SFT 冒烟脚本（Qwen3-0.6B + alpaca-zh）
# 说明：单进程，accelerate device_map 自动把模型各层分配到 2 张卡上（模型并行），
#      数据不再切分（与 DDP 的数据并行互补）。
set -e
export HOME=/root
export CUDA_VISIBLE_DEVICES=0,1
BASE=/root/ms-swift/experiments
MODEL=/mnt/e/models/models/Qwen/Qwen3-0___6B
mkdir -p "$BASE/logs"

swift sft \
    --model "$MODEL" \
    --tuner_type lora \
    --device_map auto \
    --dataset "$BASE/datasets/alpaca_gpt4_data_zh/train.csv#500" \
    --val_dataset "$BASE/datasets/self_cognition/self_cognition.jsonl" \
    --torch_dtype bfloat16 \
    --max_steps 50 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 --lora_rank 8 --lora_alpha 32 \
    --target_modules all-linear --gradient_accumulation_steps 8 \
    --eval_steps 25 --save_steps 25 --save_total_limit 2 --logging_steps 5 \
    --max_length 1024 \
    --output_dir "$BASE/output/sft_tp" \
    --warmup_ratio 0.05 --dataloader_num_workers 2
```

关键点：
- **不设 `NPROC_PER_NODE`**（单进程多卡，与 DDP 的两进程写法相反）；
- 显式 `--device_map auto`（accelerate 按每卡剩余显存自动分配层）。

**验证结果（已跑通）**：50/50 步，1m15s，train_loss 1.837，eval_loss 2.32，checkpoint-25/50 已存。

日志核心证据（模型确已切分到双卡）：

```
[INFO:swift] model.hf_device_map:
  model.embed_tokens / lm_head / model.layers.0~19   → 0（卡0）
  model.layers.20~27 / model.norm / model.rotary_emb → 1（卡1）
```

**运行方式（启动命令）**：

```bash
# WSL 交互式会话内（最简单）：
conda activate ms-swift
cd /root/ms-swift
bash experiments/sft_tp_2gpu.sh

# Windows PowerShell 后台运行（推荐，防 WSL 发行版自动关闭杀进程）：
#   wsl -d Ubuntu-24.04 -- bash -c "export HOME=/root; cd /root/ms-swift && bash experiments/sft_tp_2gpu.sh > experiments/logs/sft_tp.log 2>&1"
# 查看进度：
#   wsl -d Ubuntu-24.04 -- bash -c "tail -f /root/ms-swift/experiments/logs/sft_tp.log"
# 实测产物：experiments/output/sft_tp/v0-20261019-085826/（checkpoint-25/50）
```

### 10.4 脚本二：`sft_megatron_tp_2gpu.sh`（Megatron 真 TP，**已跑通** 2026-10-19）

> 本章曾标注“未验证（缺依赖）”，现已全部闭环：依赖安装 + 补丁 + 两卡 38/38 步训练完成。
> 完整依赖安装与排错记录见 10.6，脚本为现行可用版本。

```bash
#!/bin/bash
# 两卡【真正张量并行 TP】SFT 脚本（Megatron-LM 后端，已跑通）
# 说明：标准 TP——把每一层内部权重（注意力头/FFN 列）切分到多卡，前向/反向 all-reduce 拼合。
# 用法：megatron sft --tensor_model_parallel_size 2（TP 切分大小 = 显卡数）
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
```

**运行方式（启动命令）：**

```bash
conda activate ms-swift
cd /root/ms-swift
bash experiments/sft_megatron_tp_2gpu.sh

# Windows PowerShell 后台运行（推荐，防 WSL 发行版自动关闭杀进程）：
#   wsl -d Ubuntu-24.04 -- bash -c "export HOME=/root; cd /root/ms-swift && bash experiments/sft_megatron_tp_2gpu.sh > experiments/logs/sft_megatron_tp.log 2>&1"
# 查看进度：tail -f /root/ms-swift/experiments/logs/sft_megatron_tp.log
```

**实测结果（2026-10-19）：**
- 日志头部：`TP: 2, PP: 1, VPP: None, CP: 1, EP: 1, ETP: 1`，rank0/rank1 各绑一张卡
- 模型结构全 TP 切分：`linear_qkv: ColumnParallelLinear(TP=2)`、`linear_fc1: ColumnParallelLinear(TP=2)`、`linear_proj/linear_fc2: RowParallelLinear(TP=2)`、`output_layer: ColumnParallelLinear(TP=2)`
- 38/38 步跑完，train_loss 1.73~1.98（正常波动），每步 ~1.05s，显存 6.31 GiB/卡
- 产物：`experiments/output/sft_megatron_tp/v12-20261019-143411/`（checkpoint-25/38，Megatron 权重 + safetensors 双格式）

### 10.5 三种并行方式对比（结合实测）

| 方式 | 切分对象 | 进程数 | 通信 | 实测 | 适用场景 |
|---|---|---|---|---|---|
| DDP（已跑通） | 数据 | 2 | 每步梯度 all-reduce | SFT 3m28s/100步 | 模型单卡装得下，提速 |
| device_map（已跑通） | 模型按层 | 1 | 跨卡传激活 | 1m15s/50步，9.46GB | 模型超单卡显存 |
| Megatron TP（已跑通） | 模型层内权重 | 2 | 每层 all-reduce | 38步/44s，6.31GB/卡 | 超大模型，单层都放不下 |

> 实测观察：0.6B 模型上 device_map 的 train_loss 与 DDP 几乎一致（1.84 vs 1.80）；Megatron TP 同样正常收敛（loss 1.7~2.0），流程正确性没问题。小模型 TP 通信开销大于收益，速度不占优——TP 的价值在大模型上才体现。

### 10.6 Megatron 真 TP 全流程排错记录（重点坑）

**一、依赖安装（全预编译路线，无 nvcc 源码编译）**

```bash
pip install 'megatron-core>=0.16' 'mcore-bridge>=1.6.3'
# 实际装出：megatron-core 0.19.2 + mcore-bridge 1.6.5（ms-swift requirements/megatron.txt 声明项）
```

**二、Transformer Engine（TE）之争——本支线最大的坑**

Megatron 默认用 TE 做算子加速，但 TE 的完整安装需要源码编译（与全预编译原则冲突），且元包安装会连环报错。**最终决策：彻底卸载 TE，让 megatron-core 走官方支持的 `transformer_impl='local'`（原生 PyTorch 算子 + flash-attn）**——TE 只是算子加速库，与张量并行本身无关。

```bash
pip uninstall transformer-engine transformer_engine_cu13
rm -rf /opt/miniconda3/envs/ms-swift/lib/python3.12/site-packages/transformer_engine*
# 验证：megatron 日志出现 "Transformer Engine and Apex are not installed. Falling back to Torch optimizers" 即为 local 分支生效
```

**三、补丁清单（全部在 site-packages 内，共 4 处文件 13 个补丁）**

| 文件 | 补丁 | 说明 |
|---|---|---|
| megatron/core/extensions/transformer_engine.py | except ImportError→Exception | TE 元包缺扩展抛 FileNotFoundError 会打穿 |
| megatron/core/transformer/transformer_config.py | transformer_impl 默认 'transformer_engine'→'local' | local 实现入口 |
| megatron/core/tensor_parallel/random.py | except ModuleNotFoundError→Exception | 同上兜底 |
| mcore_bridge/model/register.py | ①use_transformer_engine 由硬编码 True 改为按 config.transformer_impl 判断 ②_set_linear_is_expert 过滤 None 的 TE 类 | mcore_bridge 假定 TE 存在 |
| mcore_bridge/bridge/gpt_bridge.py | ①TE 系 patch 全部按可用性开关 ②tuners import 保护→LoraParallelLinear=None ③非 MLA/非 MoE 分支 fused layer_norm_weight 落到独立 layernorm（attn+mlp 两处） | TE 融合 QKV/FC1 layernorm 在 local 实现不存在 |
| mcore_bridge/model/mm_gpts/deepseek_v41.py 等 6 文件 | 模块级 `import transformer_engine` 全部 try/except→None | 裸 import 打穿 |
| mcore_bridge/config/model_config.py | persist_layer_norm True→False | torch LayerNorm 不支持 persist |
| megatron/core/jit.py | enable_jit_fuser 改为 noop_decorator | 见下面 Blackwell/triton 坑 |

**四、Flage 修正**

| 现象 | 根因 | 解法 |
|---|---|---|
| `world_size (1) is not divisible by 2` | Megatron TP 要求进程数 = tensor_model_parallel_size | NPROC_PER_NODE 1→2 |
| `remaining_argv: ['--max_steps', '50']` | megatron 后端不认 max_steps | 移除，用 num_train_epochs 控制 |
| `AssertionError: persist_layer_norm not supported by torch LayerNorm` | mcore_bridge 默认 persist_layer_norm=True | 补丁改 False |
| `AssertionError: sequence parallel not supported by torch LayerNorm` | sequence_parallel 是 TE 专属优化 | 脚本移除 --sequence_parallel |
| `AssertionError: Packed sequence is not supported by DotProductAttention` | local attention 不支持 packed sequence（变长打包是 TE 特性） | 脚本加 --padding_free false |
| `ModuleNotFoundError: scaled_masked_softmax_cuda` | fused softmax 需要编译扩展 | 脚本加 --masked_softmax_fusion false |

**五、Blackwell（RTX 5060 Ti / sm_120）× triton 编译坑**

| 现象 | 根因 | 解法 |
|---|---|---|
| triton 编译 cuda_utils.so 报 `fatal error: cuda.h` | conda 环境无 CUDA toolkit 头文件 | `conda install -c nvidia cuda-cudart-dev`，并把 `$PREFIX/include/cuda*.h` 软链到 `$PREFIX/targets/x86_64-linux/include/`（triton 的 -I 搜索路径） |
| `TypeError: expected str...not NoneType` in inductor | Blackwell(arch≥100) 走 ptxas_blackwell knob，找不到 ptxas | `conda install -c nvidia cuda-nvcc`（含 ptxas），脚本 export TRITON_PTXAS_PATH / TRITON_PTXAS_BLACKWELL_PATH |
| `LLVM ERROR: PTX version 8.4 does not support target 'sm_120a'` | triton 3.8 自带 LLVM 只生成 PTX 8.4，sm_120a 最低要 8.7 | 冒烟场景直接禁用 jit_fuser（patch megatron/core/jit.py 为 noop），bias_swiglu 等退化为纯 PyTorch 执行 |

**六、Windows 侧编辑 WSL 脚本的坑**

用 PowerShell `>` 重定向或 Windows 编辑器改写 WSL 里的 .sh 会导致 CRLF 行尾/编码损坏（`--model: command not found`）。解法：在 WSL 内用 heredoc 或 `sed -i 's/\r$//'` 修复，保证 LF + UTF-8。


---

## 十一、WSL 用 DeepSpeed（ZeRO-2 / AutoTP，已跑通 2026-10-19）

> 背景：GRPO 阶段曾因 deepspeed 未装而省略 `--deepspeed`；本章补齐并额外验证了 DeepSpeed 的 AutoTP 张量并行。

### 11.1 安装（无 nvcc 路线）

```bash
pip install deepspeed
# 装出 deepspeed 0.19.7（纯 Python wheel，自带 hjson/msgpack/ninja 等）
```

- import 时的 `inference_core_ops / cutlass_ops ... compatibility check failed (CUDA_HOME does not exist)` 为**正常警告**：
  那些是推理加速算子，需要 nvcc 编译；ZeRO 数据并行（stage 1/2/3）与优化器分片是纯 PyTorch 实现，不需要编译即可用。
- 结论：**WSL 下 deepspeed 与之前 verl 的 nvcc 编译坑无关**，装好直接用。

### 11.2 用法（ms-swift 侧）

- `--deepspeed zero2`：预设配置名（`zero0/zero1/zero2/zero3/zero2_offload/zero3_offload`，在 `swift/config/*.json`），或传自定义 json 路径。
- `--deepspeed_autotp_size 2`：DeepSpeed AutoTP（自动张量并行，把层内权重切到多卡），**仅支持 full 参数微调**（`--tuner_type full`）。
- 与 `--device_map` 不兼容（ms-swift 会直接报错）。

### 11.3 三个脚本与实测

| 脚本 | 配置 | 结果 |
|---|---|---|
| `sft_ds_zero2_2gpu.sh` | SFT + LoRA + zero2 | 100/100 步，train_loss 1.80（与 DDP 一致），2m58s，8.45 GiB |
| `rl_grpo_ds_2gpu.sh` | GRPO + LoRA + zero2 | 20/20 步，train_loss 0.0012，reward 0.09~0.41 波动，5m20s，2.53 GiB |
| `sft_ds_autotp_2gpu.sh` | SFT + full + zero2 + AutoTP(2) | 50/50 步，train_loss 3.72（full 从头学收敛慢属正常），1m43s，10.97 GiB，`tensor_parallel: {'autotp_size': 2}` |

### 11.4 结论与选型

- 显存视角：ZeRO-2 省的是**优化器状态+梯度**分片；0.6B + LoRA 场景参数太少，优势不明显（8.45GB 甚至高于纯 DDP 的 LoRA），**大模型 full 微调才划算**。
- AutoTP 是 DeepSpeed 版的"张量并行"：一行参数即可（对比 Megatron 真 TP 需要整套补丁），但仅 full 微调可用；0.6B 上通信开销 > 收益，速度同样不占优。
- 日常小模型冒烟：继续用现有 DDP 脚本即可；deepspeed 留给模型装不下/要省显存的场景（zero2_offload/zero3 等可进一步压显存）。


---

## 十二、更大模型实测：Qwen3-4B + DeepSpeed（2026-10-19）

> 动机：验证 deepspeed 在大模型上的显存价值（0.6B + LoRA 时 ZeRO-2 无优势）。

### 12.1 E 盘模型探查结论

- E 盘只有两个模型：
  - `Qwen/Qwen3-0___6B`（1.5G，HF safetensors）——训练全程使用。
  - `ISTA-DASLab/Qwen3___8-Flash-Next-GSQ-RCO-Coder-GGUF`（56G）——**GGUF 量化格式**（IQ1_M 量化 + mmproj 多模态投影），llama.cpp 生态专用，**量化权重已丢失精度、不能用于 ms-swift SFT/RL 训练**。
- 结论：E 盘没有可直接训练的大模型，需从 ModelScope 另下 HF safetensors 版。

### 12.2 下载 Qwen3-4B（8GB 权重）

```bash
export HOME=/root
/opt/miniconda3/envs/ms-swift/bin/modelscope download \
    --model Qwen/Qwen3-4B --local_dir /mnt/e/models/models/Qwen/Qwen3-4B
# 7.6G，3 个 safetensors 分片，HF 格式，可直接训练
```

### 12.3 双卡 4B 训练：三次 OOM，根因是 WSL 内存上限（铁证）

- `sft_ds_zero3_qwen4b.sh`（full + ZeRO-3）：**GPU OOM**——ZeRO-3 虽分片存储参数，但前向每层仍要 allgather 完整权重（8GB）+ activation，16GB 卡放不下。
- `sft_ds_zero3_qwen4b_v2.sh`（full + ZeRO-3 + max_length 256 + expandable_segments）：仍 GPU OOM。
- `sft_ds_zero2_qwen4b_lora.sh`（LoRA + ZeRO-2）：**训练正常**（10 步 eval_loss 2.23、显存 9.5GB），但训练中途进程被 SIGHUP/SIGTERM 杀掉，三次复现。
- 内核日志铁证（`/var/log/syslog`）：

```
kernel: Out of memory: Killed process 796 (python3.12) total-vm:148443528kB, anon-rss:7514300kB
kernel: oom-kill:constraint=CONSTRAINT_NONE, ..., global_oom, task_memcg=/init.scope, task=python3.12
```

- **根因**：WSL VM 默认内存 = 物理内存 50%（本机 31.2GB → VM 仅 15GB）。双卡训练 2 个 rank 进程各占 ~7.5GB 物理内存（加载完整 8GB 权重）+ 系统开销 > 15GB → 内核 global OOM 杀进程。**不是 ms-swift/deepspeed 的问题，是 VM 内存天花板**，任何训练参数微调都救不了（每进程加载 8GB 权重是硬需求）。
- 注意区分两个 OOM：**GPU 侧 OOM**（full 微调，显存 16GB 不够）与**系统侧 OOM**（双卡 LoRA，VM 内存 15GB 不够）。

### 12.4 单卡 4B + LoRA + ZeRO-2：跑通 ✅

不扩容内存的前提下，唯一可行路线是**单卡**（单进程 RSS ~7.5GB < 15GB；显存 8GB 权重 + LoRA < 16GB）：

```bash
bash experiments/sft_ds_zero2_qwen4b_lora_single.sh
```

- 坑：单卡 deepspeed 走 MPI 发现路径，报 `No module named 'mpi4py'` → `conda install -p /opt/miniconda3/envs/ms-swift -c conda-forge mpi4py`（预编译，装出 4.1.2）。
- 实测：10/10 步，train_loss 2.117，eval_loss 3.03，显存 8.54 GiB，checkpoint-10 已存，ERR 0。

### 12.5 结论与建议

| 方案 | 结果 |
|---|---|
| 4B full（zero2/zero3）双卡 | GPU 16GB OOM，需 TP/Megatron 或更大显存 |
| 4B LoRA 双卡 | 系统 OOM（VM 15GB 内存不够，内核杀进程） |
| 4B LoRA 单卡 | ✅ 可跑（8.5GB 显存，VM 内存足够） |
| 0.6B 全流程（DDP/TP/Megatron/DS） | ✅ 双卡全跑通（前 11 章） |

- 若要在**双卡**上跑 4B：给 WSL 扩容——`C:\Users\LBY\.wslconfig` 加 `memory=24GB`（物理 31GB，留 ~7GB 给 Windows），然后 `wsl --shutdown` 重启生效（会中断当前所有 WSL 会话）。
- 消费级 16GB 卡 + 4B 的务实路线：**单卡 LoRA** 或 **双卡 Megatron/device_map TP + LoRA**。

### 12.6 扩容 WSL 内存后：双卡 4B 跑通 ✅（2026-10-19）

- 修改 `C:\Users\LBY\.wslconfig`：`[wsl2]` 段加 `memory=24GB`（物理 31.2GB，留 ~7GB 给 Windows），`wsl --shutdown` 重启生效。验证：`MemTotal 24608840 kB ≈ 23.5GB`。
- 重跑之前 OOM 的脚本（`sft_ds_zero2_qwen4b_lora_v2.sh`，双卡 LoRA + ZeRO-2，20 步）：**20/20 步，train_loss 1.725，9.53 GiB/卡，checkpoint-20 已存，0 错误**，训练全程无 OOM。
- 结论：**WSL 内存上限就是此前"双卡 4B 必被杀"的唯一瓶颈**。动态内存机制下 24GB 上限平时不占 Windows 内存（见第十二章开头说明），只在训练峰值时按需使用。

### 12.7 最终结论表（更新）

| 方案 | 结果 |
|---|---|
| 4B full（zero2/zero3）双卡 | GPU 16GB OOM，需 TP/Megatron 或更大显存 |
| 4B LoRA 双卡（WSL 15GB 时） | 系统 OOM（内核杀进程，有 /var/log/syslog 铁证） |
| 4B LoRA 双卡（WSL 扩容 24GB 后） | ✅ 20/20 步，train_loss 1.725，9.53GB/卡 |
| 4B LoRA 单卡（WSL 15GB 时） | ✅ 10/10 步，train_loss 2.12，8.54GB |
| 0.6B 全流程（DDP/TP/Megatron/DS） | ✅ 双卡全跑通（前 11 章） |


---

## 十三、4B 在不同 ZeRO 策略下的实测对比（2026-10-19）

> 统一配置：Qwen3-4B + LoRA(rank8) + 双卡 + max_length 256 + 500 条 + 10 步 + batch 1 + grad_accum 4 + gradient checkpointing。
> 脚本：`sft_4b_zero_matrix.sh`（循环 7 个策略）；日志 `logs/sft_4b_zero_matrix.log`、`logs/sft_4b_zero3_offload.log`。

### 13.1 实测结果表

| 策略 | 结果 | train_loss | 显存/卡 | 训练耗时 | 说明 |
|---|---|---|---|---|---|
| **无 DS（纯 DDP）** | ✅ | 2.181 | 8.49 GiB | **16.9s** | 最快；无 deepspeed 引擎开销 |
| zero0（分片关闭） | ✅ | 2.224 | 8.62 GiB | 34.9s | 走 DS 引擎但零分片，白付初始化开销 |
| zero1（优化器分片） | ✅ | 2.182 | 8.82 GiB | 47.0s | LoRA 优化器状态极小，无收益 |
| zero2（优化器+梯度分片） | ✅ | 2.181 | 8.54 GiB | 31.3s | 与 DDP 持平；LoRA 场景无显存优势 |
| zero3（全分片） | ✅ | 2.181 | 8.54 GiB | **152.1s** | **最慢**：每层前向 allgather 完整权重，通信开销 9 倍 |
| zero2_offload（CPU offload） | ❌ | — | — | — | MissingCUDAException |
| zero3_offload（CPU offload） | ❌ | — | — | — | 同上 |

### 13.2 两个关键结论

1. **4B + LoRA 场景：ZeRO 没有显存价值，反而拖慢速度**。
   - LoRA 可训练参数只有 ~千万级，优化器状态（Adam）几乎可忽略 → ZeRO-2/3 分片省不下显存（全部 8.5~8.8 GiB 持平，瓶颈是冻结权重 8GB）。
   - 速度：纯 DDP 最快（16.9s），zero3 最慢（152s，allgather 通信 9 倍开销）。**LoRA 冒烟/训练请直接用纯 DDP，别上 deepspeed。**
   - ZeRO 的真正价值在 **full 微调大模型**（优化器状态 16GB 起步，分片才能装下）——但 4B full 在 16GB 卡上 GPU 侧 OOM（见 12.3），ZeRO 救不了显存上限。

2. **offload 版本是"零编译"路线的硬边界**（与 verl/nvcc 同源问题）。
   - `zero2_offload`/`zero3_offload` 的 CPU 优化器走 **DeepSpeedCPUAdam**，这是需要 **nvcc 编译的 CUDA 算子**；本环境无 CUDA_HOME（全预编译 wheel 路线），直接报：
     `deepspeed.ops.op_builder.builder.MissingCUDAException: CUDA_HOME does not exist, unable to compile CUDA op(s)`
   - 若要用 CPU offload 压显存：需先安装 CUDA toolkit（`conda install -c nvidia cuda-toolkit` 等）提供 nvcc，或换无 CPUAdam 依赖的方案。

### 13.3 选型建议（4B + 双卡 + LoRA）

| 需求 | 选型 |
|---|---|
| 最快冒烟/训练 | **纯 DDP（无 --deepspeed）** |
| 与 zero 兼容性验证 | zero2（与 DDP 同 loss 同显存，仅多引擎开销） |
| 显存吃紧要省 | 换更小 max_length/batch；offload 不可用（需 nvcc） |
| full 微调 4B | 需更大显存卡或 TP/Megatron（见 12.7） |


---

# 14. offload 尝试、环境事故与恢复（2026-10-19 ~ 10-20）

> 结论先行：**offload 路线（zero2_offload / zero3_offload）已正式放弃**。期间装 CUDA 工具包时 conda solver 把 pytorch/flash-attn 悄悄降级，环境一度不可用；已完整恢复回 torch 2.14.1(cuda134) + flash-attn 2.8.3，并用原 SFT 脚本复跑验证通过。本文档记录全过程踩坑，供日后环境修复参考。

## 14.1 为什么试 offload

第十三章实测 4B + LoRA 时，显存瓶颈在**冻结权重 8GB**（16GB 卡 × 2），zero2/zero3 都救不了。offload（优化器状态/参数挪 CPU）是理论上的压显存手段，于是按用户要求尝试 `zero2_offload` / `zero3_offload`。

## 14.2 尝试过程与连环坑

1. **CPUAdam 需要 nvcc 编译**：offload 的 CPU 优化器走 `DeepSpeedCPUAdam`，需要编译 CUDA 算子，本环境（全预编译 wheel、无 CUDA_HOME）直接报 `MissingCUDAException`。
2. **发现 env 里其实有 nvcc 12.4**（cuda-nvcc 12.4.131 被 conda 装乱），编译报 `CUDAMismatchException`（nvcc 12.4 vs torch 编译的 CUDA 13.x 不符）。
3. **尝试装 cuda-nvcc 13.4.92 的连环坑**：
   - 清华 main 新式包布局：nvcc 移到 `targets/x86_64-linux/bin/`；
   - 手动建的软链曾形成**循环**（ln -s 指向自己）；
   - `conda install` 反复被 `python=3.12` 的 pin 挡住；
   - 拉取 repodata 反复 `SSLEOFError`（pkgs/r 通道不稳）；
   - `--force-reinstall` 对已记录未解压的包无效。
4. **改用 pip 装 `nvidia-cuda-nvcc`（13.4.92）成功**，`setup_nvcc_build.sh` 软链进 `targets/x86_64-linux/` 后 CPUAdam 仍报版本不匹配。

## 14.3 重大事故：pytorch 被悄悄降级

排查中发现：装 cuda 相关 conda 包时，**conda solver 把 pytorch 从 2.14.1 降级成 2.5.1(cuda126)、libtorch 2.5.1、flash-attn 从 2.8.3 降级成 2.7.2**，整个环境破坏（`torch.__version__ == 2.5.1.post207`）。这是 offload 尝试最深的坑：**用 conda 装 CUDA 工具包时，务必先检查 solver 计划里是否动了 pytorch/libtorch/flash-attn**。

## 14.4 恢复过程（全部离线，利用 conda 缓存）

缓存里保留了原装的精确 build（清华 main / conda-forge）：

| 包 | 精确 build |
|---|---|
| pytorch | `2.14.1=cuda134_mkl_py312_h54e5dea_300` |
| libtorch | `2.14.1=cuda134_mkl_hf2d2af2_300` |
| flash-attn | `2.8.3=py312haae3585_4` |
| cuda-nvrtc | `13.4.92=h7354ed3_1` |

恢复命令（solver 在线卡死/SSL 失败时，用 `--offline --no-deps` 逐组装回）：

```bash
conda install -p $PREFIX -y --offline --no-deps   'pytorch=2.14.1=cuda134_mkl_py312_h54e5dea_300'   'libtorch=2.14.1=cuda134_mkl_hf2d2af2_300'   'flash-attn=2.8.3=py312haae3585_4'   'cuda-nvrtc=13.4.92=h7354ed3_1'
```

之后 `import torch` 报一连串缺库，逐个补齐（按出现顺序）：

1. **libmkl_intel_lp64.so.3**：mkl 包"记录在但文件没解压"（conda 缓存 bug 复发）。离线 `--force-reinstall --no-deps 'mkl=2026.1.0=hd2095e1_245'`（**必须 2026.1.0**，旧版 mkl 只提供 `.so.2`，torch 2.14 要 `.so.3`）。
2. **libcublasLt.so.13 / libcublas.so.13 等 CUDA 13 运行库**：`--offline --no-deps` 装回 `cuda-cudart / libcublas=13.8.0.4 / libcufft=12.4.0.43 / libcurand=10.4.4.72 / libcusolver=12.3.5.3 / libcusparse=12.8.6.72 / libcudnn=9.26.0.51`（全部清华 main 缓存）。
3. **torch 2.14 的库查找机制**：`torch/__init__.py` 的 `_preload_cuda_lib` 遍历 **sys.path**，从 `site-packages/nvidia/<folder>/lib/` 找运行库（pip 布局）。用脚本把 `$PREFIX/lib` 里的 13 系列库**软链**成该布局（共 119 个链接，覆盖 cublas/cudnn/cuda_nvrtc/cuda_runtime/cufft/cuda_cupti/curand/nvjitlink/cusparse/cusolver/nccl）。
4. **libcudart.so.13.4.92 真实文件缺失**：清华 main 的 cuda 包已拆分——`cuda-cudart`（元包，只装 lib/ 软链）+ `cuda-cudart_linux-64`（真实文件在 `targets/x86_64-linux/lib/`）。手动从缓存 `.conda` 包解压 `cuda-cudart_linux-64-13.4.92` 补齐（脚本：zipfile 读 pkg 归档 → zstandard 解压 → tarfile 解出 → 复制到 env）。
5. **libprotobuf.so.35.1.0**：`libprotobuf=7.35.1`（soname 规则 = **去掉主版本号的次版本**：7.35.1 → `.35.1.0`；之前误装 5.28.2 只有 `.28.2.0`）。
6. **libcupti.so.13**：`cuda-cupti=13.4.92`（清华 main）。
7. **libcudss.so.0**：`libcudss=0.8.0.10`（缓存离线装；cuDSS 是 CUDA 13 新增库，torch 2.14 硬链接）。

## 14.5 恢复验证（已通过）

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
# 2.14.1 13.4 True
python -c 'import flash_attn; print(flash_attn.__version__)'   # 2.8.3
# 双卡可见：2 NVIDIA GeForce RTX 5060 Ti
```

原脚本冒烟复跑：`bash experiments/sft_2gpu.sh` → **100/100 步完成，train_loss 1.802**（与事故前 1.80 一致），eval_loss 2.288，双卡 8.39 GiB/卡。

## 14.6 最终结论

- **offload（zero2_offload / zero3_offload）放弃**：CPUAdam 需要 nvcc 编译，与本项目"全预编译零编译"路线的硬约束冲突；且即使编译成功，LoRA 场景显存瓶颈在冻结权重，offload 收益有限。
- **环境已恢复**，全部既有脚本（DDP / TP / Megatron / DeepSpeed / 4B ZeRO 矩阵）应可复跑；如需复跑 Megatron 脚本，注意 site-packages 上的 13 处补丁仍在（见 10.5）。
- **教训**：① 在本环境用 conda 装任何 CUDA 工具包前，先 `conda install --dry-run` 检查 solver 计划，防止 pytorch/libtorch/flash-attn 被连带降级；② 清华 main 新式 cuda 包分"元包 + _linux-64 拆分包"，真实库文件在后者；③ torch 2.14 起 CUDA 库从 `site-packages/nvidia/*/lib` 查找（pip 布局），conda 装的库需要软链补齐；④ 缓存包解压用 `conda install --offline --force-reinstall --no-deps` 或手动 zipfile+zstandard 解包，别依赖 conda 的"已安装"记录。
