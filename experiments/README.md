# ms-swift WSL 训练实验（experiments）

> 2026-10-18 搭建与验证记录。目的：验证 ms-swift 在 WSL2 双卡环境下 SFT、RL（DPO）、RL（GRPO）流程可端到端训练。
> 2026-11-05 追加：多模态（图文）CPT 流程（Qwen2-VL-2B + mock 图文 CPT 集、Gemma4-E2B）验证通过。
> 2026-11-07 整理：脚本按用途分目录（`sft/` `rl/` `cpt/` `infer/`），说明文档移到 `docs/`，根目录只保留本 README 与数据/日志/产物目录。

## 0. 目录结构

```
experiments/
├── README.md                      # 本文件（总索引）
├── docs/                          # 说明文档
│   ├── ms-swift安装到训练全流程记录.md
│   ├── ms-swift快速上手.md
│   ├── ms-swift多模态CPT开发记录.md
│   └── megatron_patches_notes.md
├── pt/                            # 纯文本预训练 demo（PT 数据格式：{"text": ...}）
├── sft/                           # 文本 SFT（DDP / 模型并行 / Megatron TP / DeepSpeed ZeRO / 4B）
├── rl/                            # RL：DPO / GRPO（含 DeepSpeed 变体）
├── cpt/                           # 多模态 CPT（Qwen2-VL / Gemma4 + mock 数据生成 + start 包装）
├── infer/                         # 推理与调试脚本（含丢图复现、官方样例）
├── datasets/                      # 数据集（含 mock_vl_cpt）
├── logs/                          # 运行日志（不入库）
└── output/                        # 训练产物（不入库）
```

**运行约定**：先 `cd /root/ms-swift`，再 `bash experiments/<子目录>/<脚本>.sh`。脚本内 `--output_dir output/...` 是相对仓库根的路径，因此不要 `cd` 进子目录再执行。

旧路径 → 新路径（2026-11-07 整理，历史文档中出现的旧路径按此对应）：

| 旧 | 新 |
|---|---|
| `experiments/sft_*.sh` | `experiments/sft/sft_*.sh` |
| `experiments/rl_*.sh` | `experiments/rl/rl_*.sh` |
| `experiments/cpt_*.sh`、`start_cpt_*.sh`、`make_mock_vl_cpt.py` | `experiments/cpt/…` |
| `experiments/infer_gemma4.py`、`gemma4_hf_*.py`、`repro_*.py` | `experiments/infer/…` |
| `experiments/*.md`（README 除外） | `experiments/docs/…` |

## 1. 环境

| 项 | 值 |
|---|---|
| WSL 发行版 | Ubuntu-24.04（WSL2，`D:\WSL\Ubuntu-24.04\ext4.vhdx`） |
| conda | `/opt/miniconda3`（conda 26.7.1） |
| 环境 | `ms-swift`（Python 3.12.15） |
| torch | 2.14.1（cuda134，conda-forge 预编译，支持 sm_120） |
| flash-attn | 2.8.3（pkgs/main 预编译） |
| torchvision | 0.29.2+cu134（VLM 训练必需，pip `--no-deps` 装，未动 torch） |
| qwen_vl_utils / decord | 0.0.14 / 0.6.0（Qwen2-VL 图像预处理依赖） |
| transformers | 5.19.0（未改动；Gemma4 patch 全在 ms-swift 侧） |
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
| 多模态 CPT | `mock_vl_cpt/`（mock 生成） | 64+8 条图文对：matplotlib 合成图表/示意图（72 张 512px）+ 每图配 250~450 字中文描述；`cpt/make_mock_vl_cpt.py` 可复现 |
| PT demo | `demo_pt_cpt/pt_{train,val}.jsonl`（mock） | 10+2 条纯文本语料，每行 `{"text": ...}`，内容取自本项目真实经验 |
| CPT demo | `demo_pt_cpt/cpt_{train,val}.jsonl` + `images/`（mock） | 8+2 条图文对（旧版 messages 结构），图从 `mock_vl_cpt` 挑选复制，答案与图内容一致；`datasets/make_demo_pt_cpt.py` 可复现 |
| CPT 结构化 demo | `demo_pt_cpt/structured/cpt_text_images_{train,val}.jsonl`（mock） | 8+2 条图文对，每行 `{"text": "…", "images": ["/abs/x.png"]}`，**无 role**；`datasets/make_demo_text_images.py` 可复现 |
| PT 结构化 demo | `demo_pt_cpt/structured/pt_text_{train,val}.jsonl`（mock） | 10+2 条纯文本语料，每行只有 `{"text": ...}` |

**PT 与 CPT 的数据格式差异**（`datasets/make_demo_text_images.py` 生成）：

| | 一行数据长什么样 | 训练命令 |
|---|---|---|
| PT（纯文本预训练） | `{"text": "Gemma 4 的 E2B 与 E4B 是同一代的两档模型……"}` | `swift pt --use_chat_template false --loss_scale all` |
| CPT（图文，**推荐**） | `{"text": "这是一张柱状图……", "images": ["/root/ms-swift/experiments/datasets/demo_pt_cpt/images/bar_00.png"]}` | 同上，`--use_chat_template false`，text 里**不带** `<image>` 占位 |
| CPT（图文，旧版 messages） | `{"messages": [{"role":"user","content":"<image>请详细描述这张图片……"},{"role":"assistant","content":"这是一张柱状图……"}], "images": ["…"]}` | `swift pt --use_chat_template true --loss_scale all` |

差别就三点：PT 与图文 CPT 现在**同一套 key**（`text` + `images`），不再需要 role；`text` 里不要写 `<image>` 占位符，图片位置由 `images` 字段决定（框架自行插入 image pad）；旧版 messages 写法仍可用，但一定带 role 标记 token。

两点容易混淆的地方：

- **纯文本 CPT 与 PT 完全同构**：都用 `{"text": ...}`，没有 role、不套模板。旧版把图文也写成 `messages`，是因为当时的例子走的是 LLaVA-Pretrain 那套 `MessagesPreprocessor`（`swift/dataset/dataset/mllm.py:1011`，user="请描述这张图"、assistant=caption）。实测 `{"text": …, "images": […]}` + `--use_chat_template false` 同样被 `swift pt` 接受，且序列里**零 role 标记**：Gemma4 图文样本解码出来是 `<|image>` + 图像 pad token + 正文 + `<eos>`，没有 `<start_of_turn>`；Qwen3.5 纯文本解码出来是正文 + `<|im_end|>`，没有 `<|im_start|>user`。
- **判据不是外壳而是 loss 覆盖范围**：SFT 默认 `loss_scale=default`，只算 assistant 段（`swift/arguments/base_args/template_args.py:140`）；PT/CPT 用 `loss_scale=all`，正文与模板标记全部算 loss（`swift/arguments/pretrain_args.py:10` 默认即 `all`）。旧版 messages 路径一定会把消息成对解析成 user/assistant 并注入 role 标记（`swift/template/base.py:1444-1449`），追求零模板噪声就走 `text` 字段。

> ⚠️ zhihu_rlhf_3k 的 chosen/rejected 回答超长，直接用于 DPO 会出现 `logits/chosen=nan` 且 OOM；**改用 ms-swift 注册表官方数据集 `swift/orca_dpo_pairs` 后训练完全正常**。zhihu 数据保留作参考。

## 3. 脚本（按目录分类）

### 3.1 pt/（5 个，PT/CPT demo：text + images 结构）

- `pt/pt_text_demo.sh`：纯文本 PT 最小 demo（Qwen3-0.6B + `{"text": ...}` 语料；`--use_chat_template false --loss_scale all`，不套模板、全 token 算 loss）
- `pt/pt_text_gemma4_1gpu.sh`：纯文本 PT（Gemma4-E2B，10 条语料 2 epoch）
- `pt/pt_text_qwen35_1gpu.sh`：纯文本 PT（Qwen3.5-0.8B，比 Qwen3-0.6B 更新；**必须显式** `--model_type qwen3_5 --template qwen3_5`，否则自动匹配有歧义报错）
- `pt/cpt_text_images_gemma4_1gpu.sh`：图文 CPT（Gemma4-E2B，`{"text","images"}` 无 role，`freeze_vit/aligner`）
- `pt/cpt_text_images_qwen3vl_1gpu.sh`：图文 CPT（Qwen3-VL-2B-Instruct，同样无 role；template 由框架自动匹配）

### 3.2 sft/（13 个，文本 SFT）

- `sft/sft_2gpu.sh`：两卡 DDP SFT（LoRA，Qwen3-0.6B + alpaca-zh）
- `sft/sft_tp_2gpu.sh`：两卡**模型并行** SFT（`--device_map auto`，单进程多卡）
- `sft/sft_megatron_tp_2gpu.sh`：两卡**真张量并行 TP**（Megatron 后端，**已跑通**，见 docs 全流程记录第十章）
- `sft/sft_ds_zero2_2gpu.sh`：两卡 **DeepSpeed ZeRO-2** SFT（LoRA）
- `sft/sft_ds_autotp_2gpu.sh`：两卡 **DeepSpeed AutoTP 张量并行** SFT（`--deepspeed_autotp_size 2`，full 微调）
- `sft/sft_ds_zero2_qwen4b_lora.sh` / `_single.sh` / `_v2.sh` / `_v3.sh`：4B 单卡/双卡 ZeRO-2 LoRA 系列
- `sft/sft_ds_zero3_qwen4b.sh` / `_v2.sh`：4B ZeRO-3
- `sft/sft_4b_zero_matrix.sh`：4B ZeRO 矩阵对比（nods / zero0 / 1 / 2 / 3）
- `sft/sft_4b_zero3_offload.sh`：ZeRO-3 + offload（CPUAdam 需 nvcc，**未跑通，留档**）

### 3.3 rl/（8 个，覆盖 ms-swift 支持的 9 种 `--rlhf_type` 中的 6 种）

- `rl/rl_dpo_2gpu.sh`：两卡 DDP DPO（LoRA，`swift/orca_dpo_pairs#500`）
- `rl/rl_orpo_2gpu.sh`：两卡 DDP **ORPO**（单阶段偏好对齐，chosen/rejected 拼一条序列，**不需要 ref model**）
- `rl/rl_simpo_2gpu.sh`：两卡 DDP **SimPO**（无参考模型，长度归一化 + reward margin，`beta 2.0` / `simpo_gamma 1.0`）
- `rl/rl_cpo_2gpu.sh`：两卡 DDP **CPO**（SimPO 偏好项 + 一条 SFT(NLL) 项，`cpo_alpha 1.0`，无 ref model）
- `rl/rl_kto_2gpu.sh`：两卡 DDP **KTO**（只要「好/坏」二进制标签，用 `ultrafeedback-binarized-preferences-cleaned-kto`，不需要成对比较）
- `rl/rl_rm_2gpu.sh`：两卡 DDP **Reward Model**（同一份偏好数据训打分器，`center_rewards_coefficient 0.01`；产物是 value head 打分模型，供 PPO 的 `--reward_model` 用）
- `rl/rl_grpo_2gpu.sh`：两卡 DDP GRPO（LoRA，gsm8k + accuracy reward，native rollout）
- `rl/rl_grpo_ds_2gpu.sh`：两卡 **DeepSpeed ZeRO-2** GRPO

算法谱系速记：**离线偏好**（DPO / ORPO / SimPO / CPO，喂 chosen+rejected）→ **离线标签**（KTO，只喂好/坏）→ **打分器**（RM，产出 reward 模型）→ **在线采样**（PPO / GRPO，模型自己生成再由 reward 打分）。`--rlhf_type ppo` 需要额外 `--reward_model`（本仓库暂无可用 checkpoint，未写脚本）；`--rlhf_type gkd` 需要 teacher model（未写脚本）。

### 3.4 cpt/（9 个，多模态图文 CPT）

- `cpt/cpt_vl_demo.sh`：**最小可跑 demo**（Gemma4-E2B + 8 条 mock 图文，2 epoch / 8 步，10.3s，10.75 GiB；验证 `images` 字段与 `<image>` 占位）
- `cpt/cpt_vl_2gpu.sh`：两卡多模态 CPT（Qwen2-VL-2B-Instruct + mock 图文集；`swift pt` + `--use_chat_template false --loss_scale all`，LoRA，冻结 ViT/aligner）
- `cpt/cpt_vl_infer.sh`：Qwen 版 CPT adapter 推理（`swift infer --adapters <ckpt>`）
- `cpt/make_mock_vl_cpt.py`：mock 图文集生成器（8 类图，64 训练 + 8 验证 = 72 张 512px PNG，同 seed 数值对齐）
- `cpt/cpt_gemma4_1gpu.sh`：Gemma4-E2B 单卡 CPT（**已跑通**，16/16 步）
- `cpt/cpt_gemma4_2gpu.sh`：Gemma4 双卡（WSL 下 `device not ready`，**未跑通，留档**）
- `cpt/cpt_gemma4_infer.sh`：Gemma4 CLI 推理（需 `swift/model/models/gemma.py` 适配层补丁，坑 14/15）
- `cpt/start_cpt_vl.sh` / `cpt/start_cpt_gemma4.sh`：后台启动包装（setsid + 打印首屏日志）

### 3.5 infer/（5 个，推理与调试）

- `infer/gemma4_official_demo.py`：**按 ModelScope 官方模型卡样例**调用 Gemma4（modelscope `AutoProcessor` + `AutoModelForMultimodalLM`，图文/纯文本两种模式，含 `parse_response` 解析）；卡号/模型/图片/模式/最大 token 全部写成脚本头常量，`python` 直接跑，不需要命令行参数
- `infer/infer_gemma4.py`：Gemma4 确定性推理验收脚本（encode → collate → generate，8/8 通过）
- `infer/gemma4_hf_runner.py`：**纯 transformers** 版 Gemma4 推理（类封装，`CUDA_VISIBLE_DEVICES` 锁定 1 号卡）
- `infer/gemma4_hf_simple.py`：24 行极简版（同样只用 1 号卡）
- `infer/repro_gemma4_image_drop.py`：复现"推理看不到图片"（A/B/C 三段对照，证明差异只在是否消费 `mm_encoder_outputs`）

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
| 多模态 CPT（Qwen2-VL-2B + 图文集） | ✅ 16/16 步，train_loss 2.86→**1.95**，train_loss 均值 2.151，eval_loss 1.956，token_acc 0.44→0.60，**5.24GB/卡**，22s；推理 8/8 张图，图表数值/占比与 GT 高度一致 |
| 多模态 CPT（Gemma4-E2B + 图文集，单卡） | ✅ 16/16 步，train_loss 3.685，**10.75GiB**，30.6s（2.09 samples/s）；推理 8/8 张图全部正确识别并描述（图表类型/数值/布局与图像一致）；双卡在 WSL 下受 `device not ready`（GPU mapping）限制无法跑 |
| Gemma4 纯 transformers 推理 | ✅ 1 号卡跑通；bar_00 输出 48/76/106/84、pie_02 输出 13/31/38/18，与 mock GT 完全一致 |
| Gemma4 纯 transformers 推理（内置架构图） | ✅ 不传 `--image` 直接读笔记里的 Gemma-1 解码层架构图，正确复述 `hidden states (b,s,2048)`、`RMSNorm eps 1e-6`、`q/k/v_proj 2048→8*256 / 1*256`、`RoPE`、`input layernorm` |
| Gemma4 官方模型卡样例（modelscope API） | ✅ 1 号卡跑通（E2B 权重）；图文样例 prompt 274 token、生成 186 token / 7.5s，正确识别 GoldenGate 桥；纯文本样例 28 token → 28 token / 2.3s；`parse_response` 均返回 `{'role': 'assistant', 'content': ...}` |
| PT 纯文本 demo（`pt/pt_text_demo.sh`） | ✅ Qwen3-0.6B，10 条语料 3 epoch = 30/30 步，8s，train_loss 3.941，eval_loss 4.564→**4.414**，token_acc 0.25→0.51，显存 1.58 GiB |
| CPT 图文 demo（`cpt/cpt_vl_demo.sh`） | ✅ Gemma4-E2B，8 条图文 2 epoch / 8 步，10.3s，train_loss 4.104，eval_loss 5.773→**5.428**，显存 10.75 GiB，无 `IndexError`（`<image>` 占位与图片索引均正确） |
| 纯文本 PT（`pt/pt_text_gemma4_1gpu.sh`） | ✅ Gemma4-E2B，10 条 2 epoch = 20/20 步，8.9s，train_loss 8.2，eval_loss **7.121**，显存 10.43 GiB |
| 纯文本 PT（`pt/pt_text_qwen35_1gpu.sh`） | ✅ Qwen3.5-0.8B，20/20 步，11.3s，train_loss 3.725，eval_loss **4.172**，显存 **2.28 GiB** |
| 图文 CPT 无 role（`pt/cpt_text_images_gemma4_1gpu.sh`） | ✅ Gemma4-E2B，8 条 2 epoch / 8 步，8.95s，train_loss 4.832，eval_loss **6.219**，10.61 GiB；序列解码为 `<|image>`+image pad+正文+`<eos>`，**零 role 标记**，`images_dir` 已生成 |
| 图文 CPT 无 role（`pt/cpt_text_images_qwen3vl_1gpu.sh`） | ✅ Qwen3-VL-2B-Instruct，8/8 步，9.07s，train_loss 1.871，eval_loss **2.813**，4.63 GiB；token_acc 0.42→0.67 |
| RL ORPO（`rl/rl_orpo_2gpu.sh`） | ✅ 20/20 步，60.7s，train_loss 1.666，14.42 GiB；`nll_loss`/`log_odds_ratio` 指标正常，无 ref model |
| RL SimPO（`rl/rl_simpo_2gpu.sh`） | ✅ 20/20 步，61.6s，train_loss 3.43，14.42 GiB；`rewards/margins` 在 ±0.3 附近抖动 |
| RL CPO（`rl/rl_cpo_2gpu.sh`） | ✅ 20/20 步，60.4s，train_loss 7.826（含 NLL 项，量级偏大属正常），14.42 GiB |
| RL KTO（`rl/rl_kto_2gpu.sh`） | ✅ 20/20 步，50.4s，train_loss 0.472，12.83 GiB；367 条有效样本（`label` true/false），`rewards/margins` 0.28~0.56 |
| RL Reward Model（`rl/rl_rm_2gpu.sh`） | ✅ 20/20 步，44.1s，train_loss 0.616，**1.94 GiB**；`rewards/accuracies` 0.875，`rewards/margins` 0.39~0.83（chosen 稳定高于 rejected） |

> 以上 5 个 RL 脚本都是**跑通性冒烟**（`max_steps 20`、400 条样本），只证明算法链路、数据映射和显存可以跑，不代表效果。ORPO/SimPO/CPO 在 20 步内 margin 仍在 0 附近抖动，要看收敛得把 `max_steps` 提到 100+。

产物目录：`experiments/output/sft/`、`experiments/output/rl_dpo/`（v5 为最终成功版本）、`experiments/output/rl_grpo/`（v2 为最终成功版本）、`experiments/output/sft_tp/`、`experiments/output/cpt-gemma4-1gpu/`、`experiments/output/rl_{orpo,simpo,cpo,kto,rm}/`。

## 5. 注意事项（踩坑记录）

1. **HOME 问题**：通过非交互 `wsl -d ... -- bash -c` 调命令时 `HOME` 被透传为 Windows 路径，conda 读不到 `/root/.condarc`。解决：`export HOME=/root`。
2. **后台任务**：`nohup ... &` 在 wsl 命令退出后会被 WSL 发行版自动关闭杀掉。解决：让 wsl 进程常驻（如 PowerShell 后台任务持有），或前台执行由调用方托管。
3. **tsv 数据集**：ms-swift 本地加载不支持 `.tsv` 扩展名（报 `Couldn't reach 'tsv' on the Hub`）。转 jsonl。
4. **preference 列名**：ms-swift DPO 数据不识别裸 `rejected` 列，标准格式为 `rejected_response`（或 `rejected_messages`）。
5. **DPO 显存**：16GB 卡上 DPO 双序列 + batch2 + max_length1024 会偶发 OOM 警告（训练可继续，PyTorch 自动重分配）；建议 batch=1 或 max_length=512。
6. **GRPO 依赖**：`--deepspeed` 需要 deepspeed 包（未装且可能触发 nvcc 编译）→ 0.6B 小模型去掉即可；GRPO trainer 需要 `msgspec` 包（`pip install msgspec`）。
7. **GRPO 列映射**：`MathAccuracy` reward 强制要求数据的 `solution` 列。gsm8k 原始列是 question/answer，需 `--disable_auto_column_mapping true --columns '{"question": "query", "answer": "solution"}'`。
8. **`swift --help`** 有 KeyError 小 bug，不影响子命令（`swift sft --help` 等正常）。
9. **nvcc 冲突**：本路线全用预编译 wheel（torch/flash-attn/vllm），无需 Linux nvcc，规避了 verl 时代的多卡编译冲突。
10. **没有 `cpt` 子命令**：ms-swift 4.x CLI 只有 `pt`（预训练）；CPT 就用 `swift pt`。源码里 `SwiftPretrain` 直接继承 `SwiftSft`，`PretrainArguments` 只覆盖两个默认值：`use_chat_template=False`、`loss_scale='all'`（全 token 算 loss）——即「SFT 流水线 + 全 token loss」。因此多模态 CPT = VLM + 图文数据集 + `swift pt`，无需另找多模态专用入口。
11. **参数名是 `--tuner_type` 不是 `--train_type`**：4.7 版 argparse 里没有 train_type，写错会报 `ValueError: remaining_argv: ['--train_type', 'lora']`（脚本第一次跑就是这样失败的）。
12. **VLM 训练缺依赖链**：Qwen2-VL 需要 `qwen_vl_utils>=0.0.6` + `decord`；而 `qwen_vl_utils` 又依赖 **torchvision**（纯文本 SFT 用不到，所以之前没装）。torchvision 必须匹配 torch 2.14.1+cuda13.4 → 用 `pip install "torchvision==0.29.2+cu134" --index-url https://download.pytorch.org/whl/cu134 --no-deps`，**不要**走 conda solver（会动 pytorch，踩过坑）。
13. **Gemma4 数据占位符必须用标准 `<image>`**：预写成 `<|image|>` 会导致 image token 双计（模板 replace_tag 会再插一次），`gemma.py:309` 报 `IndexError: list index out of range` → 全样本 encode 失败。单张 512px 图 = 256 个 image token(258880)，pixel_values shape `(1, 2520, 768)`，整条样本 encode 452 token。
14. **transformers 5.19 的 Gemma4 `get_image_features().pooler_output` 是 tuple**（按图 split），旧版是单 tensor。ms-swift patch 当 tensor 用会报 `AttributeError: 'tuple' object has no attribute 'to'`。已在 `swift/model/models/gemma.py` image/video 两处补 `torch.cat(..., dim=0)`（**改的是 ms-swift 适配层，transformers 安装包未动**）。
15. **推理 generate 丢图（最隐蔽）**：transformers 5.19 的 `generate()` 会在首步把 `pixel_values` 预编码成 `mm_encoder_outputs` 再传 forward，而 ms-swift 的 Gemma4 patch 只认 `pixel_values` → 走"假图"分支，模型回答"请提供图片"（训练不受影响，因为训练直接 forward 传 pixel_values）。修复：`swift/model/models/gemma.py` 的 patch forward 增加 `mm_encoder_outputs` 参数并消费 `mm_encoder_outputs['image'/'video'].pooler_output`（torch.cat 后 masked_scatter 注入）。复现与验证：`infer/repro_gemma4_image_drop.py`。
16. **单卡必须显式 `NPROC_PER_NODE=1`**：漏了会走 DataParallel 包装，PEFT 套一层后 `model.device` 取不到参数，报 `AttributeError: 'LoraModel' object has no attribute 'device'` → `StopIteration`（`swift/template/base.py:1732` 的 `pre_forward_hook`）。四套 text/images 脚本都写了这个变量。
17. **`--max_completion_length` 不属于 `swift pt`**：它属于 RLHF，误传报 `ValueError: remaining_argv: ['--max_completion_length', '64']`。
18. **Qwen3.5 的 model_type / template 自动匹配都有歧义**：`--model_type` 报 `['qwen3_5','ovis_ocr2','qwen3_5_emb','wemm_embedding']`，`--template` 报 `['qwen3_5','qwen3_8']`，两者都必须显式传 `qwen3_5`。
19. **`text` + `images` 结构里 text 不能写 `<image>` 占位符**：图片位置由 `images` 字段决定，框架自己插 image pad；写了占位符会重复计数。
20. **`{"text":…, "images":[…]}` 确实能零 role**：实测序列解码为 `<|image>` + 256 个 image pad token + 正文 + `<eos>`，无 `<start_of_turn>`；Qwen3.5 纯文本为正文 + `<|im_end|>`，无 `<|im_start|>user`。验证脚本思路见日志：从训练日志取出 `[INPUT_IDS]` 再 `tokenizer.decode`。

## 6. 日常使用

```bash
wsl -d Ubuntu-24.04
conda activate ms-swift
cd /root/ms-swift

# 预训练 demo（PT / CPT，各 10 秒内跑完）
bash experiments/pt/pt_text_demo.sh                 # 纯文本 PT demo（Qwen3-0.6B，30 步 8s）
bash experiments/cpt/cpt_vl_demo.sh                 # 图文 CPT demo（Gemma4-E2B，messages 旧格式，8 步 10s）
bash experiments/pt/pt_text_gemma4_1gpu.sh          # 纯文本 PT（Gemma4-E2B，text 结构）
bash experiments/pt/pt_text_qwen35_1gpu.sh          # 纯文本 PT（Qwen3.5-0.8B，text 结构）
bash experiments/pt/cpt_text_images_gemma4_1gpu.sh  # 图文 CPT（Gemma4-E2B，text+images 无 role）
bash experiments/pt/cpt_text_images_qwen3vl_1gpu.sh # 图文 CPT（Qwen3-VL-2B，text+images 无 role）
python experiments/datasets/make_demo_pt_cpt.py     # 重新生成 messages 版 demo 数据（可选）
python experiments/datasets/make_demo_text_images.py # 生成 text/images 版 + 纯文本版数据（可选）

# 文本训练
bash experiments/sft/sft_2gpu.sh              # SFT（DDP）
bash experiments/rl/rl_dpo_2gpu.sh            # RL DPO
bash experiments/rl/rl_orpo_2gpu.sh           # RL ORPO（无 ref model）
bash experiments/rl/rl_simpo_2gpu.sh          # RL SimPO（无 ref model，beta 2.0）
bash experiments/rl/rl_cpo_2gpu.sh            # RL CPO（偏好项 + SFT 项）
bash experiments/rl/rl_kto_2gpu.sh            # RL KTO（二进制标签）
bash experiments/rl/rl_rm_2gpu.sh             # Reward Model（打分器，供 PPO 用）
bash experiments/rl/rl_grpo_2gpu.sh           # RL GRPO
bash experiments/sft/sft_ds_zero2_2gpu.sh     # SFT（ZeRO-2）
bash experiments/sft/sft_megatron_tp_2gpu.sh  # SFT（Megatron TP）

# 多模态 CPT
bash experiments/cpt/cpt_vl_2gpu.sh           # Qwen2-VL-2B 双卡
bash experiments/cpt/cpt_gemma4_1gpu.sh       # Gemma4-E2B 单卡
python experiments/cpt/make_mock_vl_cpt.py    # 重新生成 mock 图文集（可选）

# 推理 / 调试
python experiments/infer/infer_gemma4.py      # Gemma4 确定性验收（需 gemma.py 适配层补丁）
python experiments/infer/gemma4_hf_runner.py  # 纯 transformers，默认 1 号卡；不传参即用内置图（Gemma-1 解码层架构图）
python experiments/infer/gemma4_hf_runner.py --image <path>   # 换成别的图
python experiments/infer/gemma4_hf_runner.py --no_image       # 纯文本，不带图
python experiments/infer/gemma4_hf_simple.py  # 极简版，同样默认用内置图
python experiments/infer/repro_gemma4_image_drop.py   # 复现丢图（A/B/C 对照）
python experiments/infer/gemma4_official_demo.py      # ModelScope 官方样例；图片/文本/思考开关都改脚本头常量，无需参数
```
