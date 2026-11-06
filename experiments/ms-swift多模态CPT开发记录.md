# ms-swift 多模态图文 CPT 开发记录（Qwen2-VL-2B / Gemma4-E2B）

> 环境：WSL Ubuntu-24.04 + conda env `ms-swift`（Python 3.12），2×RTX 5060 Ti 16GB（sm_120）
> 版本：torch 2.14.1 / CUDA 13.4 / flash-attn 2.8.3 / transformers 5.19.0 / ms-swift 4.7.0.dev0（源码安装）
> 模型（E 盘本地，禁止下载）：`/mnt/e/models/qwen/Qwen2-VL-2B-Instruct`、`/mnt/e/models/model/gemma-4-E2B-it`

本文记录两个多模态（图文）CPT 脚本从调研、数据准备、脚本编写到跑通的全过程，重点是每一步踩到的坑和最终的解决方式。

---

## 0. 背景：什么是 ms-swift 的 CPT

CPT（Continual Pre-Training，持续预训练）在 ms-swift 4.x 里**没有 `cpt` 子命令**。CLI 的 `ROUTE_MAPPING` 只有 `pt / sft / infer / merge-lora / web-ui / deploy / rollout / rlhf / sample / export / eval / app`。

- **CPT 实际入口是 `swift pt`**（`swift/cli/pt.py` → `pretrain_main` → `SwiftPretrain`）。
- `SwiftPretrain` 直接继承 `SwiftSft`，`PretrainArguments` 只覆盖两个默认值：
  - `use_chat_template=False`（不套对话模板）
  - `loss_scale='all'`（全部 token 参与 loss，而不是只算 assistant 部分）
- 所以"多模态 CPT"= VLM 模型 + 图文数据集 + `swift pt`，本质是 **SFT 流水线 + 全 token loss**。

这个认知直接决定了脚本形态：不需要找多模态专用入口，复用 SFT 参数体系即可。

---

## 1. 第一版：Qwen2-VL-2B 图文 CPT（已跑通并提交）

### 1.1 数据：自己 mock 的图文训练集

用户允许 mock 数据，目标是"多模态图文 CPT 训练集"。用 matplotlib 合成 8 类图：

| 类型 | 说明 |
|---|---|
| 柱状图 / 折线图 / 饼图 / 散点图 | 带数值的统计图 |
| 流程图 / 要点卡 | 结构示意 |
| 产品视图 / 区域布局 | 设计类 |

- **规模**：64 训练 + 8 验证 = 72 张 512px PNG
- **关键设计**：图上数值与文本描述**同源**（同一 random seed），即"图里画什么，描述里就写什么"，这是图文对齐训练质量的基础
- 图内标签用英文（WSL 无中文字体，中文会变豆腐块）
- 数据格式：

```json
{"messages": [
    {"role": "user", "content": "<image>请详细描述这张图片的内容，包括图表类型、数据细节和结论。"},
    {"role": "assistant", "content": "这是一张柱状图，展示了...（250~450 字）"}
  ],
 "images": ["/abs/path.png"]}
```

生成器：`experiments/make_mock_vl_cpt.py`，产物在 `experiments/datasets/mock_vl_cpt/`。

### 1.2 脚本要点（cpt_vl_2gpu.sh）

```bash
swift pt \
  --model /mnt/e/models/qwen/Qwen2-VL-2B-Instruct \
  --train_type lora \
  --dataset .../mock_vl_cpt/cpt_train.jsonl \
  --val_dataset .../mock_vl_cpt/cpt_val.jsonl \
  --max_length 2048 --batch_size 1 --gradient_accumulation_steps 4 \
  --lora_rank 8 --lora_alpha 16 --learning_rate 1e-4 \
  --max_steps 16 --warmup_ratio 0.05 \
  --torch_dtype bf16 --output_dir output/cpt-vl-2gpu
```

### 1.3 踩坑记录（Qwen 版）

**坑 1：参数名是 `--tuner_type` 不是 `--train_type`**
- 现象：`swift pt --train_type lora` 直接报 `ValueError: remaining_argv: ['--train_type', 'lora']`
- 原因：4.7 版 argparse 里只有 `--tuner_type`（沿用 SFT 参数体系）
- 解决：改名即可。

**坑 2：VLM 训练缺依赖链（最典型）**
- 现象：加载 Qwen2-VL 数据集时报错，提示缺 `qwen_vl_utils`；装上后又缺 `decord`；装完再报 torchvision 不匹配。
- 原因：`qwen_vl_utils` 依赖 torchvision；之前纯文本 SFT 用不到，所以整个链路从没装过。而 torchvision 必须和 torch 2.14.1+cuda13.4 严格匹配。
- 解决（**关键经验：不要走 conda solver**）：
  ```bash
  pip install "qwen_vl_utils>=0.0.6" decord
  pip install "torchvision==0.29.2+cu134" --index-url https://download.pytorch.org/whl/cu134 --no-deps
  ```
  如果走 `conda install` 或裸 `pip install torchvision`，solver 会**动 pytorch 版本**，导致整套 CUDA 环境崩掉（踩过，教训很深）。

### 1.4 结果

- 16/16 步，train_loss 2.86 → 1.95（均值 2.151），eval_loss 1.956，token_acc 0.44 → 0.60
- 显存 **5.24 GiB/卡**，22s 跑完
- 推理 `swift infer --adapters` 8/8 通过，饼图占比 53%/28%/16%/2% 等与 GT 高度一致
- 提交 commit `1877ee85d`（已 push）

---

## 2. 第二版：Gemma4-E2B 图文 CPT（单卡跑通 + 推理修复，已提交）

### 2.1 模型预分析

`gemma-4-E2B-it`（9.6G 单文件 safetensors）：

- `architectures: Gemma4ForConditionalGeneration`，**vision + audio 双塔**（audio hidden 1024 / 12 层）
- config 关键 token id：`image_token_id=258880`、`audio_token_id=258881`、`video_token_id=258884`、`boi_token_id=255999`、`boa_token_id=256000`
- ms-swift 内置支持：`swift/template/templates/gemma.py` 的 `Gemma4Template`（占位符 `<|image|>/<|audio|>/<|video|>`，注册名 `gemma4_nothinking` 与 `gemma4`）；`swift/model/models/gemma.py` 有 `_patch_gemma4_forward`

### 2.2 数据适配：必须用标准 `<image>` 占位符

**坑 3：预写 `<|image|>` 导致 image token 双计 → IndexError**

- 现象：训练一开始就崩，`gemma.py:309` `splited_tokens = [splited_tokens[i] for i in sorted_order]` 报 `IndexError: list index out of range`，随后全样本 encode 失败，报次生错误 `Failed to retrieve the dataset...increasing max_length`（容易被误导成 max_length 不够）。
- 我的错误操作：写了个 `make_gemma4_ds.py` 把数据里 `<image>` **预先替换成 `<|image|>`**（想当然觉得模板用 `<|image|>` 就应写成这样）。
- 根因（实验证实）：模板的 `replace_tag` 看到 `<|image|>` 会**再插一次** image token，于是 `input_ids` 里 image token 计数=2（256×2），而 processor 按 `'\n'` 切出的媒体段数=1 → 索引越界。
- 对比实验：
  - 标准 `<image>` → `n_images=1, cnt=1, segs=1, OK len=277`
  - `<|image|>` → `cnt=2` → FAIL
- 解决：删掉 `gemma4_cpt_train/val.jsonl`，脚本直接指回标准 tag 的 `cpt_train.jsonl / cpt_val.jsonl`。
- 附带教训：`--use_chat_template true` 不是根因，改它没用；真正的问题是数据 tag。

### 2.3 训练启动与适配层修复（改的是 ms-swift，不是 transformers）

**坑 4：transformers 5.19 的 pooler_output 是 tuple**

- 现象：encode 通过后，训练第一轮 forward 崩：`AttributeError: 'tuple' object has no attribute 'to'` at `swift/model/models/gemma.py:297`。
- 根因：transformers 5.19 的 `Gemma4ForConditionalGeneration.get_image_features().pooler_output` **按图 split 成 tuple**（旧版 5.6.2 是单个 tensor），ms-swift 的 patch 当 tensor 用。
- 解决：改 **ms-swift 自己的适配层代码** `/root/ms-swift/swift/model/models/gemma.py`（`_patch_gemma4_forward` 内部），image/video 两处加：
  ```python
  if isinstance(image_features, (tuple, list)):
      image_features = torch.cat(image_features, dim=0)
  ```
  （video 同理）
- 注意：**没有改 transformers 安装包**。transformers 自带的 `Gemma4Model.forward` 本来就正确处理 tuple；是 ms-swift 在加载模型时用 `_patch_gemma4_forward` 整段替换了这个 forward，被替换的那份代码写于 transformers 5.6.x 时代，接口没跟上 5.19。所以要补的是 ms-swift 这一层。改动最终同步进了 E 盘 fork 仓库。

**坑 5：双卡在 WSL 下崩（环境限制，放弃双卡）**

- 现象：`RuntimeError: CUDA driver error: device not ready`，驱动日志 `Failed to create GPU mapping`。崩溃点位于 gradient checkpointing 的 `checkpoint()` → `per_layer_projection` → peft lora → dropout。
- 尝试过：降 `max_length 1024`、`dataloader_num_workers 0`、`--dataloader_pin_memory false`，**4 次全部复现**。
- 每次崩溃后 `nvidia-smi` 恢复 13MiB 干净状态，无残留进程 → 不是显存不足（单卡仅 10.75 GiB / 16 GiB）。
- 结论：**WSL 下双卡 + 9.6G 模型的内存映射限制**（"Failed to create GPU mapping"），非显存问题。
- 决策：交付单卡版本（用户原始要求是"跑通一版多模态 CPT"，未指定卡数；Qwen 版双卡是加分项不是必选项）。

### 2.4 单卡成功参数（cpt_gemma4_1gpu.sh）

```bash
NPROC_PER_NODE=1
swift pt \
  --model /mnt/e/models/model/gemma-4-E2B-it \
  --template gemma4_nothinking \
  --train_type lora --lora_rank 8 --lora_alpha 16 \
  --dataset .../mock_vl_cpt/cpt_train.jsonl \
  --val_dataset .../mock_vl_cpt/cpt_val.jsonl \
  --max_length 1024 --batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --max_steps 16 --warmup_ratio 0.05 \
  --torch_dtype bf16 --use_chat_template true --loss_scale all \
  --freeze_vit true --freeze_aligner true \
  --dataloader_num_workers 0 --output_dir output/cpt-gemma4-1gpu
```

结果：**16/16 步，train_loss 3.685，memory 10.75 GiB，train_runtime 30.61s（2.09 samples/s）**，checkpoint 落盘。

### 2.5 推理链路排查：最隐蔽的坑（mm_encoder_outputs）

**现象**：`swift infer --adapters` 8/8 跑完，但模型回答"请提供您希望我分析的图片"——**图根本没进模型**。

排查过程（一层层剥）：

1. **离线模板 encode**（真实 val 首条）→ `len=452`，keys 含 `pixel_values(1,2520,768)`、`image_position_ids`、`mm_token_type_ids`，image token 计数 256 → **模板层正常**。
2. **`TransformersEngine(M, adapters).template.encode(...)`** → 同样 452 → 正常。
3. **`engine.infer(...)`** → 仍没图。
4. **手动 `model.generate(**collated)`** → 输出一大段"正确描述"？——**这是假象**！因为 encode 时带了 assistant 消息，我 decode 的是**输入里的 GT 文本**，不是模型生成的。剔除 assistant 重跑 → 全 8 条 prompt=287，输出全是"请提供图片"。
5. **A/B forward 实验**：同一输入换两张不同图直接 `model(**inputs)` → `logits diff = 48.25`（巨大差异）→ **forward 路径图是有效的，训练是真的多模态**。问题只出在 `generate` 路径。
6. **追踪 generate 首步 forward 入参**：patch 内层 `Gemma4Model.forward` 打印 → 第一个调用 `pixel_values=None, image_position_ids=None` → **generate 首步就丢了图**。
7. **再查 `prepare_inputs_for_generation`**：打印 kwargs → 首步传入的是 `['next_sequence_length', 'mm_token_type_ids', 'attention_mask', 'position_ids', 'mm_encoder_outputs', ...]` —— **pixel_values 在进入 generate 时已被 transformers 5.19 预编码成 `mm_encoder_outputs`**。

**根因**：transformers 5.19 的 `generate()` 会在首步把 `pixel_values` 通过视觉塔预编码成 `mm_encoder_outputs`（dict，key 为 `'image'/'video'`，value 为 `BaseModelOutputWithPooling`），再传给 forward。而 **ms-swift 的 `_patch_gemma4_forward` 只认 `pixel_values`，完全忽略 `mm_encoder_outputs`** → 图像特征被静默丢弃 → 模型只看到文本。

（训练不受影响，因为训练直接 `model(**inputs)` 走 pixel_values 分支。）

**解决**：给 `swift/model/models/gemma.py` 的 patch forward（同样的 ms-swift 适配层，transformers 未动）增加 `mm_encoder_outputs` 参数并消费：

```python
mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
if pixel_values is None and mm_encoder_outputs.get('image') is not None:
    image_features = torch.cat(mm_encoder_outputs['image'].pooler_output, dim=0)
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    # ... 与 pixel_values 分支相同的 masked_scatter 注入逻辑
# video 同理：mm_encoder_outputs.get('video')
```

修复后：
- 自定义推理脚本 `experiments/infer_gemma4.py`（encode → collate → generate）8/8 全过：柱状图数值、折线图趋势、饼图占比、散点图关系、流程图步骤、要点卡内容、草图布局全部与图像一致。
- `swift infer` CLI 链路同样恢复正常（同一条路径）。

**为什么自定义脚本而不修 engine**：`TransformersEngine.infer` 的丢图是同一根因（generate 首步 mm_encoder_outputs），修好 ms-swift 适配层 patch 后 engine 路径也通；自定义脚本作为确定性复现和验收手段保留。

### 2.6 修复边界：改的是 ms-swift 适配层，不是 transformers

这一点容易被误解，单独说明：

- **transformers 安装包一行未改**。验证方式是对比文件修改时间：

  ```
  /opt/miniconda3/envs/ms-swift/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py
    → 2026-10-18 20:55（pip 安装时刻，之后从未变动）
  /root/ms-swift/swift/model/models/gemma.py
    → 2026-11-05 23:27（本次修改）
  ```

- **为什么改 ms-swift 就够**：transformers 5.19 原生的 `Gemma4Model.forward` 本身已正确处理新接口（源码里就是 `mm_encoder_outputs.get("image")` → `torch.cat(pooler_output)` → `masked_scatter`）。问题在于 ms-swift 加载模型时用 `_patch_gemma4_forward` **整段替换**了该 forward，替换代码停留在 5.6.x 写法。源头（transformers）是对的，坏的是被替换进去的那份副本。
- **为什么这样更稳**：把补丁放在 ms-swift 层，`pip install -U transformers` / 重装环境都不会静默冲掉它。反之直接改 `site-packages` 下的文件，下次升级 transformers 就被覆盖，且难以追溯。
- **代价**：这是本地 fork 的改动，ms-swift 上游升级时需要重新确认 `gemma.py` 的补丁是否已被官方修复，避免重复打补丁。已记录在 `experiments/README.md` 坑 14/15。

---

## 3. 经验总结

1. **CPT = `swift pt`**：别找 `cpt` 命令，本质是 SFT 流水线 + `loss_scale=all` 全 token loss。
2. **占位符用模板默认 tag**：数据里写什么 tag，要对照模板 `replace_tag` 的行为，写错会静默双计或索引越界；先做单条 encode 冒烟（数 image token 数）再跑全量。
3. **transformers 大版本升级会改多模态接口**：5.19 把 `pooler_output` 变 tuple、`generate` 引入 `mm_encoder_outputs`。ms-swift 用 monkey-patch 覆盖了模型 forward，patch 没跟上新接口时**不报错、只静默丢图**（训练正常、推理退化）。修复要补在 ms-swift 适配层，**不要去改 transformers 安装包**（下次 pip 升级就会被覆盖）。
4. **推理丢图先做 A/B 实验**：同一输入两张不同图对比 logits，能秒级区分"模型没能力"和"图没进模型"两个完全不同的方向。
5. **手动 generate 的"正确输出"可能是 decode 了输入 GT**：encode 时务必剔除 assistant 消息再推理。
6. **WSL 多卡大模型有限制**：9.6G 模型双卡在 WSL 下 `Failed to create GPU mapping`，与显存无关；单卡可稳定跑。
7. **安装依赖绝不走 conda solver**：torchvision 等 CUDA 包必须 `pip install --index-url ... --no-deps`，否则 solver 会动 pytorch 导致全崩。

---

## 4. 产物与提交

| 文件 | 说明 |
|---|---|
| `experiments/make_mock_vl_cpt.py` | mock 图文集生成器（8 类图，64+8=72 张，同 seed 数值对齐） |
| `experiments/cpt_vl_2gpu.sh` | Qwen2-VL-2B 双卡 CPT（已跑通） |
| `experiments/cpt_vl_infer.sh` | Qwen 版 adapter 推理 |
| `experiments/cpt_gemma4_1gpu.sh` | Gemma4-E2B 单卡 CPT（已跑通） |
| `experiments/cpt_gemma4_2gpu.sh` | Gemma4 双卡脚本（WSL 限制，未跑通，留档） |
| `experiments/cpt_gemma4_infer.sh` | Gemma4 CLI 推理（ms-swift 适配层补丁修复后可用） |
| `experiments/infer_gemma4.py` | Gemma4 确定性推理脚本（8/8 验收） |
| `swift/model/models/gemma.py` | **ms-swift 适配层补丁**（transformers 未改）：pooler tuple + mm_encoder_outputs（坑 4/5 的修复） |
| `experiments/README.md` | 结果表 + 踩坑 13/14/15 |

提交记录：
- `1877ee85d` — Qwen2-VL-2B 图文 CPT（80 files）
- `db2991cc7` — Gemma4-E2B 图文 CPT + ms-swift gemma.py 适配补丁（7 files, +212/-51）
