#!/usr/bin/env python3
# 运行：python experiments/infer/repro_gemma4_image_drop.py（纯 transformers 复现丢图，A/B/C 三段对照）
"""
复现 Gemma4 在 transformers 5.19 下的"推理看不到图片"问题（只用 1 号卡）

背景（ms-swift 上的真实故障）：
  transformers 5.19 的 generate() 首步不传 pixel_values，而是把视觉特征预编码进
  mm_encoder_outputs 再交给 forward。旧版 ms-swift 的 patch forward 只认 pixel_values、
  不消费 mm_encoder_outputs → 图特征被静默丢弃 → 模型回答"请提供图片"。

本脚本用纯 transformers 复现该机制（不依赖 ms-swift），三段对照：
  阶段 A  直接 model(**inputs)（pixel_values 有值）→ 图有效（对照组，证明图本身没问题）
  阶段 B  把 forward 的 mm_encoder_outputs 强制置 None（等价旧 patch 行为）+ generate → 丢图
  阶段 C  恢复原生 forward + generate → 正常识图（修复后效果）

用法：python3 repro_gemma4_image_drop.py [图片路径]
"""
import os
import inspect
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只用 1 号卡，必须在 import torch 之前

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402

MODEL = "/mnt/e/models/model/gemma-4-E2B-it"
IMAGE = sys.argv[1] if len(sys.argv) > 1 else "/root/ms-swift/experiments/datasets/mock_vl_cpt/images/bar_00.png"
QUESTION = "请详细描述这张图片的内容"

print(f"[env] transformers={__import__('transformers').__version__}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
model.eval()

# ---- 构造输入（与正常推理一致：只放 user 轮，避免 decode 到输入里的 GT） ----
prompt = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}],
    tokenize=False, add_generation_prompt=True,
)
inputs = processor(text=prompt, images=[Image.open(IMAGE).convert("RGB")], return_tensors="pt").to("cuda:0")
input_len = inputs["input_ids"].shape[-1]
print(f"[input] tokens={input_len}, keys={sorted(inputs.keys())}", flush=True)

# ---- 定位内层 Gemma4Model，包装它的 forward ----
inner = getattr(model, "model", None)
if inner is None or not hasattr(inner, "forward"):
    raise SystemExit(f"未找到内层 model，属性列表: {[n for n, _ in model.named_children()]}")

native_forward = inner.forward
sig = inspect.signature(native_forward)
trace = {}


def _record(kw_bound):
    pv = kw_bound.arguments.get("pixel_values")
    meo = kw_bound.arguments.get("mm_encoder_outputs")
    trace["pixel_values"] = None if pv is None else tuple(pv.shape)
    trace["mm_encoder_outputs"] = None if meo is None else {k: type(v).__name__ for k, v in meo.items()}


def probing_forward(*args, **kwargs):
    """记录 generate 首次 forward 的入参，回答'图从哪一步丢的'"""
    if not trace:
        bound = sig.bind_partial(*args, **kwargs)
        _record(bound)
    return native_forward(*args, **kwargs)


def legacy_forward(*args, **kwargs):
    """模拟旧版 ms-swift patch：不认识 mm_encoder_outputs。先记录原始入参，再把它丢掉"""
    bound = sig.bind_partial(*args, **kwargs)
    if not trace:
        _record(bound)  # 记录的是丢弃【之前】的原始入参
    if "mm_encoder_outputs" in bound.arguments:
        bound.arguments["mm_encoder_outputs"] = None
    return native_forward(*bound.args, **bound.kwargs)


def decode_generate(tag):
    trace.clear()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    text = processor.batch_decode(out[:, input_len:], skip_special_tokens=True)[0].strip()
    print(f"\n[{tag}] 首步 forward 入参: pixel_values={trace.get('pixel_values')}, "
          f"mm_encoder_outputs={trace.get('mm_encoder_outputs')}", flush=True)
    print(f"[{tag}] 输出: {text[:260]}", flush=True)
    return text


# ============ 阶段 A：直接 forward，pixel_values 有值 → 图有效 ============
print("\n" + "=" * 70 + "\n[A] 直接 forward（pixel_values 存在）——证明图本身有效", flush=True)
with torch.inference_mode():
    fwd_out = model(**inputs)
print(f"[A] logits shape={tuple(fwd_out.logits.shape)}  "
      f"（另有 pixel_values 参与，说明视觉塔被调用）", flush=True)

# ============ 阶段 B：模拟旧 patch，丢 mm_encoder_outputs ============
print("\n" + "=" * 70 + "\n[B] 旧版行为：forward 忽略 mm_encoder_outputs + generate —— 复现丢图", flush=True)
inner.forward = legacy_forward
ans_b = decode_generate("B")

# ============ 阶段 C：恢复原生 forward ============
print("\n" + "=" * 70 + "\n[C] 恢复原生 forward + generate —— 修复后正常", flush=True)
inner.forward = probing_forward  # 原生行为 + 记录入参
ans_c = decode_generate("C")

# ============ 结论 ============
print("\n" + "=" * 70, flush=True)
print(f"[结论] 阶段B是否丢图: {'是 —— 复现成功' if ('图片' in ans_b or 'image' in ans_b.lower()) and len(ans_b) < 80 else '否'}", flush=True)
print(f"[结论] 阶段C是否恢复: {'是 —— 正常识图' if len(ans_c) > len(ans_b) else '否'}", flush=True)
