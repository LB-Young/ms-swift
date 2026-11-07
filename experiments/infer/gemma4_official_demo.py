#!/usr/bin/env python3
# 用途：按 ModelScope 官方模型卡（google/gemma-4-E4B-it）的样例，用 Gemma4 处理多模态（图文）任务
# 来源：https://modelscope.cn/models/google/gemma-4-E4B-it  ->  Getting Started / Code for processing Images
# 差异：官方样例在线加载 E4B；本机只有 gemma-4-E2B-it，默认加载本地权重，调用方式与官方逐行一致
# 运行：cd /root/ms-swift && python experiments/infer/gemma4_official_demo.py
"""
Gemma4 官方样例复现（modelscope 版 API）

官方卡片给出的三条核心调用，本脚本原样照搬：
  1) from modelscope import AutoProcessor, AutoModelForMultimodalLM
  2) model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
     inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                            return_tensors="pt", add_generation_prompt=True)
  3) outputs = model.generate(**inputs, max_new_tokens=...)
     response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
     processor.parse_response(response, prefix=inputs["input_ids"])

输入格式（官方）：
  {"type": "image", "url": <图片地址>} 放在 {"type": "text", ...} 之前
  （卡片 Best Practices 第 4 条：图像内容放在文本之前）

所有参数都写在下面的「运行配置」区，直接改常量即可，运行时不需要任何命令行参数。
"""
import os
import time
import urllib.request

# ============================ 运行配置（改这里） ============================
GPU = "1"                                   # 使用 1 号卡；必须在 import torch 之前生效
MODEL = "/mnt/e/models/model/gemma-4-E2B-it"  # 本地权重；官方是 google/gemma-4-E4B-it
MODE = "image"                              # "image" 图文 / "text" 纯文本
IMAGE = ("/mnt/e/transformers/bayon-note/Gemma/gemma-1/_arch/"
         "02_3单个_GemmaDecoderLayer_解剖.png")   # 本地图或 http(s) URL
MAX_NEW_TOKENS = 1024                       # 输出上限；官方样例是 512，调小更快
THINKING = False                            # True 打开思考链（enable_thinking=True）
# 想跑官方 GoldenGate 样例，把 IMAGE 换成下面这张图即可：
# IMAGE = ("https://raw.githubusercontent.com/google-gemma/cookbook/"
#          "refs/heads/main/apps/sample-data/GoldenGate.png")
# ==========================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = GPU

from modelscope import AutoProcessor, AutoModelForMultimodalLM  # noqa: E402  官方样例的导入路径


def fetch_image(url_or_path):
    """图片是 URL 时先落到本地再交给 processor，便于离线复跑。"""
    if url_or_path.startswith(("http://", "https://")):
        fn = os.path.join("/root/tmp", url_or_path.rsplit("/", 1)[-1])
        if not os.path.exists(fn):
            req = urllib.request.Request(url_or_path, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                with open(fn, "wb") as f:
                    f.write(r.read())
        return fn
    return url_or_path


def build_messages():
    if MODE == "text":
        # 官方纯文本样例
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short joke about saving RAM."},
        ]
    # 官方图片样例：图片在文本之前
    return [
        {"role": "user", "content": [
            {"type": "image", "url": fetch_image(IMAGE)},
            {"type": "text", "text": "What is shown in this image?"},
        ]}
    ]


def main():
    import transformers
    print(f"[env] transformers={transformers.__version__} CUDA_VISIBLE_DEVICES={GPU}", flush=True)
    print(f"[cfg] mode={MODE} max_new_tokens={MAX_NEW_TOKENS} thinking={THINKING}", flush=True)
    print(f"[cfg] image={IMAGE if MODE == 'image' else '-'}", flush=True)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL)
    print(f"[load] processor={type(processor).__name__}", flush=True)
    try:
        # 官方写法：dtype="auto" + device_map="auto"
        model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype="auto", device_map="auto")
    except Exception as e:  # 没有 accelerate 时退化为单卡直放
        print(f"[load] device_map=auto 失败({type(e).__name__}), 回退 .to('cuda'): {e}", flush=True)
        model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype="auto").to("cuda")
    print(f"[load] {type(model).__name__} in {time.time() - t0:.1f}s | device={model.device}", flush=True)

    kw = {"enable_thinking": True} if THINKING else (
        {"enable_thinking": False} if MODE == "text" else {})
    inputs = processor.apply_chat_template(
        build_messages(), tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, **kw).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"[prompt] {input_len} tokens | keys={list(inputs.keys())}", flush=True)

    t1 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    new_tokens = outputs.shape[-1] - input_len
    print(f"[generate] {new_tokens} new tokens in {time.time() - t1:.1f}s", flush=True)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    print("\n===== raw response =====")
    print(response.strip())

    parsed = processor.parse_response(response, prefix=inputs["input_ids"])
    print("\n===== parse_response =====")
    print(repr(parsed))


if __name__ == "__main__":
    main()
