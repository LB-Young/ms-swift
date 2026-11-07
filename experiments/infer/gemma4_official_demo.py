#!/usr/bin/env python3
# 用途：按 ModelScope 官方模型卡（google/gemma-4-E4B-it）的样例，用 Gemma4 处理多模态（图文）任务
# 来源：https://modelscope.cn/models/google/gemma-4-E4B-it  ->  Getting Started / Code for processing Images
# 差异：官方样例在线加载 E4B；本机只有 gemma-4-E2B-it，默认加载本地权重，调用方式与官方逐行一致
# 运行：cd /root/ms-swift && python experiments/infer/gemma4_official_demo.py
#       cd /root/ms-swift && python experiments/infer/gemma4_official_demo.py --mode text
"""
Gemma4 官方样例复现（modelscope 版 API）

官方卡片给出的三条核心调用，本脚本原样照搬：
  1) from modelscope import AutoProcessor, AutoModelForMultimodalLM
  2) model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
     inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                            return_tensors="pt", add_generation_prompt=True)
  3) outputs = model.generate(**inputs, max_new_tokens=512)
     response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
     processor.parse_response(response, prefix=inputs["input_ids"])

输入格式（官方）：
  {"type": "image", "url": <图片地址>} 放在 {"type": "text", ...} 之前
  （卡片 Best Practices 第 4 条：图像内容放在文本之前）

本机适配说明：
  - 默认 1 号卡（CUDA_VISIBLE_DEVICES=1，在 import modelscope/torch 之前设置）
  - E4B 权重未下载，默认用本地 gemma-4-E2B-it；可用 --model 指向任意 Gemma4 目录
  - 官方 image url 会先下载到 /root/tmp 再交给 processor，避免每次重复走网络
  - --thinking 打开推理模式（官方：system prompt 首加 <|think|>，模板参数 enable_thinking=True）

用法：
  python gemma4_official_demo.py                     # 官方 GoldenGate.png 图文样例
  python gemma4_official_demo.py --image 本地图片     # 换成本地图片
  python gemma4_official_demo.py --mode text         # 官方纯文本样例
  python gemma4_official_demo.py --thinking          # 带思考链
"""
import argparse
import os
import sys
import time
import urllib.request


def _prequeue_gpu():
    """在 import torch 之前确定用哪张卡：--gpu 优先，默认 1 号卡。"""
    if "--gpu" in sys.argv:
        i = sys.argv.index("--gpu")
        if i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


_prequeue_gpu()

from modelscope import AutoProcessor, AutoModelForMultimodalLM  # noqa: E402  官方样例的导入路径

MODEL = "/mnt/e/models/model/gemma-4-E2B-it"
IMG_URL = ("https://raw.githubusercontent.com/google-gemma/cookbook/"
           "refs/heads/main/apps/sample-data/GoldenGate.png")


def fetch_image(url_or_path):
    """官方样例直接给 URL；这里先落到本地再传给 processor，便于离线复跑。"""
    if url_or_path.startswith(("http://", "https://")):
        fn = os.path.join("/root/tmp", url_or_path.rsplit("/", 1)[-1])
        if not os.path.exists(fn):
            req = urllib.request.Request(url_or_path, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                with open(fn, "wb") as f:
                    f.write(r.read())
        return fn
    return url_or_path


def build_messages(mode, image):
    if mode == "text":
        # 官方纯文本样例
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short joke about saving RAM."},
        ]
    # 官方图片样例：图片在文本之前
    return [
        {"role": "user", "content": [
            {"type": "image", "url": fetch_image(image)},
            {"type": "text", "text": "What is shown in this image?"},
        ]}
    ]


def main():
    ap = argparse.ArgumentParser(description="Gemma4 official model-card demo")
    ap.add_argument("--model", default=MODEL, help="模型目录或 repo id")
    ap.add_argument("--mode", choices=("image", "text"), default="image")
    ap.add_argument("--image", default=IMG_URL, help="图片 URL 或本地路径")
    ap.add_argument("--gpu", default="1", help="使用的卡号，默认 1")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--thinking", action="store_true", help="开启 thinking 模式")
    args = ap.parse_args()

    import transformers
    print(f"[env] transformers={transformers.__version__} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"[load] processor={type(processor).__name__}", flush=True)
    try:
        # 官方写法：dtype="auto" + device_map="auto"
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model, dtype="auto", device_map="auto")
    except Exception as e:  # 没有 accelerate 时退化为单卡直放
        print(f"[load] device_map=auto 失败({type(e).__name__}), 回退 .to('cuda'): {e}", flush=True)
        model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype="auto").to("cuda")
    print(f"[load] {type(model).__name__} in {time.time() - t0:.1f}s | device={model.device}", flush=True)

    messages = build_messages(args.mode, args.image)
    kw = {"enable_thinking": True} if args.thinking else (
        {"enable_thinking": False} if args.mode == "text" else {})
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, **kw).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"[prompt] {input_len} tokens | keys={list(inputs.keys())}", flush=True)

    t1 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(f"[generate] {outputs.shape[-1] - input_len} new tokens in {time.time() - t1:.1f}s", flush=True)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    print("\n===== raw response =====")
    print(response.strip())

    parsed = processor.parse_response(response, prefix=inputs["input_ids"])
    print("\n===== parse_response =====")
    print(repr(parsed))


if __name__ == "__main__":
    main()
