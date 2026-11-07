#!/usr/bin/env python3
"""
Gemma4-E2B 纯 transformers 推理脚本（不依赖 ms-swift）

- 默认使用 **1 号卡**（CUDA_VISIBLE_DEVICES=1，进程完全看不到 0 号卡）
- 类封装：Gemma4Runner.load() 加载模型，Gemma4Runner.chat() 图文对话
- 走官方 processor.apply_chat_template，图片用 {"type": "image"} 由模板自动插入特殊 token，
  不手工写 <|image|> / <start_of_image>，避免占位符写错（这是 ms-swift 版踩过的坑）

用法：
  python3 gemma4_hf_runner.py                            # 默认用内置图片（Gemma-1 解码层架构图）
  python3 gemma4_hf_runner.py --image /path/to.png       # 换成别的图
  python3 gemma4_hf_runner.py --no_image                 # 不带图，纯文本测试
"""
import os
import sys
import time
import argparse


def _early_gpu_index(default: int = 1) -> int:
    """在 import torch 之前从 sys.argv 里提前解析 --gpu，保证 CUDA_VISIBLE_DEVICES 先生效。"""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--gpu="):
            return int(a.split("=", 1)[1])
    return default


GPU_INDEX = _early_gpu_index(1)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)  # 必须在 import torch 之前

# 内置默认测试图：Gemma-1 笔记里的单层 DecoderLayer 架构图（不传 --image 时直接用这张）
DEFAULT_IMAGE = "/mnt/e/transformers/bayon-note/Gemma/gemma-1/_arch/02_3单个_GemmaDecoderLayer_解剖.png"

import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image  # noqa: E402


class Gemma4Runner:
    """Gemma4 图文推理封装（transformers 原生 Auto 类）。"""

    def __init__(self, model_path: str, dtype=torch.bfloat16, verbose: bool = True):
        self.model_path = model_path
        self.verbose = verbose
        # CUDA_VISIBLE_DEVICES 已把目标卡重映射为 cuda:0
        self.device = "cuda:0"
        self.processor = self._load_processor()
        self.model = self._load_model(dtype)

    # ---------- 加载 ----------
    def _load_processor(self):
        t0 = time.time()
        processor = transformers.AutoProcessor.from_pretrained(self.model_path)
        if self.verbose:
            print(f"[load] processor: {type(processor).__name__}  ({time.time() - t0:.1f}s)", flush=True)
        return processor

    def _load_model(self, dtype):
        t0 = time.time()
        kwargs = dict(device_map={"": self.device}, low_cpu_mem_usage=True, dtype=dtype)
        # transformers 5.x 主打 AutoModelForImageTextToText；保留兜底以防模型类型未注册
        candidates = ("AutoModelForImageTextToText", "AutoModelForConditionalGeneration", "AutoModel")
        last_err = None
        for name in candidates:
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(self.model_path, **kwargs)
                if self.verbose:
                    print(f"[load] model: {name} -> {type(model).__name__}  ({time.time() - t0:.1f}s)", flush=True)
                    print(f"[load] device={next(model.parameters()).device}, dtype={next(model.parameters()).dtype}", flush=True)
                return model
            except Exception as e:  # 换下一个 Auto 类再试
                last_err = e
                if self.verbose:
                    print(f"[load] {name} failed: {type(e).__name__}: {e}", flush=True)
        raise RuntimeError(f"所有 Auto 类加载失败: {last_err}")

    # ---------- 推理 ----------
    def chat(self, image_path: str | None = None, question: str = "请描述这张图片。",
             max_new_tokens: int = 512, do_sample: bool = False) -> str:
        content = []
        if image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs_kwargs = {"text": prompt, "return_tensors": "pt"}
        if image_path:
            img = Image.open(image_path).convert("RGB")
            inputs_kwargs["images"] = [img]

        inputs = self.processor(**inputs_kwargs).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        if self.verbose:
            print(f"[chat] prompt tokens={input_len}, keys={sorted(inputs.keys())}", flush=True)

        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if not do_sample:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None
        t0 = time.time()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        dt = time.time() - t0
        new_tokens = output[:, input_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        if self.verbose:
            print(f"[chat] generate {new_tokens.shape[-1]} tokens in {dt:.1f}s", flush=True)
        return text

    def __repr__(self):
        return f"Gemma4Runner(model={self.model_path}, device={self.device}, gpu_index={GPU_INDEX})"


def main():
    ap = argparse.ArgumentParser(description="Gemma4-E2B transformers 推理（默认 1 号卡）")
    ap.add_argument("--model", default="/mnt/e/models/model/gemma-4-E2B-it")
    ap.add_argument("--gpu", type=int, default=1, help="可见 GPU 序号（默认 1，即只用 1 号卡）")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="测试图片路径（默认内置 Gemma-1 解码层架构图）")
    ap.add_argument("--no_image", action="store_true", help="纯文本模式，不传图片")
    ap.add_argument("--question", default="请详细描述这张图片的内容，包括类型、数据和结论。")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    print(f"=== Gemma4 HF runner | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} ===", flush=True)
    print(f"torch {torch.__version__}, transformers {transformers.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"visible devices: {torch.cuda.device_count()} -> {torch.cuda.get_device_name(0)}", flush=True)

    runner = Gemma4Runner(args.model, verbose=True)
    img = None if args.no_image else args.image
    print(f"[image] {'disabled (pure text)' if args.no_image else args.image}", flush=True)
    t_all = time.time()
    answer = runner.chat(img, args.question, max_new_tokens=args.max_new_tokens)
    print("=" * 60, flush=True)
    print(f"[Q] {args.question}", flush=True)
    print(f"[A] {answer}", flush=True)
    print("=" * 60, flush=True)
    print(f"[done] inference {time.time() - t_all:.1f}s (model already loaded)", flush=True)


if __name__ == "__main__":
    main()
