import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只用 1 号卡，必须在 import torch 之前

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = "/mnt/e/models/model/gemma-4-E2B-it"
DEFAULT_IMAGE = "/mnt/e/transformers/bayon-note/Gemma/gemma-1/_arch/02_3单个_GemmaDecoderLayer_解剖.png"
IMAGE = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
Q = sys.argv[2] if len(sys.argv) > 2 else "请详细描述这张图片的内容"

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})

content = ([{"type": "image"}] if IMAGE else []) + [{"type": "text", "text": Q}]
prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

kw = {"text": prompt, "return_tensors": "pt"}
if IMAGE:
    kw["images"] = [Image.open(IMAGE).convert("RGB")]
inputs = processor(**kw).to("cuda:0")

out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
