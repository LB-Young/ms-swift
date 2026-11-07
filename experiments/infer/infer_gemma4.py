#!/usr/bin/env python
"""Gemma4 CPT adapter inference over the mock VL val set.

Why not `swift infer --val_dataset`?
  ms-swift 4.7's infer engine drops the `images` field for this Gemma4 setup,
  so the model only sees text and answers "please provide an image".
  This script bypasses the engine: template.encode -> data_collator -> model.generate,
  which reproduces the training-time image path exactly.
"""
import json
import os
import torch

os.environ.setdefault('HOME', '/root')

from swift.infer_engine import TransformersEngine
from swift.utils import to_device

M = '/mnt/e/models/model/gemma-4-E2B-it'
BASE = '/root/ms-swift/experiments'
CKPT = os.path.join(BASE, 'output/cpt-gemma4-1gpu/v0-20261105-222437/checkpoint-16')
VAL = os.path.join(BASE, 'datasets/mock_vl_cpt/cpt_val.jsonl')
OUT = os.path.join(BASE, 'output/cpt-gemma4-1gpu/infer_results.jsonl')

eng = TransformersEngine(M, adapters=[CKPT], max_batch_size=1,
                         torch_dtype=torch.bfloat16, model_type='gemma4')
tmpl, model = eng.template, eng.model
model.eval()

rows = [json.loads(l) for l in open(VAL, encoding='utf-8')]
results = []
for i, row in enumerate(rows):
    # inference must NOT include the assistant answer
    prompt_msgs = [m for m in row['messages'] if m['role'] != 'assistant']
    enc = tmpl.encode({'messages': prompt_msgs, 'images': row['images']})
    inputs = to_device(tmpl.data_collator([enc]), model.device)
    n_prompt = inputs['input_ids'].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    response = tmpl.tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
    labels = row['messages'][-1]['content']
    rec = {'response': response, 'labels': labels,
           'messages': row['messages'], 'images': row['images']}
    results.append(rec)
    print(f'[{i + 1}/{len(rows)}] prompt={n_prompt} resp={len(response)}')
    print('  resp:', response[:110].replace('\n', ' '))

with open(OUT, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('saved', len(results), '->', OUT)
