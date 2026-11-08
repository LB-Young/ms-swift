#!/usr/bin/env python3
"""生成 text/images 结构（无 role）的 CPT demo 数据 + 纯文本 PT 数据。

图文版：每行 {"text": "图片描述", "images": ["/abs/x.png"]}，text 里不带 <image> 占位符，
        图片由 images 字段单独给出，配合 swift pt --use_chat_template false 使用。
纯文本版：每行 {"text": "语料"}，直接从上一级 pt_{train,val}.jsonl 复制。

运行：/opt/miniconda3/envs/ms-swift/bin/python /root/ms-swift/experiments/datasets/make_demo_text_images.py
"""
import json
import os
import shutil

BASE = '/root/ms-swift/experiments/datasets/demo_pt_cpt'
OUT = os.path.join(BASE, 'structured')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    os.makedirs(OUT, exist_ok=True)

    # 图文：把原 messages 结构里的 assistant 正文抽出来当 text，丢掉 role 外壳
    for split in ('train', 'val'):
        src = read_jsonl(os.path.join(BASE, f'cpt_{split}.jsonl'))
        rows = []
        for item in src:
            caption = item['messages'][-1]['content']
            assert '<image>' not in caption, 'caption 里不能带 <image> 占位符'
            rows.append({'text': caption, 'images': item['images']})
        dst = os.path.join(OUT, f'cpt_text_images_{split}.jsonl')
        write_jsonl(dst, rows)
        print(f'cpt_text_images_{split}.jsonl: {len(rows)} 条, {os.path.getsize(dst)} 字节')

    # 纯文本：直接复用已有 PT 语料
    for split in ('train', 'val'):
        dst = os.path.join(OUT, f'pt_text_{split}.jsonl')
        shutil.copy(os.path.join(BASE, f'pt_{split}.jsonl'), dst)
        print(f'pt_text_{split}.jsonl: {len(read_jsonl(dst))} 条, {os.path.getsize(dst)} 字节')

    print(f'[done] {OUT}')


if __name__ == '__main__':
    main()
