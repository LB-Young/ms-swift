#!/usr/bin/env python3
# 用途：生成 PT（纯文本预训练）与 CPT（多模态图文继续预训练）的演示数据集
# 输出：experiments/datasets/demo_pt_cpt/{pt_train,pt_val,cpt_train,cpt_val}.jsonl + images/
# 说明：CPT 的图文条目从 mock_vl_cpt 里按图挑选并复制，答案与图内容一致，可直接训练
# 运行：cd /root/ms-swift && python experiments/datasets/make_demo_pt_cpt.py
"""生成两类 demo 数据，格式分别是：
  PT  -> {"text": "..."}                                   纯文本长语料，无 role，不套模板
  CPT -> {"messages": [...], "images": ["/abs/x.png"]}     图文对，content 里用 <image> 占位
"""
import json
import os
import shutil

BASE = "/root/ms-swift/experiments/datasets"
SRC = os.path.join(BASE, "mock_vl_cpt")
OUT = os.path.join(BASE, "demo_pt_cpt")
IMG_OUT = os.path.join(OUT, "images")

# PT 语料：10 条训练 + 2 条验证，内容取自本项目真实经验（架构要点 / 训练踩坑）
PT_TEXT = [
    "Gemma 4 的 E2B 与 E4B 是同一代的两档模型。E2B 的有效参数量约 2.3B，"
    "E4B 的有效参数量约 4.5B，如果把嵌入层也算进来，E4B 的权重量级接近 8B。"
    "两者共用一个 262K 的词表，结构上都是 42 层、滑动窗口 512、上下文 128K，"
    "并且同时支持文本、图像和音频三种模态。视觉编码器只有约 150M，音频编码器约 300M，"
    "真正的容量集中在语言主干上。",

    "滑动窗口注意力是 Gemma 系列省显存的关键设计。窗口大小 512 意味着每个 token 只对"
    "前 512 个 token 做注意力，超出窗口的历史被丢弃，因此长序列的注意力开销是线性的"
    "而不是平方的。代价是超长距离的依赖要靠多层堆叠间接传递。训练时如果把 max_length "
    "设到 1024 而窗口只有 512，后 512 个 token 看不到最前面的内容，这一点在评估长文档"
    "任务时容易误判。",

    "ms-swift 里 PT 和 CPT 共用 swift pt 这个入口。命令行只有 pt 预训练，没有单独的 cpt "
    "子命令，因为 SwiftPretrain 直接继承 SwiftSft，PretrainArguments 只覆盖两个默认值："
    "use_chat_template 设为 false，loss_scale 设为 all。也就是说，预训练就是所谓 SFT 流水线"
    "加上全 token 计算 loss，多模态 CPT 只要换成视觉语言模型加图文数据集即可，不需要另找入口。",

    "loss_scale 决定哪些 token 参与损失计算。all 表示整条序列每个 token 都算，适合预训练和"
    "继续预训练，因为目标就是让模型学会语料本身的分布；assistant 表示只在助手回复上算，"
    "适合指令微调，避免把用户提问也当成学习目标。做 CPT 时如果用错成 assistant，模型只学到"
    "答案而不学语料风格，继续预训练的效果会明显打折。",

    "多模态数据的占位符必须写成标准的 <image>。如果预先把图片位置写成 <|image|>，模板在套用"
    "时会再插入一次图像标记，导致 image token 被双计，编码阶段直接抛出列表越界错误。"
    "一张 512 像素的图大约对应 256 个图像 token，加上文字后整条样本约有 450 个 token，"
    "像素张量的形状是 1 乘 2520 乘 768。",

    "transformers 5.19 的 generate 会在第一步把像素输入预先编码成 mm_encoder_outputs，"
    "再交给 forward。任何自己重写 forward 的适配层如果只认 pixel_values 参数，就会在这一步"
    "拿不到图像特征，走进没有图的假分支，模型于是回答请提供图片。训练路径不受影响，"
    "因为训练是直接调用 forward 并传入 pixel_values 的。",

    "在 WSL 里用双卡跑视觉语言模型训练时，可能遇到 CUDA driver error: device not ready，"
    "日志里伴随 Failed to create GPU mapping。降低 max_length、把 dataloader 工作进程设为 0、"
    "关闭 pin_memory 都无效，说明问题出在 WSL 的 GPU 映射而不是数据管道。单卡可以正常跑通，"
    "双卡目前只能绕开。",

    "LoRA 微调视觉语言模型时，常见做法是冻结视觉编码器和对齐层，只训练语言侧的适配器。"
    "冻结 ViT 能显著降低显存，因为视觉塔的前向不需要保存梯度；冻结对齐层则保证图像特征"
    "的投影不被少量数据带偏。参数量方面，秩 8、alpha 16 的配置在 2B 级模型上大约只增加"
    "几百万可训练参数。",

    "torchvision 是视觉语言模型训练的隐性依赖。纯文本微调不会用到它，所以环境里常常没装；"
    "一旦切到多模态数据，图像预处理就会导入 torchvision 并报模块缺失。安装时必须匹配"
    "torch 的 CUDA 版本，用预编译轮子并加 no-deps，避免包管理器顺带把 pytorch 换掉。",

    "评估继续预训练是否有效，不能只看训练损失。更可靠的做法是固定一组下游问题做对比："
    "同一批图片、同一批提问，比较训练前后的回答是否更贴合图像细节。如果损失降了但回答质量"
    "没有变化，通常是学习率过大导致适配器过拟合，或者数据里图文对本身不匹配。",

    "把训练脚本按用途分成 sft、rl、cpt、infer 几个子目录，比全部平铺在一个目录里更容易"
    "维护。脚本内部的输出路径用相对仓库根的写法，执行前先切换到仓库根目录，这样换机器或"
    "换目录都不会断链。每个脚本开头写清用途、要点、输出和运行方式，接手的人不用翻代码。",

    "显存不足时的排查顺序：先看 batch size 和序列长度，再看是否冻结了视觉塔，然后看是否"
    "开启了梯度检查点，最后才考虑换更小的模型或用量化。很多看起来像显存不足的问题，"
    "实际是某一步把整条长序列和全部中间激活都保留了下来。",
]

# CPT 用图：从 mock_vl_cpt 里挑 10 张（8 训练 + 2 验证），答案沿用与图匹配的原始条目
PICKS_TRAIN = ["bar_00.png", "bar_08.png", "pie_02.png", "pie_10.png",
               "line_01.png", "line_09.png", "schematic_04.png", "scatter_03.png"]
PICKS_VAL = ["map_07.png", "product_06.png"]


def load_src_rows():
    rows = []
    for name in ("cpt_train.jsonl", "cpt_val.jsonl"):
        path = os.path.join(SRC, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def pick_row(rows, pic):
    """挑出图片文件名匹配的那一条，返回改写好绝对路径的副本。"""
    for r in rows:
        imgs = r.get("images") or []
        if imgs and imgs[0].endswith(pic):
            rr = dict(r)
            rr["images"] = [os.path.join(IMG_OUT, pic)]
            return rr
    return None


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    os.makedirs(IMG_OUT, exist_ok=True)

    # 1) 复制图片，保证 demo 数据集自带素材
    for pic in PICKS_TRAIN + PICKS_VAL:
        src = os.path.join(SRC, "images", pic)
        if not os.path.exists(src):
            raise SystemExit(f"缺图：{src}")
        shutil.copy2(src, os.path.join(IMG_OUT, pic))

    # 2) PT：纯文本
    write_jsonl(os.path.join(OUT, "pt_train.jsonl"),
                [{"text": t} for t in PT_TEXT[:10]])
    write_jsonl(os.path.join(OUT, "pt_val.jsonl"),
                [{"text": t} for t in PT_TEXT[10:]])

    # 3) CPT：图文，答案与图一致
    rows = load_src_rows()
    cpt_train, cpt_val = [], []
    for pic in PICKS_TRAIN:
        r = pick_row(rows, pic)
        if r is None:
            raise SystemExit(f"mock_vl_cpt 里找不到 {pic} 对应条目")
        cpt_train.append(r)
    for pic in PICKS_VAL:
        r = pick_row(rows, pic)
        if r is None:
            raise SystemExit(f"mock_vl_cpt 里找不到 {pic} 对应条目")
        cpt_val.append(r)

    write_jsonl(os.path.join(OUT, "cpt_train.jsonl"), cpt_train)
    write_jsonl(os.path.join(OUT, "cpt_val.jsonl"), cpt_val)

    print(f"[done] {OUT}")
    for name in ("pt_train.jsonl", "pt_val.jsonl", "cpt_train.jsonl", "cpt_val.jsonl"):
        p = os.path.join(OUT, name)
        n = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  {name}: {n} 条, {os.path.getsize(p)} 字节")
    print(f"  images/: {len(os.listdir(IMG_OUT))} 张")


if __name__ == "__main__":
    main()
