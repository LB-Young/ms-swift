#!/usr/bin/env python3
# 运行：python experiments/cpt/make_mock_vl_cpt.py（输出 experiments/datasets/mock_vl_cpt/，可重复执行）
"""生成 mock 多图文 CPT 数据集（图文对，长文本描述）。

输出目录: /root/ms-swift/experiments/datasets/mock_vl_cpt/
  images/         64+8 张 512x512 png
  cpt_train.jsonl 64 条训练样本
  cpt_val.jsonl   8 条验证样本

数据格式（ms-swift 多模态 messages 格式）:
  {"messages": [{"role":"user","content":"<image>..."},
                {"role":"assistant","content":"..."}],
   "images": ["/abs/path.png"]}
"""
import json
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = '/root/ms-swift/experiments/datasets/mock_vl_cpt'
IMG_DIR = os.path.join(OUT, 'images')
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#B279A2']

USER_PROMPT = '<image>请详细描述这张图片的内容、图中呈现的数据，并给出你从中能得出的结论。'


def gen_bar(ax, rng):
    cats = ['Product A', 'Product B', 'Product C', 'Product D']
    vals = rng.integers(30, 120, 4)
    bars = ax.bar(cats, vals, color=COLORS[:4])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha='center', fontsize=9)
    ax.set_title('Quarterly Sales by Product', fontsize=11)
    ax.set_ylabel('Units (k)')
    return ('这是一张柱状图，展示了四个产品在某一季度的销售情况。横轴为产品名称，依次为 Product A、'
            'Product B、Product C、Product D，纵轴为销量，单位为千件。其中 Product A 销量为 %d 千件，'
            'Product B 为 %d 千件，Product C 为 %d 千件，Product D 为 %d 千件。从图中可以看出，'
            '%s 的销量最高，%s 的销量最低，两者差距约为 %d 千件。整体来看，四个产品的销量分布%s，'
            '说明各产品之间的市场需求存在明显差异。如果后续需要做资源倾斜，建议优先保障销量最高的产品线，'
            '同时分析销量最低产品的原因，判断是定价、渠道还是产品力问题。'
            % (vals[0], vals[1], vals[2], vals[3],
               cats[int(vals.argmax())], cats[int(vals.argmin())],
               int(vals.max() - vals.min()),
               '较为均衡' if vals.std() < 30 else '不够均衡'))


def gen_line(ax, rng):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    vals = rng.integers(40, 100, 6)
    ax.plot(months, vals, marker='o', color=COLORS[0], linewidth=2)
    for x, v in zip(months, vals):
        ax.text(x, v + 3, str(v), ha='center', fontsize=8)
    ax.set_title('Monthly Active Users Trend', fontsize=11)
    ax.set_ylabel('Users (k)')
    ax.grid(alpha=0.3)
    trend = '上升' if vals[-1] > vals[0] else '下降'
    return ('这是一张折线图，展示了某产品一到六月的月活跃用户数变化趋势。横轴为月份，从 Jan 到 Jun，'
            '纵轴为用户数，单位为千人。具体数值为：一月 %d 千人，二月 %d 千人，三月 %d 千人，'
            '四月 %d 千人，五月 %d 千人，六月 %d 千人。整体呈现%s趋势，其中最高点出现在%s，'
            '为 %d 千人；最低点出现在%s，为 %d 千人。从走势看，用户规模在%s阶段变化最明显，'
            '建议结合该时间段的运营活动、版本发布或推广投放来归因，判断增长或下滑是否可持续。'
            % (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], trend,
               months[int(vals.argmax())], int(vals.max()),
               months[int(vals.argmin())], int(vals.min()),
               '后半段' if abs(vals[-1] - vals[0]) > 20 else '中间段'))


def gen_pie(ax, rng):
    labels = ['Direct', 'Search', 'Social', 'Email']
    vals = rng.dirichlet(np.ones(4)) * 100
    vals = np.round(vals / vals.sum() * 100).astype(int)
    vals[-1] = 100 - vals[:-1].sum()
    ax.pie(vals, labels=labels, autopct='%d%%', colors=COLORS[:4], startangle=90)
    ax.set_title('Traffic Source Share', fontsize=11)
    return ('这是一张饼图，展示了网站流量的来源构成。四个来源分别为 Direct、Search、Social 和 Email，'
            '占比依次为百分之 %d、百分之 %d、百分之 %d 和百分之 %d，合计百分之一百。'
            '可以看出，%s 是最大的流量来源，接近总量的一半；%s 的占比最小。'
            '如果目标是提升整体转化，建议优先优化占比最高来源的落地页体验，同时对占比低但增长潜力大的渠道做小规模测试。'
            % (vals[0], vals[1], vals[2], vals[3],
               labels[int(vals.argmax())], labels[int(vals.argmin())]))


def gen_scatter(ax, rng):
    x = rng.uniform(0, 10, 20)
    y = 2 * x + rng.normal(0, 2, 20)
    ax.scatter(x, y, color=COLORS[2], alpha=0.7)
    ax.set_title('Ad Spend vs Revenue', fontsize=11)
    ax.set_xlabel('Spend (k)')
    ax.set_ylabel('Revenue (k)')
    ax.grid(alpha=0.3)
    corr = np.corrcoef(x, y)[0, 1]
    return ('这是一张散点图，横轴为广告投入（单位千），纵轴为营业收入（单位千），图中共有二十个数据点。'
            '从点的分布看，投入与收入呈现%s的正相关关系，相关系数约为 %.2f。'
            '这意味着广告投入增加时，收入总体随之上升，但点存在一定离散，说明投入的边际效率并不稳定。'
            '建议进一步按渠道拆分数据，找出边际回报最高的投放组合，避免盲目加大整体预算。'
            % ('较强' if corr > 0.6 else '中等', corr))


def gen_schematic(ax, rng):
    ax.axis('off')
    boxes = ['Data\nCollection', 'Preprocessing', 'Model\nTraining', 'Evaluation', 'Deployment']
    xs = np.linspace(0.05, 0.8, 5)
    for i, (x, b) in enumerate(zip(xs, boxes)):
        ax.add_patch(plt.Rectangle((x, 0.42), 0.14, 0.16, fc=COLORS[i % 6], ec='black'))
        ax.text(x + 0.07, 0.5, b, ha='center', va='center', fontsize=7.5, color='white')
        if i < 4:
            ax.annotate('', xy=(xs[i + 1] - 0.005, 0.5), xytext=(x + 0.145, 0.5),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.set_title('ML Pipeline Overview', fontsize=11)
    return ('这是一张流程示意图，展示了机器学习项目从数据到上线的五个阶段，按箭头方向依次为：'
            '数据采集、数据预处理、模型训练、效果评估和部署上线。每个阶段用不同颜色的方块表示，'
            '箭头表示数据与模型的流转方向。理解这张图的意义在于：任何一环出现问题都会影响最终效果，'
            '其中数据预处理往往是耗时最长、也最容易被忽视的一环；评估阶段需要与训练阶段口径一致，'
            '否则容易产生离线好、线上差的情况。部署之后还需要持续监控，形成闭环。')


def gen_card(ax, rng):
    ax.axis('off')
    pts = ['Fast inference', 'Low memory', 'Multi-GPU ready', 'Easy to extend']
    for i, p in enumerate(pts):
        ax.add_patch(plt.Rectangle((0.05, 0.75 - i * 0.18), 0.9, 0.14, fc='#F0F0F0', ec='#999999'))
        ax.text(0.1, 0.82 - i * 0.18, '%d. %s' % (i + 1, p), fontsize=10)
    ax.set_title('Key Features', fontsize=11)
    return ('这是一张要点卡片图，列出四个关键特性，依次为：推理速度快、内存占用低、支持多卡并行、易于扩展。'
            '这类图片通常用于产品介绍或方案对比。从工程角度看，前两点直接影响部署成本，后两点决定长期迭代效率。'
            '建议在选型时把四项分别打分，结合业务场景的权重加权，而不是只看单项宣传。')


def gen_product(ax, rng):
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0.2, 0.35), 0.6, 0.3, fc='#DDDDDD', ec='black', lw=1.5))
    ax.add_patch(plt.Circle((0.5, 0.72), 0.08, fc=COLORS[0], ec='black'))
    ax.text(0.5, 0.24, 'Device Front View (512px mock)', ha='center', fontsize=9)
    ax.set_title('Product Sketch', fontsize=11)
    return ('这是一张产品正视图示意图，主体为一个矩形设备面板，上方中央有一个圆形按钮或指示灯，'
            '整体构图简洁。此类图片多用于说明书、装配图或外观评审。示意图的价值在于快速传达结构关系，'
            '但不包含尺寸、材质等细节，实际生产前还需要工程图纸补充。')


def gen_map(ax, rng):
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0.05, 0.1), 0.55, 0.75, fc='#A8D5E2', ec='black'))
    ax.add_patch(plt.Rectangle((0.62, 0.15), 0.3, 0.4, fc='#F2C14E', ec='black'))
    ax.add_patch(plt.Circle((0.3, 0.5), 0.05, fc='red'))
    ax.text(0.3, 0.62, 'HQ', ha='center', fontsize=9)
    ax.set_title('Regional Layout Sketch', fontsize=11)
    return ('这是一张区域布局示意图，左侧蓝色大区块表示主要区域，右侧黄色区块表示次要区域，'
            '红色圆点标记总部位置。此类布局图常用于物流网络、门店分布或机房规划。'
            '从布局看，总部位于主区域中心，到两侧区域的距离相对均衡，便于统一调度。'
            '实际规划时还应叠加交通、地价和人口密度等因素综合判断。')


GEN_FUNCS = [
    ('bar', gen_bar), ('line', gen_line), ('pie', gen_pie), ('scatter', gen_scatter),
    ('schematic', gen_schematic), ('card', gen_card), ('product', gen_product), ('map', gen_map),
]


def render(tag, idx, rng):
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=110)
    name, fn = GEN_FUNCS[idx % len(GEN_FUNCS)]
    desc = fn(ax, rng)
    img_path = os.path.join(IMG_DIR, '%s_%02d.png' % (name, idx))
    fig.tight_layout()
    fig.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    return img_path, desc


def make_split(start, count, path):
    rows = []
    for i in range(count):
        idx = start + i
        rng = np.random.default_rng(seed=1000 + idx)
        img_path, desc = render('', idx, rng)
        rows.append({
            'messages': [
                {'role': 'user', 'content': USER_PROMPT},
                {'role': 'assistant', 'content': desc},
            ],
            'images': [img_path],
        })
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('wrote %d samples -> %s' % (len(rows), path))


if __name__ == '__main__':
    make_split(0, 64, os.path.join(OUT, 'cpt_train.jsonl'))
    make_split(64, 8, os.path.join(OUT, 'cpt_val.jsonl'))
    n_img = len([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
    print('images: %d' % n_img)
