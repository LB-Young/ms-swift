# ms-swift 强化学习（RLHF / RL）算法详解笔记

> 范围：本机 `/root/ms-swift` 仓库（`swift/` 包）中 `swift rlhf` 子命令实际支持的 9 种 `rlhf_type`
> —— `dpo`、`cpo`、`simpo`、`orpo`、`kto`、`rm`、`ppo`、`grpo`、`gkd`，以及 GRPO 分支下
> 十余种「损失变体 / advantage 估计器 / 采样与纠偏技巧」。
> 所有公式均对应当前源码实现（不是论文原版照抄），差异处会明确标注。
>
> 源码入口：`swift/pipelines/train/rlhf.py` → `swift/rlhf_trainers/*.py`
> 参数定义：`swift/arguments/rlhf_args.py`、`swift/rlhf_trainers/args_mixin.py`
> 实验脚本：`experiments/rl/rl_*.sh`

---

## 目录

0. [一张图看懂：ms-swift 的 RL 全家桶](#0-一张图看懂ms-swift-的-rl-全家桶)
1. [统一符号与数据流](#1-统一符号与数据流)
2. [偏好优化族：DPO / CPO / SimPO / ORPO](#2-偏好优化族dpo--cpo--simpo--orpo)
3. [KTO：无配对数据的点级偏好](#3-kto无配对数据的点级偏好)
4. [RM：奖励模型训练](#4-rm奖励模型训练)
5. [PPO：经典 actor-critic](#5-ppo经典-actor-critic)
6. [GRPO 主族（重点）](#6-grpo-主族重点)
7. [GKD：广义知识蒸馏（含 on-policy 采样）](#7-gkd广义知识蒸馏含-on-policy-采样)
8. [算法选型与显存/工程要点](#8-算法选型与显存工程要点)
9. [参数速查表](#9-参数速查表)
10. [源码索引](#10-源码索引)

---

## 0. 一张图看懂：ms-swift 的 RL 全家桶

`rlhf_type` 是唯一的总开关。九种类型按「数据形态 + 是否需要在线采样」可以分成四类：

| 类别 | rlhf_type | 数据形态 | 需要 ref_model | 需要在线 rollout | 损失函数位置 |
|---|---|---|---|---|---|
| 离线偏好（配对） | `dpo` / `cpo` / `simpo` / `orpo` | chosen/rejected 成对 | dpo 需要；cpo/simpo/orpo 不需要 | 否 | `dpo_trainer.py` / trl CPO、ORPO |
| 离线偏好（非配对） | `kto` | 单条 + 好/坏标签 | 需要 | 否 | trl KTO + `kto_trainer.py` |
| 奖励建模 | `rm` | 成对 + 打分目标 | 不需要 | 否 | `reward_trainer.py` |
| 在线 RL | `ppo` / `grpo` | prompt（+ 可验证答案） | ppo/grpo 需要（β=0 时 grpo 可省） | 是 | trl PPO / `grpo_trainer.py` |
| 蒸馏 | `gkd` | prompt（+ 可选答案） | 不需要（用 teacher） | 可在线（vLLM） | `gkd_loss.py` |

GRPO 是「一个框架 + 多个算法开关」的设计，同一个 `--rlhf_type grpo` 下还能组合出：

```
advantage 估计器 : grpo | rloo | reinforce_plus_plus      (--advantage_estimator)
奖励归一化       : group | batch | none | gdpo            (--scale_rewards)
重要性采样粒度   : token | sequence | sequence_token      (--importance_sampling_level → GSPO 系)
策略损失         : grpo | bnpo | dr_grpo | dapo | cispo | sapo | real | fipo   (--loss_type)
采样增强         : dynamic_sample / overlong_filter / top_entropy_quantile / soft_overlong
离策略纠偏       : rollout_importance_sampling_mode / off_policy_sequence_mask_delta
蒸馏增强         : OPD-RL(teacher_kl_coef) / RLSD(advantage_reweight) / SDAR(sdar_loss_coef) / CHORD(chord_*)
```

---

## 1. 统一符号与数据流

### 1.1 符号约定

| 符号 | 含义 |
|---|---|
| $x$ | prompt（输入） |
| $y$ | completion / response；$y_t$ 为第 $t$ 个 token |
| $\pi_\theta$ | 正在训练的策略模型 |
| $\pi_{\text{ref}}$ | 参考模型（SFT 后的初始策略，冻结） |
| $\pi_{\text{old}}$ | 采样时的策略快照（importance sampling 的分母） |
| $\pi_{\text{roll}}$ | rollout 引擎（vLLM）实际采样用的策略 |
| $\log \pi(y\mid x)$ | completion 上的 **token logp 之和**：$\sum_{t=1}^{|y|}\log \pi(y_t\mid x,y_{<t})$ |
| $\beta$ | KL / 温度系数（DPO 温度、GRPO 的 KL 系数） |
| $G$ | `--num_generations`，每题采样条数 |
| $K$（$\mu$） | `--num_iterations`，同一批数据更新几次（GRPO 论文里的 $\mu$） |
| $\rho_t$ | token 级重要性采样比 $\pi_\theta(y_t)/\pi_{\text{old}}(y_t)$ |
| $A$ | advantage（序列级标量，广播到 token） |

**关键实现约定**（`dpo_trainer.py:147-206`、`grpo_trainer.py:939`）：

- logp 只在 **completion token** 上累加，`loss_mask` 由模板（`--loss_scale last_round`，RLHF 默认）给出；
- DPO 族默认 **sum**（不是平均）；IPO 例外，内部会除以 completion token 数（`_get_ipo_sequence_logps`，`dpo_trainer.py:224`）；
- GRPO 默认按 **token 平均**，另有 4 种归一化方式（见 6.4）。

### 1.2 数据是怎么流动的

```
离线（dpo/cpo/simpo/orpo/kto/rm）:
  dataset → template.encode(成对/单条) → 前向算 logp → 损失 → 反向

在线（ppo/grpo）:
  prompt → rollout(vLLM 或原生 generate) 采样 G 条
        → reward_funcs / reward_model 打分 → (可选 KL) → advantage
        → 前向算 ρ、KL → 策略损失 → 反向（μ 次）
        → (可选) sync ref model
```

---

## 2. 偏好优化族：DPO / CPO / SimPO / ORPO

### 2.1 DPO（`--rlhf_type dpo`）

**核心思想**：把「带 KL 约束的 RLHF 目标」解析地转成对偏好对的二分类损失，不再需要在线采样和 advantage。

理论推导（RLHF 最优解）：

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac{1}{\beta}r(x,y)\Big)
$$

反解出隐式奖励，代入 Bradley–Terry 模型，得到 DPO 损失：

$$
\mathcal{L}_{\text{DPO}}=-\log\sigma\Big(\beta\Big[\underbrace{\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\text{chosen 相对 log-ratio}}-\underbrace{\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\text{rejected 相对 log-ratio}}\Big]\Big)
$$

实现里把这两项拆成四个 logp：

```python
# dpo_trainer.py:240-258
chosen_logratios  = chosen_logps  - ref_chosen_logps
rejected_logratios= rejected_logps- ref_rejected_logps
logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
loss   = -logsigmoid(beta * logits)
```

`--reference_free true`（trl 原生参数）会把 ref 项置 0，退化成「只比 chosen/rejected 的绝对概率」。

**隐式奖励（日志里看的 rewards）**：

$$
\hat r_w=\beta\big(\log\pi_\theta(y_w)-\log\pi_{\text{ref}}(y_w)\big),\quad
\hat r_l=\beta\big(\log\pi_\theta(y_l)-\log\pi_{\text{ref}}(y_l)\big)
$$

`rewards/accuracies = mean(\hat r_w > \hat r_l)`，`rewards/margins = mean(\hat r_w-\hat r_l)`（`dpo_trainer.py:360-433`）。这两个指标是判断 DPO 是否真的在学的第一手信号：accuracy 长期 ≤0.5 说明数据或 β 有问题。

#### 2.1.1 DPO 的 14 种 `--loss_type`（swift 全部实现）

`dpo_trainer.py:263-358` 逐个实现了下面的损失；多值可混用（MPO），见 2.1.3。

| loss_type | 公式（$\ell=\beta\cdot\text{logits}$） | 说明 / 出处 |
|---|---|---|
| `sigmoid` | $-\log\sigma(\ell)$ | 标准 DPO |
| `robust` | $\dfrac{-\log\sigma(\ell)(1-\epsilon)+\log\sigma(-\ell)\epsilon}{1-2\epsilon}$ | 鲁棒版，需 $\epsilon<0.5$ |
| `exo_pair` | $\sigma(\ell)\big(\log\sigma(\ell)-\log(1-\epsilon)\big)+\sigma(-\ell)\big(\log\sigma(-\ell)-\log\epsilon\big)$ | EXO，正负两侧都平滑 |
| `hinge` | $\max(0,1-\ell)$ | 合页损失（SVM 式 margin） |
| `ipo` | $\big(\text{logits}-\frac{1}{2\beta}\big)^2$ | IPO；**logp 先按 token 长度平均** |
| `bco_pair` | $-\log\sigma(\beta c-\delta)-\log\sigma(-(\beta r-\delta))$ | c/r 是各自相对 log-ratio，$\delta$ 为 running mean |
| `sppo_hard` | $\big(a-\frac{1}{2\beta}\big)^2+\big(b+\frac{1}{2\beta}\big)^2$ | a/b = chosen/rejected 的 $\log\pi_\theta-\log\pi_{\text{ref}}$ |
| `nca_pair` | $-\log\sigma(\beta c)-\tfrac12\log\sigma(-\beta c)-\tfrac12\log\sigma(-\beta r)$ | NCA |
| `aot` | $-\log\sigma(\beta\Delta_{\text{sorted}})$（Δ 按 $\text{logit}-\text{ref logit}$ 排序后相减） | AOT |
| `aot_unpaired` | 同上，但用各自 $\log\pi_\theta-\log\pi_{\text{ref}}$ 排序后相减 | AOT 非配对版 |
| `apo_zero` | $\big(1-\sigma(\beta c)\big)+\sigma(\beta r)$ | APO-zero：只压 chosen 上升过快 |
| `apo_down` | $\sigma(\beta c)+\big(1-\sigma(\beta(c-r))\big)$ | APO-down：允许 chosen 下降 |
| `discopop` | $-\log\sigma(\ell)\,(1-m)+\exp(-\ell)\,m$，$m=\sigma(\ell/\tau)$ | DiscoPOP 动态损失切换，$\tau$=`--discopop_tau` |
| `sft` | $\mathcal{L}_{\text{NLL}}$（chosen 的 token 级 NLL） | 退化为纯 SFT，可与其它损失混合 |

> 注意：`hinge`/`ipo`/`bco_pair`/`sppo_hard`/`nca_pair`/`apo_zero`/`apo_down` 与 `--label_smoothing>0` 冲突，代码会给出警告并忽略平滑（`dpo_trainer.py:71-77`）。

#### 2.1.2 三个 DPO 增强开关

**(a) cDPO：`--label_smoothing ε`** —— 承认偏好标签有噪声：

$$
\mathcal{L}=-(1-\epsilon)\log\sigma(\ell)-\epsilon\log\sigma(-\ell)
$$

**(b) RPO：`--rpo_alpha α`** —— 在 DPO 损失上补回 SFT 项，防止「只会拉开差距、不会生成」：

$$
\mathcal{L}=\mathcal{L}_{\text{DPO}}+\alpha\,\mathcal{L}_{\text{NLL}}(y_w)
$$

`NLL` 在 `concatenated_forward` 里算：chosen 的 per-token logp 取负再平均（`dpo_trainer.py:179`）。

**(c) LD-DPO：`--ld_alpha α`** —— 抑制长度偏好。把 completion 切成「公共前缀 $l_p=\min(l_w,l_l)$」与「尾部」两段，两段 logp 加权：

$$
\log\pi_{\text{LD}}(y)=\underbrace{\sum_{t\le l_p}\log\pi(y_t)}_{\text{front}}+\alpha\underbrace{\sum_{t>l_p}\log\pi(y_t)}_{\text{rear}}
$$

实现见 `dpo_trainer.py:180-200`（`front_mask`/`rear_mask` 按 `completion_position_ids < public_lengths` 切）。$\alpha=0$ 时尾部完全不参与，等价于只看公共前缀。

#### 2.1.3 MPO：多种 `loss_type` 混合

`--loss_type 'sigmoid' 'bco_pair' --loss_weights 1.0 0.5`：逐项求和（`dpo_trainer.py:386-415`）。仅 DPO 支持（`rlhf_args.py:320-327` 有断言）。

#### 2.1.4 f-divergence 变体

`--f_divergence_type js_divergence`：在 logits 上减去

$$
\text{softplus}(\text{chosen\_logratio})-\text{softplus}(\text{rejected\_logratio})
$$

`--f_divergence_type alpha_divergence`：改用 α-散度形式（`dpo_trainer.py:243-248`）。这两个是 trl 的扩展参数，swift 透传。

### 2.2 CPO（`--rlhf_type cpo`）

**一句话**：DPO 的偏好项 + chosen 上的 NLL 项，**不需要参考模型**（把 ref 用 SFT 项替代）。

$$
\mathcal{L}_{\text{CPO}}=-\log\sigma\Big(\beta\big[\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)\big]\Big)+\alpha\,\big(-\log\pi_\theta(y_w\mid x)\big)
$$

- $\alpha$ = `--cpo_alpha`（默认 1.0），$\alpha\to0$ 即为纯对比损失；
- 实现委托 trl 的 `CPOTrainer`，swift 只做参数搬运（`cpo_trainer.py`，其中 `assert ref_model is None`）；
- 代码里若 `--loss_type simpo` 且 `cpo_alpha>0`，会告警「你其实在用 CPO-SimPO」（`cpo_trainer.py:31-37`）。

### 2.3 SimPO（`--rlhf_type simpo`）

**一句话**：CPO 去掉 ref，再把 logp 做**长度归一化**并加 margin，让「好答案的似然密度」直接高于坏答案。

$$
\mathcal{L}_{\text{SimPO}}=-\log\sigma\Big(\frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x)-\gamma\Big)+\alpha\,\mathcal{L}_{\text{NLL}}(y_w)
$$

- $\gamma$ = `--simpo_gamma`（默认 1.0，论文建议 0.5–1.5）；
- $\beta$ 在 simpo 下默认 **2.0**（其它算法 0.1），见 `rlhf_args.py:192-194`；
- $\alpha$ = `--cpo_alpha`，置 0 才是「纯 SimPO」。

> CPO / SimPO 的分母 $\beta$ 与长度归一化在 trl 内部实现，swift 侧只是把 `loss_type`、`cpo_alpha`、`simpo_gamma` 传下去。

### 2.4 ORPO（`--rlhf_type orpo`）

**一句话**：SFT 损失 + 赔率比（odds ratio）惩罚，单阶段、无 ref、无额外对齐阶段。

$$
\mathcal{L}_{\text{ORPO}}=\mathcal{L}_{\text{SFT}}+\lambda\,\mathcal{L}_{\text{OR}},\qquad
\mathcal{L}_{\text{OR}}=-\log\sigma\Big(\log\frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\Big)
$$

其中 $\text{odds}_\theta(y)=\dfrac{\pi_\theta(y\mid x)}{1-\pi_\theta(y\mid x)}$。

- swift 侧 `orpo_trainer.py` 仅断言 `ref_model is None`，损失完全由 trl 的 `ORPOTrainer` 计算；
- ⚠️ **$\lambda$ 未在 swift 的 `RLHFArguments` 中暴露**（没有 `lambda_or` 字段），走 trl 默认值（0.1）。需要改就改 trl config 或直接 monkey-patch；
- 对于**多模态模型**，`loss_scale` 会强制用 `default` 而不是 `last_round`（`rlhf_args.py:301-306`），避免视觉 token 引起的 label 形状问题。

### 2.5 四者横向对比

| 维度 | DPO | CPO | SimPO | ORPO |
|---|---|---|---|---|
| ref 模型 | 必须 | 无 | 无 | 无 |
| 长度归一化 | 无（可 IPO/LD 变体） | 无 | 有 | 无 |
| margin $\gamma$ | 无 | 无 | 有 | 无（用 odds 隐式） |
| 稳定项 | RPO $\alpha$ 可选 | NLL $\alpha$ | NLL $\alpha$ | SFT 项常驻 |
| swift 默认 β | 0.1 | 0.1 | **2.0** | 0.1 |
| 显存 | 模型+ref（LoRA 可省 ref 副本） | 单模型 | 单模型 | 单模型 |

---

## 3. KTO：无配对数据的点级偏好

**动机**：真实业务里往往只有「这条回答好/坏」的二元反馈，没有同一 prompt 下的成对样本。KTO 把偏好学习转成对**单条样本**的效用分类。

**核心公式**（KTO 论文形式，trl 实现即此）：

$$
v(x,y)=\sigma\Big(\beta\big[\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}-z_0\big]\Big)
$$

$$
\mathcal{L}_{\text{KTO}}=\lambda_w\Big(1-v(x,y_w)\Big)+\lambda_l\Big(1-v(x,y_l)\Big)
$$

- $z_0$ 是 KL 参照点（batch 内 $\mathbb{E}[\log\pi_\theta-\log\pi_{\text{ref}}]$ 的滑动估计，trl 里 `kl` 项）；
- $\lambda_w$ / $\lambda_l$ = `--desirable_weight` / `--undesirable_weight`，用来抵消好/坏样本数量不平衡（默认 1.0 / 1.0）；
- `--loss_type apo_zero_unpaired` 会把 KL 项关掉（`kto_trainer.py:44-46` 的 `self.calculate_KL=False`）。

**实现要点**：
- `kto_trainer.py` 覆写了 `forward`，从 batch 里分别取 `completion_*` 与 `KL_completion_*` 前缀字段，chosen/rejected 按 `batch['label']` 拆分（`kto_trainer.py:67-79`）；
- `args.disable_dropout = True`（KTO 强制关 dropout，否则 KL 项噪声太大）；
- `--precompute_ref_log_probs true` 可预先缓存 ref logp，省一半前向。

> KTO 的 KL 是**点级**（单条序列与 ref 的 KL），不是 GRPO 那种逐 token k3。这个 `.KL` 项在 trl 里是一个标量参照，不是正则项本身。

---

## 4. RM：奖励模型训练

`--rlhf_type rm`，目标是为 GRPO/PPO 或后续排序提供打分器。

**Bradley–Terry 损失**（`reward_trainer.py:47-55`）：

$$
\mathcal{L}_{\text{RM}}=-\log\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)-m\big)
$$

- $m$ = 数据集里的 `margin` 字段（可选，缺省为 0）；
- `--center_rewards_coefficient c`：鼓励奖励零均值，抑制奖励尺度漂移：
  
$$
\mathcal{L}\leftarrow\mathcal{L}+c\cdot\mathbb{E}\big[(r_w+r_l)^2\big]
$$

- 输出是 `logits`（标量奖励），评估指标为 `rewards/accuracies`、`rewards/margins`；
- 训练好后：`--rlhf_type grpo --reward_model <rm 路径> --reward_model_plugin ...`，或 GRPO 里直接把 `reward_model` 当最后一个奖励源（`rlhf_args.py:146-149`）。

---

## 5. PPO：经典 actor-critic

`--rlhf_type ppo` 走 trl 的 `PPOTrainer`（`ppo_trainer.py` 只做 swift 的模型/数据打补丁，`create_loss_and_eval_metric` 直接返回 `{}`）。

**策略损失（clipped surrogate）**：

$$
\mathcal{L}^{\text{CLIP}}_t=-\min\Big(\rho_t A_t,\ \text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\Big),\quad
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}
$$

**价值损失（带裁剪）**：

$$
\mathcal{L}^{VF}=\max\Big((V_\theta(s_t)-R_t)^2,\ \big(\text{clip}(V_\theta, V_{\text{old}}-\epsilon_v, V_{\text{old}}+\epsilon_v)-R_t\big)^2\Big)
$$

**GAE（$\gamma$, $\lambda$）**：

$$
\hat A_t=\sum_{l=0}^{T-t}(\gamma\lambda)^l\,\delta_{t+l},\qquad
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

**总损失**：

$$
\mathcal{L}=\mathcal{L}^{\text{CLIP}}+c_1\mathcal{L}^{VF}+c_2\,\mathbb{E}\big[\log\pi_\theta(a_t\mid s_t)-\log\pi_{\text{ref}}(a_t\mid s_t)\big]
$$

对应参数（`rlhf_args.py:114-127`）：`--num_ppo_epochs 4`、`--cliprange 0.2`、`--vf_coef 0.1`、`--cliprange_value 0.2`、`--gamma 1.0`、`--lam 0.95`、`--kl_coef 0.05`、`--whiten_rewards`、`--num_mini_batches 1`、`--missing_eos_penalty`。

> PPO 在 LLM 场景要同时跑 policy / value / reward / ref 四个模型，显存压力最大。**实践上大模型一律优先 GRPO**（去掉 value model，用组内均值当 baseline）。

---

## 6. GRPO 主族（重点）

`--rlhf_type grpo` 是 swift 里最活跃的分支（`grpo_trainer.py` 2200+ 行，`rl_core/advantage.py` 400 行）。

### 6.1 骨架：组采样 + 组内 baseline

对每个 prompt 采 $G$ 条回答 $\{y_i\}_{i=1}^G$，用**组内均值**代替 critic 估计 baseline：

$$
A_i=\frac{r_i-\text{mean}(r_{1:G})}{\text{std}(r_{1:G})}
$$

$\text{std}$ 的具体算法由 `--scale_rewards` 决定，见 6.2。

**策略损失**（`grpo_trainer.py:1052-1058`）：

$$
\mathcal{L}=-\frac{1}{\sum_t m_t}\sum_t m_t\min\Big(\rho_t A_t,\ \text{clip}(\rho_t,1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})A_t\Big)
$$

- $\epsilon_{\text{low}}$ = `--epsilon`（默认 0.2），$\epsilon_{\text{high}}$ = `--epsilon_high`（缺省等于 `--epsilon`）；
- 非对称裁剪（DAPO 的关键改动）：只放宽上界，允许低概率 token 更大步上升；
- `--delta`：双边裁剪（Dual-Clip），把 $\rho_t$ 上界再压到 $\delta$，防止极端 off-policy 样本主导（`grpo_trainer.py:1053-1054`）。

**KL 正则**（`grpo_trainer.py:972-976`，k3 估计器）：

$$
\widehat{\text{KL}}_t=\exp(\delta_t)-\delta_t-1,\qquad \delta_t=\log\pi_{\text{ref}}(y_t)-\log\pi_\theta(y_t)
$$

- 数值上 $\delta_t$ 先 clamp 到 $[-20,20]$，结果再 clamp 到 $[-10,10]$；
- `--beta`（grpo 默认 0.04）控制强度；`--beta 0` 时 `ref_model` 自动置空（`rlhf_args.py:292-293`），省一份模型。

### 6.2 advantage 估计器 × 奖励归一化

`rl_core/advantage.py:58-99` 是纯张量实现（HF / Megatron / Ray 共用）。先按权重合并多奖励：

$$r_i=\sum_f w_f\,r_{i,f}$$

#### (a) `--advantage_estimator`

| 取值 | 公式 | 特点 |
|---|---|---|
| `grpo` | $A_i=r_i-\bar r_{\text{group}}$ | 标准；$\bar r$ 为组内均值 |
| `rloo` | $A_i=\dfrac{G}{G-1}\big(r_i-\bar r_{\text{group}}\big)$ | 留一法 baseline，无偏且放大 $G/(G-1)$ 倍 |
| `reinforce_plus_plus` | 同 grpo 的分子，但 std 用 **batch 级** | 方差更小，$G=1$ 也能用 |

`--kl_in_reward`（默认：grpo→False，rloo/reinforce++→True）决定 KL 放在哪：

$$
r_i \leftarrow r_i-\beta\,\text{KL}_i \quad(\text{`kl_in_reward=true`})
$$

否则 KL 作为逐 token 损失项加在策略损失里（6.1）。

#### (b) `--scale_rewards`

| 取值 | $\text{std}$ 的计算 | 适用 |
|---|---|---|
| `group`（默认） | 每个 prompt 组内 $\text{std}(r_{1:G})$ | 标准 GRPO，题目难度自适应 |
| `batch` | 整个 batch 的 $\text{std}(r)$ | REINFORCE++ / 跨题可比 |
| `none` | 不做标准化 | Dr.GRPO 思路，避免「难度偏差」 |
| `gdpo` | **每个奖励函数分别组内标准化**，再求和、再全局标准化 | 多奖励（如 accuracy+format）尺度不一致时 |

**GDPO 细节**（`advantage.py:83-94`）：

$$
\tilde r_{i,f}=\frac{r_{i,f}-\mu_{f,\text{group}}}{\sigma_{f,\text{group}}+\epsilon},\qquad
A_i=\frac{\sum_f w_f\tilde r_{i,f}-\overline{\sum_f w_f\tilde r_{i,f}}}{\text{std}+\epsilon}
$$

也就是「先按函数归一化，再加权求和，再整体归一化」——避免某一项奖励量纲过大淹没其它项。

> GSPO 论文（`scale_rewards='none'` + sequence-level IS）与 Dr.GRPO（去 std）都指向同一个经验：**批次级 std 会给简单题过大梯度**。

### 6.3 重要性采样粒度：GSPO 系（`--importance_sampling_level`）

`grpo_trainer.py:1008-1025`：

| 取值 | 定义 | 语义 |
|---|---|---|
| `token`（默认） | $\rho_t=\exp(\log\pi_\theta(y_t)-\log\pi_{\text{old}}(y_t))$ | 逐 token 比（PPO 式） |
| `sequence` | $\rho_i=\exp\Big(\frac{1}{|y_i|}\sum_t\big(\log\pi_\theta(y_t)-\log\pi_{\text{old}}(y_t)\big)\Big)$ | **GSPO**：整条序列一个比 |
| `sequence_token` | $\rho_t=\exp\big(\text{sg}[s_i]+\log\pi_\theta(y_t)-\text{sg}[\log\pi_\theta(y_t)]\big)$ | GSPO-token：在序列比上再加 token 残差（stop-grad 形式） |

**GSPO 的动机**：token 级比在长序列上累积方差极大，一个 token 的 $\rho$ 爆掉会让整条序列的梯度噪声爆炸。序列级比把「整条回答」当作一个动作，裁剪也更符合「回答级」语义。官方推荐 $\epsilon$ 用 3e-4 量级（比 token 级小两个数量级）。

### 6.4 策略损失的归一化变体（`--loss_type`）

`grpo_trainer.py:1036-1134` 与常量 `WINDOW_NORMALIZED_LOSS_TYPES`（第 81 行）。

| loss_type | 逐 token 损失 | 归一化 | 提出方 |
|---|---|---|---|
| `grpo`（默认） | $-\min(\rho_t A_t,\text{clip}(\rho_t)A_t)$ | 每个序列内 token 平均，再 batch 平均 | GRPO |
| `bnpo` | 同上 | **全 batch token 平均**（batch-normalized） | 社区实践 |
| `dr_grpo` | 同上 | 除以 $B\cdot L_{\max}$（固定分母） | Dr.GRPO：消除长度/难度偏差 |
| `dapo` | 同上 | 除以**全局 completion token 总数**（跨进程） | DAPO |
| `cispo` | $-\text{sg}\big[\text{clip}(\rho_t,\max=\epsilon_{\text{high}})\big]\,A_t\,\log\pi_\theta(y_t)$ | 同 dapo（窗口归一化） | CISPO：只裁上界、保留梯度 |
| `sapo` | $-\text{sg}\big[g(\rho_t)\big]A_t$ | 序列 token 平均 → batch 平均 | SAPO：软自适应门 |
| `real` | 见下（组内 listwise logistic） | 组级，无归一化（强制 `scale_rewards='none'`） | REAL |
| `fipo` | 基础项 × 未来 KL 影响权重 | 同 dapo | FIPO |

#### CISPO
$$\mathcal{L}_t=-\text{sg}\big[\min(\rho_t,\epsilon_{\text{high}})\big]\,A_t\,\log\pi_\theta(y_t)$$
裁剪作用在**系数**上且被 stop-grad，梯度只从 $\log\pi_\theta$ 那条路来 → 不会因为 $\rho$ 被裁而死梯度。适合「$\rho$ 分布很宽」的长序列场景。

#### SAPO（Soft Adaptive Policy-gradient Optimization）
$$
g_{\pm}(\rho)=\sigma\big(\tau_\pm(\rho-1)\big)\cdot\frac{4}{\tau_\pm},\qquad
g(\rho)=\begin{cases}g_+(\rho),&A>0\\ g_-(\rho),&A\le 0\end{cases}
$$
- 用平滑门替代硬 clip：$\rho\approx1$ 时门≈常数（梯度近似 PPO），偏离越远门越饱和；
- 正/负样本用不同温度：`--tau_pos 1.0`、`--tau_neg 1.05`（负样本更宽容，因为负 advantage 的 $\rho$ 噪声更危险）。

#### REAL（组内正负对比，listwise）
对每个 prompt 组，把「正 advantage 集合」与「负 advantage 集合」做对称 logistic：

$$
\mathcal{L}=\frac{1}{|G|}\sum_{\text{groups}}\Big[\log\big(1+\sum_{i\in\text{neg}}e^{s_i/\tau}\big)+\log\big(1+\sum_{i\in\text{pos}}e^{-s_i/\tau}\big)\Big]
$$
$$
s_i=\frac{1}{|y_i|}\sum_t\big(\log\pi_\theta(y_t)-\log\pi_{\text{old}}(y_t)\big)
$$

- $\tau$ = `--real_tau`（默认 0.5）；
- 实现见 `grpo_trainer.py:1090-1125`；组内必须同时存在正、负样本，否则该组跳过；
- `scale_rewards` 被强制为 `none`（`rlhf_args.py:378-383`），因为这里不再需要 adv 归一化。

#### FIPO（Future-KL Influence）
先用「未来 token 的 log-ratio 衰减和」算每个 token 的**影响力权重**（`grpo_trainer.py:885-932`）：

$$
F_t=\sum_{t'\ge t}\gamma^{t'-t}\,\big(\log\pi_\theta(y_{t'})-\log\pi_{\text{old}}(y_{t'})\big),\qquad
\gamma=2^{-1/\text{fipo\_decay\_rate}}
$$
$$
w_t=\text{clip}\big(e^{F_t},\,1,\ 1+\delta_{\text{fipo}}\big)\ (\text{high-only 默认}),\qquad \mathcal{L}_t\leftarrow w_t\cdot\mathcal{L}_t^{\text{base}}
$$

- `--fipo_decay_rate 32` → $\gamma\approx0.979$；
- `--fipo_clip_range 0.2` 控制 $w_t$ 上界；`--fipo_clip_high_only` 默认只放大不缩小；
- `--fipo_safety_threshold 4`：当 $\rho_t>4$ 且 $A<0$ 时，把权重压回 $[0.8,1]$，避免「负样本 × 大 IS 比」被进一步放大。

### 6.5 DAPO 三件套

1. **非对称裁剪**：`--epsilon 0.2 --epsilon_high 0.28`（上界放宽）。
2. **动态采样**（`grpo_trainer.py:627-677`）：丢弃「组内 reward 全同（std=0）」的样本组并重新采样，最多 `--max_resample_times 3` 次。std=0 的组 advantage 恒为 0，白算一遍前向。
3. **超长过滤**：`--overlong_filter true` 把被截断（`truncated_mask`）的 completion 从损失与指标里剔除（`grpo_trainer.py:965-970`）；更柔和的版本是奖励函数 `soft_overlong`：
   
$$
r_{\text{overlong}}=\min\Big(-\frac{\max(0,\,|y|-L_{\text{soft}})}{L_{\text{cache}}},\,0\Big)
$$
   需要同时给 `--soft_max_length`、`--soft_cache_length`（`rlhf_args.py:363-368` 会自动把 `soft_max_length` 设成 `max_completion_length`）。

4. **token-level 损失**：即 `--loss_type dapo`（6.4）。

### 6.6 熵掩码（Beyond the 80/20 Rule）

`--top_entropy_quantile q`（默认 1.0 = 关闭）：只保留熵最高的 $q$ 比例的 token 参与损失（`grpo_trainer.py:958-962`）：

$$
m_t=\mathbf{1}\big[H_t\ge \text{quantile}_{1-q}(H)\big]
$$

直觉：低熵 token 早已学好，梯度只会加剧确定性；高熵 token 才是「决策点」。配合 `--log_entropy` 可看熵曲线。

### 6.7 rollout 侧离策略纠偏

vLLM 的采样策略 $\pi_{\text{roll}}$ 与训练策略 $\pi_\theta$ 存在数值/版本差异（BF16 vs FP32、参数同步延迟）。swift 提供两套机制。

**(a) `--rollout_importance_sampling_mode`**（`grpo_trainer.py:2345-2393`）

$$
\rho^{\text{roll}}_t=\exp\big(\log\pi_\theta(y_t)-\log\pi_{\text{roll}}(y_t)\big)
$$

| 模式 | 处理 |
|---|---|
| `token_truncate` | $\min(\rho_t,C)$，逐 token 截断 |
| `token_mask` | $\rho_t>C$ 的 token 置 0 |
| `sequence_truncate` | 序列级比 $\bar\rho$ 截断到 $C$，再广播到 token |
| `sequence_mask` | $\bar\rho>C$ 的整条序列权重置 0 |

$C$ = `--rollout_importance_sampling_threshold`（默认 2.0），log-ratio 先 clamp 到 $\pm20$ 防溢出。

**(b) `--off_policy_sequence_mask_delta δ`**（DeepSeek-V3.2 的 Off-Policy Sequence Masking，`grpo_trainer.py:2395-2435`）：

$$
\text{mask}_i=\mathbf{1}\Big[\overline{\big(\log\pi_{\text{old}}-\log\pi_\theta\big)}_i>\delta\ \wedge\ A_i<0\Big]
$$

只对「旧策略明显比当前更自信 且 advantage 为负」的序列关门——这正是最容易带偏训练的样本。

### 6.8 OPD-RL：把 teacher 蒸馏塞进 GRPO

当 GRPO 同时给了 `--teacher_model`（或 `--teacher_model_server`）时，框架切成 **OPD-RL（On-Policy Distillation as RL）**：teacher 在**同一批采样 token** 上打分，把「teacher 比学生更认可这个 token」的程度作为逐 token 的额外 advantage（`rl_core/advantage.py:241-291`）：

$$
A_t=A_i+c\,\big(\log\pi_T(y_t)-\log\pi_S(y_t)\big)
$$

- $c$ = `--teacher_kl_coef`（默认 1.0）；
- 只有「符号」有意义的是 k1 估计器（signed log-ratio），监控用的 k3 指标另算（`compute_teacher_kl_per_token`）；
- teacher 的三种来源：独立小模型 ckpt / `teacher_model_server`（vLLM 或 swift deploy，支持多 teacher 的 JSON 配置）/ `disable_adapter()`（LoRA 训练下用基座当 teacher，`rlhf_args.py:426-434`，省一份模型）。

> 这是「稠密奖励」的一种廉价实现：环境奖励只给序列级 0/1，teacher 却给每个 token 一个连续信号。

### 6.9 RLSD：Self-Distilled RLVR 的 advantage 再加权

`--advantage_reweight rlsd`（`rl_core/advantage.py:294-343`）。在**不改变 advantage 符号**的前提下，用 teacher-学生 logp 差在序列内部重新分配 advantage：

$$
\Delta_t=\text{sg}\big(\log\pi_T(y_t)-\log\pi_S(y_t)\big),\qquad
w_t=\exp\big(\text{sign}(A)\cdot\Delta_t\big)
$$
$$
\tilde w_t=(1-\lambda)+\lambda\,\text{clip}(w_t,1-\epsilon_w,1+\epsilon_w),\qquad
A_t=A\cdot\text{sg}(\tilde w_t)
$$

- $\lambda$ = `--rlsd_lambda`，$\epsilon_w$ = `--rlsd_reweight_clip_range`（0.2）；
- $\lambda=0$ 退化成普通 GRPO；$\lambda=1$ 为完整 RLSD；
- `--rlsd_lambda_warmup_steps` / `--rlsd_lambda_decay_steps`：λ 线性升温/衰减（`grpo_trainer.py:2191-2207`）；
- `--rlsd_negative_only`：只对 $A<0$ 的序列做重加权（错误回答更需要「老师指路」）；
- 依赖数据集里的 `teacher_prompt` 列（带标准答案的 privileged prompt）；teacher 可以是「当前策略 + 另一个 prompt 视角」（动态自蒸馏），也可以是 `--teacher_model` 冻结 ckpt。

### 6.10 SDAR：置信度门控的蒸馏辅助损失

`--sdar_loss_coef λ_s > 0` 启用（`rl_core/advantage.py:362-420`）。**不改 advantage**，而是加一项门控蒸馏损失：

$$
\Delta_t=\log\pi_T(y_t)-\log\pi_S(y_t),\qquad g_t=\sigma\big(\beta_g\Delta_t\big)
$$
$$
\mathcal{L}_{\text{SDAR}}=\frac{\sum_t m_t\,g_t\,\Delta_t}{\sum_t m_t},\qquad
\mathcal{L}=\mathcal{L}_{\text{policy}}+\lambda_s\mathcal{L}_{\text{SDAR}}
$$

- $\beta_g$ = `--sdar_gate_beta`（默认 5.0，越大门越硬）；
- 门 $g_t$ 与 teacher logp 都 `detach`，梯度只经学生 $\log\pi_S$；
- 参考实现取 $\lambda_s=0.1$（ALFWorld 场景 0.01）；
- 与 RLSD 互斥（一个改 advantage、一个加损失，不能同开）；
- 常用于 **agentic（多轮工具调用）** 场景：teacher 看得到 skill/privileged 上下文，能纠正「看不见的信息」。

### 6.11 CHORD：GRPO 与 SFT 的课程式混合

`--chord_sft_dataset ...`（`grpo_trainer.py:2209-2213`、`utils.py:compute_chord_loss`）：

$$
\mathcal{L}=(1-\mu_s)\,\mathcal{L}_{\text{GRPO}}+\mu_s\,\mathcal{L}_{\text{SFT}},\qquad
\mu_s=\text{schedule}(s;\ \text{warmup},\text{decay},\text{peak},\text{valley})
$$

- `--chord_mu_warmup_steps` / `--chord_mu_decay_steps` / `--chord_mu_peak` / `--chord_mu_valley` 控制 $\mu_s$ 曲线；
- `--chord_enable_phi_function`：给 SFT 项乘置信度权重 $\phi=p(1-p)$（$p=e^{-\text{NLL}}$），只在「模型半懂不懂」的 token 上使劲；
- 论文：CHORD (arXiv 2508.11408)。思路是把「冷启动 SFT」和「RL 探索」揉进同一次训练，抑制 RL 阶段的格式崩塌。

### 6.12 多步更新（$\mu$ / `--num_iterations`）

`--num_iterations K>1`：同一批采样做 K 次梯度更新。第 1 步用采样时 logp 作 $\pi_{\text{old}}$，后续步骤仍然用**同一份** `old_per_token_logps`（快照），因此 $\rho$ 会逐步偏离 1，这正是 clip 起作用的场景。K 越大越省采样成本，但 off-policy 偏差越大。

### 6.13 其它工程开关（简要）

| 开关 | 作用 |
|---|---|
| `--num_generations G` | 组大小；GRPO 小于 2 会直接报错（`rlhf_args.py:409`） |
| `--use_vllm true/false` | vLLM 加速 rollout；`--vllm_mode colocate/server` |
| `--async_generate` | 用旧权重异步生成（近似 off-policy），必须 `vllm_mode=server` |
| `--multi_turn_scheduler` / `--max_turns` | 多轮工具调用（agentic）训练 |
| `--use_gym_env` / `--gym_env` | gym 环境奖励（`total_reward` 作为额外奖励列） |
| `--sync_ref_model` + `--ref_model_sync_steps` + `--ref_model_mixup_alpha` | 定期把 ref 往当前策略挪（防止 KL 约束过死） |
| `--reward_funcs accuracy format cosine repetition soft_overlong math react` | 内置可验证奖励（`swift/rewards/orm.py`），支持 `AsyncORM` 异步并行 |
| `--reward_weights` | 多奖励加权；配 `--scale_rewards gdpo` 更稳 |
| `--num_generations_eval` | 评估时用更少的采样条数省时间 |

---

## 7. GKD：广义知识蒸馏（含 on-policy 采样）

`--rlhf_type gkd`，`swift/rlhf_trainers/gkd_loss.py` + `gkd_trainer.py`。

### 7.1 损失：温度缩放 + 广义 JSD

设学生/教师 logits 为 $z_S,z_T$，温度 $T$ = `--temperature`（默认 0.9）：

$$
P_S=\text{softmax}(z_S/T),\qquad P_T=\text{softmax}(z_T/T)
$$

**广义 JSD**（$\beta$ = `--beta`，gkd 默认 0.5）：

$$
\mathcal{L}_{\text{JSD}_\beta}=\beta\,\text{KL}\big(P_T\,\|\,P_M\big)+(1-\beta)\,\text{KL}\big(P_S\,\|\,P_M\big),\qquad
P_M=\beta P_T+(1-\beta)P_S
$$

特例：
- $\beta=0$ → $\text{KL}(P_S\|P_T)$（**reverse KL**，mode-seeking，学「像不像老师」）；
- $\beta=1$ → $\text{KL}(P_T\|P_S)$（**forward KL**，mode-covering，逼学生覆盖老师全分布）；
- $0<\beta<1$ → 对称 JSD（默认 0.5，平衡两者）。

实现：`jsd_loss`（`gkd_loss.py:94-153`）分块计算，混合分布用 $\log\sum\exp$ 在 log 空间算：

$$
\log P_M=\text{logsumexp}\big(\log P_S+\log(1-\beta),\ \log P_T+\log\beta\big)
$$

### 7.2 on-policy 与数据集采样混合（`--lmbda`）

每条样本按概率掷硬币（`gkd_trainer.py:299-314`）：

$$
\text{data source}=\begin{cases}\text{STUDENT（学生自生成）}, & u\le\lambda\\ \text{DATASET（数据集答案）}, & u>\lambda\end{cases}
$$

- $\lambda$ = `--lmbda`（默认 0.5）；
- $\lambda=1$ → 纯 on-policy GKD（MiniLLM 式，学生自己采样、老师打分）；
- $\lambda=0$ → 纯离线蒸馏（固定数据集）。

### 7.3 三种 teacher 模式（`rlhf_args.py:44-52`）

| 模式 | 配置 | 效果 |
|---|---|---|
| 自蒸馏·动态 | 不给 `--teacher_model` | teacher = 当前学生权重（边学边教） |
| 自蒸馏·冻结 | `--teacher_model` 与 `--model` 相同 + LoRA + 无 `teacher_adapters` | 用 `disable_adapter()` 拿基座 logits，**零额外显存** |
| 独立 teacher | `--teacher_model` 指向别的模型 | 标准 GKD，teacher 全程冻结 $\pi_T$ |

另有 `--teacher_model_server URL`：teacher logp 走远程 API（vLLM/swift deploy），适合大 teacher；`--offload_teacher_model true` 把 teacher 换到 CPU 省显存。

### 7.4 两个附加项

- **`--sft_alpha α`**：$\mathcal{L}\leftarrow\mathcal{L}_{\text{GKD}}+\alpha\,\mathcal{L}_{\text{SFT}}$，只对 **STUDENT 生成**的样本生效（`gkd_trainer.py:191-192`）；与 Liger 融合损失不兼容（会断言）；
- **`--gkd_logits_topk k`**：只用 teacher 的 top-k logits 算 KL（省显存），配合 `teacher_model_server` 时受 vLLM `--max-logprobs` 限制（默认 20）。

**归一化**：`_compute_jsd_loss` 里 `total / num_valid`，即在有效 token 上取平均，不是 batch 平均（`gkd_trainer.py:105-114`）。

> GKD 与 GRPO 共享同一套 rollout 基础设施（`RolloutTrainerMixin`），因此也支持 vLLM 加速。

---

## 8. 算法选型与显存/工程要点

### 8.1 选型决策

| 场景 | 首选 | 理由 |
|---|---|---|
| 有成对偏好数据、想最小改动 | `dpo` + `--rpo_alpha` | 稳定、无需采样；RPO 项保住生成能力 |
| 显存紧张、只有成对数据 | `simpo` / `cpo` / `orpo` | 无需 ref 模型，省一整份权重 |
| 只有「好/坏」单条标注 | `kto` | 不需要配对 |
| 数学/代码这类**可验证**任务 | `grpo` + `accuracy` 奖励 | 组采样 baseline，效果最好的一类 |
| 长链推理、答案方差大 | `grpo --loss_type dapo --epsilon_high 0.28 --dynamic_sample` | 非对称裁剪 + 动态采样 |
| 长序列、$\rho$ 噪声爆炸 | `--importance_sampling_level sequence`（GSPO） | 序列级比，裁剪语义更一致 |
| 多奖励量纲不一致 | `--scale_rewards gdpo` | 逐函数归一化 |
| 有更强 teacher、想稠密奖励 | GRPO + `--teacher_model`（OPD-RL）/ `--advantage_reweight rlsd` | 逐 token 蒸馏信号 |
| 纯蒸馏（无奖励） | `gkd` | JSD + on-policy 混合 |
| 需要给别处提供打分器 | `rm` | Bradley–Terry |

### 8.2 显存账（单卡粗估，LoRA）

| 算法 | 模型副本 | 备注 |
|---|---|---|
| dpo | 2（policy + ref） | LoRA 时 ref 可 `disable_adapter` 复用，约 1 份 |
| cpo/simpo/orpo | 1 | 最省 |
| kto | 2 | `--precompute_ref_log_probs` 可省时间不省显存 |
| rm | 1（+ 可选 ref 无） | 输出标量 |
| ppo | 4（policy/value/reward/ref） | 最贵 |
| grpo | 2（policy + ref，β=0 时 1）+ vLLM | vLLM 与训练可 colocate |
| gkd | 1~2（student + 可选 teacher） | 自蒸馏冻结模式仅 1 份 |

### 8.3 踩过的坑（本机实测）

1. **`--num_generations` 必须 > 1**（GRPO），否则直接 ValueError（`rlhf_args.py:409-410`）。
2. **`scale_rewards='gdpo'` 需要 `reward_funcs`**；纯蒸馏（无奖励）会报错。
3. **OPD-RL 与 `real`/`fipo`/`off_policy_sequence_mask` 不兼容**：这些损失需要序列级 advantage，而 teacher 给的是 token 级信号（`rlhf_args.py:684-690`）。
4. **RLSD 与 `use_liger_kernel` 互斥**：Liger 的融合 loss 会绕过逐 token 重加权路径。SDAR 同理。
5. **RLSD 必须给 `reward_funcs`**：重加权方向依赖 $\text{sign}(A)$。
6. **`--beta 0` 时 GRPO 自动丢掉 ref_model**，但此时 KL 项恒为 0，长训容易跑飞——除非你用 teacher 蒸馏。
7. **多模态 GRPO 建议 `--truncation_strategy delete`**：左截断会切掉视觉 token 导致 forward 形状不匹配（`rlhf_args.py:156-162`）。
8. **CPO/SimPO/ORPO 传 `--ref_model` 会直接报错**（断言），别惯性复制 DPO 脚本。
9. **`--loss_type` 多值仅 DPO 支持**，且必须配 `--loss_weights`。
10. **DPO 的 `label_smoothing` 与多种 loss_type 冲突**，会被静默忽略（只告警）。
11. **`--use_vllm false`** 时走原生 generate（本机 `rl_grpo_2gpu.sh` 就是这么跑的，不需要装 vllm）。
12. **GRPO 默认 `gradient_accumulation_steps=1`**（`rlhf_args.py:528-531`），因为一个 generation batch 会被切成 `steps_per_generation` 份。

---

## 9. 参数速查表

### 9.1 总开关

```bash
swift rlhf \
  --rlhf_type {dpo|cpo|simpo|orpo|kto|rm|ppo|grpo|gkd} \
  --model <路径> --tuner_type lora --torch_dtype bfloat16 \
  --dataset <数据集> --max_length 2048 --max_completion_length 512
```

### 9.2 各算法默认值对照（`rlhf_args.py:192-267`）

| 参数 | 默认 | 生效算法 |
|---|---|---|
| `beta` | simpo 2.0 / grpo 0.04 / gkd 0.5 / 其它 0.1 | 全部 |
| `label_smoothing` | 0 | dpo（cpo 亦可） |
| `rpo_alpha` | None | dpo |
| `ld_alpha` | None | dpo |
| `discopop_tau` | 0.05 | dpo(`discopop`) |
| `cpo_alpha` | 1.0 | cpo / simpo |
| `simpo_gamma` | 1.0 | simpo |
| `desirable_weight` / `undesirable_weight` | 1.0 / 1.0 | kto |
| `center_rewards_coefficient` | None | rm |
| `temperature` | 0.9 | ppo / grpo / gkd |
| `sft_alpha` / `lmbda` | 0 / 0.5 | gkd |
| `gkd_logits_topk` | None（全词表） | gkd |

### 9.3 GRPO 专项

| 参数 | 默认 | 说明 |
|---|---|---|
| `num_generations` | 8 | 组大小 $G$ |
| `num_iterations` | 1 | 每批更新次数 $\mu$ |
| `epsilon` / `epsilon_high` | 0.2 / =epsilon | 裁剪区间 |
| `delta` | None | Dual-Clip 上界 |
| `beta` | 0.04 | KL 系数 |
| `kl_in_reward` | grpo:False / rloo,reinforce++:True | KL 放在奖励还是损失 |
| `advantage_estimator` | grpo | grpo / rloo / reinforce_plus_plus |
| `scale_rewards` | group | group / batch / none / gdpo |
| `loss_type` | grpo | grpo / bnpo / dr_grpo / dapo / cispo / sapo / real / fipo |
| `importance_sampling_level` | token | token / sequence / sequence_token |
| `dynamic_sample` / `max_resample_times` | False / 3 | DAPO 动态采样 |
| `overlong_filter` | False | 剔除截断样本 |
| `top_entropy_quantile` | 1.0（关） | 熵掩码 |
| `tau_pos` / `tau_neg` | 1.0 / 1.05 | SAPO 门温度 |
| `real_tau` | 0.5 | REAL 温度 |
| `fipo_decay_rate` / `fipo_clip_range` / `fipo_safety_threshold` | 32 / 0.2 / 4.0 | FIPO |
| `rollout_importance_sampling_mode` / `threshold` | None / 2.0 | 离策略纠偏 |
| `off_policy_sequence_mask_delta` | None | 序列级 mask |
| `teacher_kl_coef` | 1.0 | OPD-RL |
| `advantage_reweight` / `rlsd_lambda` / `rlsd_reweight_clip_range` | None / 0.5 / 0.2 | RLSD |
| `sdar_loss_coef` / `sdar_gate_beta` | 0（关）/ 5.0 | SDAR |
| `chord_mu_*` / `chord_enable_phi_function` | None / False | CHORD |

### 9.4 本机 `experiments/rl/` 脚本对照

| 脚本 | 算法 | 关键点 |
|---|---|---|
| `rl_dpo_2gpu.sh` | dpo | 需要 ref 模型 |
| `rl_cpo_2gpu.sh` | cpo | `--cpo_alpha 1.0`，无 ref |
| `rl_orpo_2gpu.sh` | orpo | 无 ref，单阶段 |
| `rl_simpo_2gpu.sh` | simpo | `--simpo_gamma`，无 ref |
| `rl_kto_2gpu.sh` | kto | 单条标注数据 |
| `rl_rm_2gpu.sh` | rm | 奖励模型 |
| `rl_grpo_2gpu.sh` | grpo | `--use_vllm false` 原生采样 + `accuracy` 奖励（gsm8k） |
| `rl_grpo_ds_2gpu.sh` | grpo | DeepSpeed 版本 |

> 补全建议：GRPO 的变体目前没有独立脚本，若要做消融，可以按 `--loss_type dapo` / `--importance_sampling_level sequence` / `--dynamic_sample true` 各加一份，对照同一随机种子。

---

## 10. 源码索引

| 内容 | 文件:位置 |
|---|---|
| `rlhf_type` 枚举 | `swift/arguments/rlhf_args.py:233` |
| β / loss_type / cpo_alpha / simpo_gamma 等定义 | `swift/arguments/rlhf_args.py:240-267` |
| 各算法 β 默认值 | `swift/arguments/rlhf_args.py:488-527` |
| GRPO 默认（beta/kl_in_reward/scale_rewards） | `swift/arguments/rlhf_args.py:340-393` |
| GRPO 全部算法开关定义 | `swift/rlhf_trainers/args_mixin.py:368-486` |
| DPO 14 种损失 | `swift/rlhf_trainers/dpo_trainer.py:263-358` |
| DPO LD-DPO / RPO / 指标 | `swift/rlhf_trainers/dpo_trainer.py:157-205, 419-437` |
| CPO 参数搬运 | `swift/rlhf_trainers/cpo_trainer.py` |
| ORPO | `swift/rlhf_trainers/orpo_trainer.py` |
| KTO forward / KL | `swift/rlhf_trainers/kto_trainer.py:50-130` |
| RM 损失 | `swift/rlhf_trainers/reward_trainer.py:38-76` |
| PPO 参数 | `swift/arguments/rlhf_args.py:114-127`；`swift/rlhf_trainers/ppo_trainer.py` |
| GRPO 策略损失 | `swift/rlhf_trainers/grpo_trainer.py:1036-1134` |
| GRPO KL（k3） | `swift/rlhf_trainers/grpo_trainer.py:972-976` |
| GSPO 重要性采样粒度 | `swift/rlhf_trainers/grpo_trainer.py:1008-1025` |
| FIPO 影响权重 | `swift/rlhf_trainers/grpo_trainer.py:885-932` |
| DAPO 动态采样 | `swift/rlhf_trainers/grpo_trainer.py:627-677` |
| rollout IS 四模式 | `swift/rlhf_trainers/grpo_trainer.py:2345-2393` |
| off-policy 序列 mask | `swift/rlhf_trainers/grpo_trainer.py:2395-2435` |
| advantage 估计器 | `swift/rl_core/advantage.py:20-99` |
| OPD-RL teacher 信号 | `swift/rl_core/advantage.py:241-291` |
| RLSD 重加权 | `swift/rl_core/advantage.py:294-343` |
| SDAR 门控蒸馏 | `swift/rl_core/advantage.py:362-420` |
| CHORD 损失 | `swift/rlhf_trainers/utils.py:compute_chord_loss` |
| GKD JSD | `swift/rlhf_trainers/gkd_loss.py:90-153, 224-271` |
| GKD on-policy 混合与 sft_alpha | `swift/rlhf_trainers/gkd_trainer.py:105-192, 299-314` |
| 内置奖励函数 | `swift/rewards/orm.py` |

### 参考文献（源码注释里引的原始论文）

- DPO — arXiv:2305.18290；IPO — 2310.12036；cDPO(label smoothing) — 2307.16892
- CPO — 2401.08417；SimPO — 2405.14734；ORPO — 2403.07691；KTO — 2402.01306
- GRPO — 2402.03300；RLOO — 2402.14740；REINFORCE++ — 2501.03262
- DAPO — 2503.14476；Dr.GRPO — 2503.20783；GSPO — 2507.18071；SAPO — 2511.20347
- CISPO、REAL — 2602.05630；FIPO — 2603.19835；GDPO — 2601.05242
- 熵掩码 "Beyond the 80/20 Rule" — 2506.01939
- DiscoPOP — 2406.08414；LD-DPO — 2410.15595；RPO — 2404.19733
- GKD — 2306.13649；CHORD — 2508.11408；RLSD — 2604.03128；SDAR — 2605.15155
- Off-Policy Sequence Masking (DeepSeek-V3.2) — 2512.02556

---

*笔记生成于 2026-11-08，对应本机 `/root/ms-swift` 当前工作副本；若上游重构了 `rlhf_trainers/`，以源码为准。*
