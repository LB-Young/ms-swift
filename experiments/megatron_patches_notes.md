# ms-swift 环境补丁记录（Megatron TP 跑通所必需）

> 适用范围：`/opt/miniconda3/envs/ms-swift`（WSL Ubuntu-24.04，Python 3.12，torch 2.14.1 cuda134）
> 这些补丁打在 **site-packages 内**（megatron-core 与 mcore-bridge 的 wheel 安装文件），不在 ms-swift 源码里。
> 原因：本项目坚持"全预编译 wheel、零 nvcc 源码编译"路线；megatron-core 默认依赖 transformer-engine（需编译 CUDA 算子），在 2×RTX 5060 Ti (sm_120, Blackwell) 上无法编译，故彻底卸载 transformer-engine，改走 megatron-core 官方 `transformer_impl='local'` 原生算子，并对相关文件打补丁兜底。

## 1. 涉及文件（13 处）

### megatron-core（0.19.2）
路径前缀：`/opt/miniconda3/envs/ms-swift/lib/python3.12/site-packages/megatron/core/`

| 文件 | 补丁内容 |
|---|---|
| `transformer_engine.py` | try/except 兜底：TE 不存在时返回 None/默认值，不抛 ImportError |
| `transformer_config.py` | `transformer_impl` 默认改为 `'local'`；`use_transformer_engine` 按 config 判断 |
| `random.py` | 兼容无 TE 时的 RNG 状态管理（cuda_rng_tracker 兜底） |
| `jit.py` | `jit_fuser` 改为 noop（triton 3.8 PTX8.4 不支持 sm_120a，禁用 fusion） |

### mcore-bridge（1.6.5）
路径前缀：`/opt/miniconda3/envs/ms-swift/lib/python3.12/site-packages/mcore_bridge/`

| 文件 | 补丁内容 |
|---|---|
| `register.py` | 注册流程对缺 TE 场景兜底 |
| `gpt_bridge.py` | GPT 桥接使用 local transformer_impl |
| `deepseek_v41.py` | DeepSeek 架构桥接兜底 |
| `deepseek_v4.py` | 同上 |
| `gated_delta_net.py` | GatedDeltaNet 层兜底 |
| `compressor.py` | MTP 压缩器兜底 |
| `mtp_layer.py` | MTP 层兜底 |
| `dsa_indexer.py` | DSA 索引器兜底 |
| `lora.py` | LoRA 桥接兜底 |
| `model_config.py` | 模型配置兼容（persist_layer_norm True→False 等） |

## 2. 关键配置决策

1. **彻底卸载 transformer-engine**：`pip uninstall transformer-engine` + 删除 site-packages 残留目录。megatron 运行必须用 `transformer_impl='local'` 原生算子。
2. **脚本侧配合**：
   - `NPROC_PER_NODE=2`（TP 必须每卡一进程）
   - 去掉 `--max_steps` / `--sequence_parallel`（megatron 下不接受）
   - 加 `--padding_free false` / `--masked_softmax_fusion false`
   - `TRITON_PTXAS_PATH`、`TRITON_PTXAS_BLACKWELL_PATH` 指向 env/bin/ptxas
3. **Blackwell (sm_120) 三个坑**：
   - `cuda-cudart-dev` 补 cuda.h 并软链到 `$PREFIX/targets/x86_64-linux/include/`
   - `cuda-nvcc` 补 ptxas
   - triton 3.8 的 PTX8.4 不支持 sm_120a → 禁 jit_fuser

## 3. 验证结果

- 最终 **38/38 步跑通**：TP:2 PP:1、Column/RowParallelLinear(TP=2)、44s、6.31GiB/卡
- 产物：`experiments/output/sft-megatron/v12-20261019-143411/`
- 脚本：`experiments/sft_megatron_tp_2gpu.sh`

## 4. 环境恢复备忘

若环境重建，顺序：
1. conda env 装 torch 2.14.1(cuda134_mkl) + flash-attn 2.8.3 + triton 3.8.0（全预编译）
2. `pip install -e /root/ms-swift`（或本仓库）
3. `pip install 'megatron-core>=0.16' 'mcore-bridge>=1.6.3'`（实际 0.19.2 / 1.6.5）
4. 卸载 transformer-engine（pip + 删残留）
5. 按上表对 site-packages 打 13 处补丁
6. 用 `sft_megatron_tp_2gpu.sh` 冒烟验证

> 完整踩坑过程见同目录《ms-swift安装到训练全流程记录.md》第十章（Megatron）与第十四章（offload 尝试与环境恢复）。
