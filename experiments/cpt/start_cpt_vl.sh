#!/bin/bash
# 用途：后台启动 Qwen2-VL-2B 双卡 CPT（setsid 起进程并打印前 40 行日志，避免非交互 WSL 会话退出杀进程）
# 日志：experiments/logs/cpt_vl_2gpu_wrapper.log
# 运行：cd /root/ms-swift && bash experiments/cpt/start_cpt_vl.sh
cd "$(dirname "$0")/.." || exit 1   # 回到 experiments，保持 output/logs 相对路径语义
mkdir -p logs output
setsid bash cpt/cpt_vl_2gpu.sh > logs/cpt_vl_2gpu_wrapper.log 2>&1 &
echo "pid=$!"
sleep 8
echo "=== alive? ==="
ps -eo pid,args | grep -E 'cpt_vl_2gpu|torch.distributed' | grep -v grep | head -3 || echo "none"
echo "=== first output ==="
head -40 logs/cpt_vl_2gpu_wrapper.log 2>/dev/null || echo "no log yet"
