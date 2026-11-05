#!/bin/bash
cd /root/ms-swift/experiments || exit 1
mkdir -p logs output
setsid bash cpt_vl_2gpu.sh > logs/cpt_vl_2gpu_wrapper.log 2>&1 &
echo "pid=$!"
sleep 5
echo "=== alive? ==="
ps -eo pid,args | grep -E 'cpt_vl_2gpu|torch.distributed' | grep -v grep | head -3 || echo "none"
echo "=== first output ==="
head -40 logs/cpt_vl_2gpu_wrapper.log 2>/dev/null || echo "no log yet"
