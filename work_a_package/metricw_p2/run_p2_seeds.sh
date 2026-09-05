#!/bin/bash
export OMP_NUM_THREADS=16
cd /home/gpu_04/DACON_baram2026
PY=/home/gpu_04/.conda/envs/lgaimers/bin/python3
$PY work_a_package/train_p2_maskprob.py --seed 1337 > work_a_package/metricw_p2/log_p2_s1337.log 2>&1
echo "[done] 1337 rc=$?"
$PY work_a_package/train_p2_maskprob.py --seed 2024 > work_a_package/metricw_p2/log_p2_s2024.log 2>&1
echo "[done] 2024 rc=$?"
echo "=== P2 SEEDS DONE ==="
