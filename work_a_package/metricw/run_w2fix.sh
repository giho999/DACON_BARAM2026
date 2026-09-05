#!/bin/bash
export OMP_NUM_THREADS=16
cd /home/gpu_04/DACON_baram2026
PY=/home/gpu_04/.conda/envs/lgaimers/bin/python3
for S in 42 1337 2024; do
  $PY work_a_package/metricw/train_metricw.py --variant W2 --seed $S > work_a_package/metricw/log_W2_seed${S}.log 2>&1
  echo "[done] W2 $S rc=$?"
done
$PY work_a_package/metricw/train_metricw.py --variant W2S --seed 42 > work_a_package/metricw/log_W2S_seed42.log 2>&1
echo "[done] W2S 42 rc=$?"
echo "=== W2 FIX DONE ==="
