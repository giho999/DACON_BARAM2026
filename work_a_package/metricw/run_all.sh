#!/bin/bash
export OMP_NUM_THREADS=16
cd /home/gpu_04/DACON_baram2026
PY=/home/gpu_04/.conda/envs/lgaimers/bin/python3
for V in W0 W1 W2; do
  for S in 42 1337 2024; do
    $PY work_a_package/metricw/train_metricw.py --variant $V --seed $S > work_a_package/metricw/log_${V}_seed${S}.log 2>&1
    echo "[done] $V $S rc=$? $(tail -1 work_a_package/metricw/log_${V}_seed${S}.log)"
  done
done
$PY work_a_package/metricw/train_metricw.py --variant W2S --seed 42 > work_a_package/metricw/log_W2S_seed42.log 2>&1
echo "[done] W2S 42 rc=$? $(tail -1 work_a_package/metricw/log_W2S_seed42.log)"
echo "=== ALL DONE ==="
