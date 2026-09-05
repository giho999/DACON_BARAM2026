"""
train_cnn_production.py — TASK5-C: cnn_gate/ 이식본(cnn_common.py)으로 실제 production 학습 재현.
2022~2024 전체 train으로 1회 학습(조기종료=내부 시간순 마지막 10%, v7_cnn_submission.py와 동일 방식),
seed=42(팀원 LB 제출과 동일 설정). 체크포인트만 저장 — 추론은 infer_cnn_production.py로 분리.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cnn_gate"))
import numpy as np
import pandas as pd
import torch

from cnn_common import TARGETS, CAPS, get_grid_tensors, train_cnn_holdout

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def main():
    t0 = time.time()
    tr = pd.read_parquet(os.path.join(PROJ, "cache_train.parquet"))
    tr["t"] = pd.to_datetime(tr["t"])
    tr = tr.sort_values("t").dropna(subset=TARGETS).reset_index(drop=True)
    tr_times = tr["t"].values
    print(f"전체 train 행={len(tr)}", flush=True)

    ldaps_all, gfs_all, valid_all = get_grid_tensors(tr_times)
    y_norm = (tr[TARGETS].values[valid_all]) / CAPS
    times_valid = tr_times[valid_all]
    print(f"그리드매칭={valid_all.sum()}/{len(tr)} ({valid_all.mean():.1%})", flush=True)

    model, ldaps_med, gfs_med, ns, best_val = train_cnn_holdout(
        ldaps_all, gfs_all, y_norm, times_valid, seed=SEED, val_frac=0.1)
    print(f"학습 완료 best_val_mse={best_val:.5f} ({time.time()-t0:.0f}s)", flush=True)

    ckpt_path = os.path.join(HERE, f"cnn_production_s{SEED}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "ldaps_ch": model.ldaps_branch.conv[0].in_channels,
        "gfs_ch": model.gfs_branch.conv[0].in_channels,
        "ldaps_med": ldaps_med, "gfs_med": gfs_med,
        "lmu": ns["lmu"], "lsd": ns["lsd"], "gmu": ns["gmu"], "gsd": ns["gsd"],
        "seed": SEED,
    }, ckpt_path)
    print(f"체크포인트 저장: {ckpt_path}  (총 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
