#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-W2CHAIN — 제출 후보 생성: submission_metricw_W2chain_frozen.csv
=====================================================================
- GBDT = W2 2025 test 3-seed 평균 (work_a_package/metricw/pred_test_W2_seed*.parquet)
- CNN = task56_dm/cnn_test_pred_2025.npz (seed42 동결 체크포인트, TASK14 결정성)
- 체인 = chain_frozen 동결 상수 (iso/shift/M1) + coordBA (G1 CF[0.7,1.0]∩비JJA +216)
- G2/G3 등 원본 유지 방식은 coordBA 규칙에 따라 적용. 즉시 제출 금지.
"""
import os, sys, pickle, importlib.util, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AP = os.path.dirname(HERE)
PROJ = os.path.dirname(AP)  # 프로젝트 루트 (/home/gpu_04/DACON_baram2026)
METRICW = os.path.join(AP, "metricw")
W2CHAIN = HERE
CNN_TEST = os.path.join(PROJ, "task56_dm", "cnn_test_pred_2025.npz")
SAMPLE = os.path.join(PROJ, "sample_submission.csv")
OUT = os.path.join(PROJ, "submission_metricw_W2chain_frozen.csv")

_spec = importlib.util.spec_from_file_location("cnn_common", os.path.join(AP, "cnn_common.py"))
cnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cnn)
CAPS = cnn.CAPS
W_CNN = cnn.W_CNN

BAND_LO, BAND_HI = 0.03, 0.15
COORD_CF_LO, COORD_CF_HI = 0.7, 1.0
COORD_SHIFT_CAP = 0.010
COORD_JJA = {6, 7, 8}


def main():
    with open(os.path.join(W2CHAIN, "frozen_constants.pkl.isos"), "rb") as f:
        isos = pickle.load(f)
    with open(os.path.join(W2CHAIN, "frozen_constants.pkl"), "rb") as f:
        c = pickle.load(f)
    shifts, prod_shift = c["shifts"], c["prod_shift"]

    # W2 test 3-seed 평균 GBDT
    gbdt = np.zeros((8760, 3))
    for S in [42, 1337, 2024]:
        df = pd.read_parquet(os.path.join(METRICW, f"pred_test_W2_seed{S}.parquet"))
        df = df.sort_values("t").reset_index(drop=True)
        gbdt += df[["pred_0", "pred_1", "pred_2"]].values / 3
    t_test = pd.to_datetime(df["t"].values)

    # CNN 동결 (test 2025)
    cn = np.load(CNN_TEST)
    cnn_pred = cn["cnn_pred"].astype(np.float64)
    cnn_t = pd.to_datetime(cn["forecast_kst_dtm"])
    # 정렬 확인
    assert (t_test.values == cnn_t.values).all(), "test t 불일치"

    # blend (w=0.20)
    blended = (1 - W_CNN) * gbdt + W_CNN * cnn_pred
    blended = np.clip(blended, 0, CAPS)

    # iso → shift → M1 (동결)
    ev = blended.copy()
    for g in range(3):
        ev[:, g] = np.clip(isos[g].transform(blended[:, g]), 0, CAPS[g])
    for g in range(3):
        ev[:, g] = np.clip(ev[:, g] + shifts[g] * CAPS[g], 0, CAPS[g])
    ev_m1 = ev.copy()
    for g in range(3):
        cap = CAPS[g]
        in_b = (ev[:, g] >= BAND_LO * cap) & (ev[:, g] <= BAND_HI * cap)
        ev_m1[in_b, g] = np.clip(ev[in_b, g] + prod_shift[g] * cap, 0, cap)
    # coordBA (동결)
    mon = t_test.month
    cf1 = ev_m1[:, 0] / CAPS[0]
    sel = (cf1 >= COORD_CF_LO) & (cf1 <= COORD_CF_HI) & (~np.isin(mon, list(COORD_JJA)))
    ev_m1[sel, 0] = np.clip(ev_m1[sel, 0] + COORD_SHIFT_CAP * CAPS[0], 0, CAPS[0])

    # 제출 형식
    ss = pd.read_csv(SAMPLE)
    out = ss.copy()
    out["kpx_group_1"], out["kpx_group_2"], out["kpx_group_3"] = (
        ev_m1[:, 0], ev_m1[:, 1], ev_m1[:, 2])
    # 문자열 포맷: base coordBA와 동일 형식 (소수 3자리 후행 0 제거)
    def fmt(v):
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-0") else "0"
    for col in ["kpx_group_1", "kpx_group_2", "kpx_group_3"]:
        out[col] = out[col].map(fmt)
    out.to_csv(OUT, index=False, encoding="utf-8")

    md5 = hashlib.md5(open(OUT, "rb").read()).hexdigest()
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    vals = out[["kpx_group_1", "kpx_group_2", "kpx_group_3"]].astype(float)
    print(f"저장: {OUT}")
    print(f"md5={md5}")
    print(f"sha256={sha}")
    print(f"shape={out.shape} · NaN={vals.isna().sum().sum()} · 음수={(vals < 0).sum().sum()}")
    print(f"cap초과: G1={(vals.kpx_group_1>21600).sum()} G2={(vals.kpx_group_2>21600).sum()} G3={(vals.kpx_group_3>21000).sum()}")
    print(f"G3 max_cf={(vals.kpx_group_3/21000).max():.4f} · G1 max={(vals.kpx_group_1).max():.1f}")


if __name__ == "__main__":
    main()
