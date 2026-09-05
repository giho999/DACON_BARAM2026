#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W2CHAIN 단계별 추적 — blend 직후 / iso 직후 / shift 직후 / M1 직후 / coordBA 직후
TOTAL을 W0 vs W2(seed42)에서 각각 산출, 어느 단계에서 역전(중복)이 발생하는지 국소화."""
import os, sys, pickle, importlib.util, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AP = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location("cnn_common", os.path.join(AP, "cnn_common.py"))
cnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cnn)
score_fn = cnn.score_fn
blend_with_fallback = cnn.blend_with_fallback
W_CNN = cnn.W_CNN
CAPS = cnn.CAPS

CAL_S, CAL_E = pd.Timestamp("2024-09-01"), pd.Timestamp("2024-11-01")
EV_S, EV_E = pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-31 23:59:59")
BAND_LO, BAND_HI = 0.03, 0.15
COORD_CF_LO, COORD_CF_HI = 0.7, 1.0
COORD_SHIFT_CAP = 0.010
COORD_JJA = {6, 7, 8}
CNN_PRED = os.path.join(AP, "frozen_cnn_pred_2024_s42.parquet")


def competition(y, p, cap):
    v = (~np.isnan(y)) & (y >= 0.1 * cap)
    yt, pt = y[v], p[v]
    if len(yt) == 0:
        return 0.0
    nmae = float(np.clip(np.mean(np.abs(yt - pt)) / cap, 0, 1))
    e = np.abs(yt - pt) / cap
    up = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
    fi = float(np.clip((yt * up).sum() / max((yt * 4.0).sum(), 1e-9), 0, 1))
    return 0.5 * (1 - nmae) + 0.5 * fi


def tot(y3, p3):
    return float(np.mean([competition(y3[:, g], p3[:, g], CAPS[g]) for g in range(3)]))


def main():
    with open(os.path.join(HERE, "frozen_constants.pkl.isos"), "rb") as f:
        isos = pickle.load(f)
    with open(os.path.join(HERE, "frozen_constants.pkl"), "rb") as f:
        c = pickle.load(f)
    shifts, prod_shift = c["shifts"], c["prod_shift"]

    frozen = pd.read_parquet(CNN_PRED)
    frozen["t"] = pd.to_datetime(frozen["t"])

    out = {}
    for V in ["W0", "W2"]:
        df = pd.read_parquet(os.path.join(AP, "metricw", f"pred_{V}_seed42.parquet"))
        df = df.sort_values("t").reset_index(drop=True)
        gbdt = df[["pred_0", "pred_1", "pred_2"]].values.astype(np.float64)
        y = df[["y_0", "y_1", "y_2"]].values.astype(np.float64)
        t = df["t"].values
        t_arr = pd.to_datetime(t)
        fz = frozen.set_index("t").reindex(t_arr)
        vm = fz["cnn_pred_0"].notna().values
        cn = fz.loc[vm, ["cnn_pred_0", "cnn_pred_1", "cnn_pred_2"]].values.astype(np.float64)
        st = {}
        st["blend"] = tot(y, blend_with_fallback(gbdt, cn, vm, W_CNN))
        ev = blend_with_fallback(gbdt, cn, vm, W_CNN).copy()
        for g in range(3):
            ev[:, g] = np.clip(isos[g].transform(ev[:, g]), 0, CAPS[g])
        st["iso"] = tot(y, ev)
        for g in range(3):
            ev[:, g] = np.clip(ev[:, g] + shifts[g] * CAPS[g], 0, CAPS[g])
        st["shift"] = tot(y, ev)
        ev_m1 = ev.copy()
        for g in range(3):
            cap = CAPS[g]
            in_b = (ev[:, g] >= BAND_LO * cap) & (ev[:, g] <= BAND_HI * cap)
            ev_m1[in_b, g] = np.clip(ev[in_b, g] + prod_shift[g] * cap, 0, cap)
        st["m1"] = tot(y, ev_m1)
        mon = t_arr.month
        cf1 = ev_m1[:, 0] / CAPS[0]
        sel = (cf1 >= COORD_CF_LO) & (cf1 <= COORD_CF_HI) & (~np.isin(mon, list(COORD_JJA)))
        ev_m1[sel, 0] = np.clip(ev_m1[sel, 0] + COORD_SHIFT_CAP * CAPS[0], 0, CAPS[0])
        st["coordBA"] = tot(y, ev_m1)
        out[V] = st

    print(f"{'단계':10s} {'W0':>10s} {'W2':>10s} {'Δ(W2-W0)':>10s}")
    for k in ["blend", "iso", "shift", "m1", "coordBA"]:
        d = out["W2"][k] - out["W0"][k]
        print(f"{k:10s} {out['W0'][k]:10.5f} {out['W2'][k]:10.5f} {d:+10.5f}")


if __name__ == "__main__":
    main()
