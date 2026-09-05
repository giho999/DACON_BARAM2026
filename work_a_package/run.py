"""
run.py — 작업물 A 재현 (M1 밴드 [0.03,0.15] 적용, self-contained)

팀원 원본 submission_ficr_w1_v7_cnn(0.65183).csv 에
M1 밴드 보정 밴드만 [0.05,0.20] → [0.03,0.15] 로 바꾸고 shift 재적합.

필요 파일 (같은 폴더에 둘 것):
  - m1_holdout_preds_2024.parquet   (v7 GBDT 2024 홀드아웃 예측)
  - frozen_cnn_pred_2024_s42.parquet (동결 CNN seed42)
  - submission_ficr_w1_v7_cnn(0.65183).csv (팀원 원본 제출파일)
  - cnn_common.py                    (blend_with_fallback, score_fn)

실행: python run.py
출력: submission_A.csv
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys_path = os.path.join(HERE, "cnn_common.py")  # noqa - import below
import importlib.util
spec = importlib.util.spec_from_file_location("cnn_common", os.path.join(HERE, "cnn_common.py"))
cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn)
score_fn = cnn.score_fn
blend_with_fallback = cnn.blend_with_fallback
W_CNN = cnn.W_CNN
CAPS = cnn.CAPS

# ── constants ──────────────────────────────────────────────────────
CAL_S, CAL_E   = pd.Timestamp("2024-09-01"), pd.Timestamp("2024-11-01")
EV_S,  EV_E    = pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-31 23:59:59")
SHIFT_GRID     = np.arange(-0.08, 0.09, 0.01)        # calibrate_total
BAND_GRID      = np.arange(-0.10, 0.101, 0.005)      # M1 세분화 그리드
BAND_LO, BAND_HI = 0.03, 0.15                        # 선택 밴드

# ── 1. calibrate_total (iso+shift, 팀 관례) ──────────────────────
def calibrate_total(cal_p, ycal, ev_p):
    cal_c, ev_c = cal_p.copy(), ev_p.copy()
    for g in range(3):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(cal_p[:, g], ycal[:, g])
        cal_c[:, g] = np.clip(iso.transform(cal_p[:, g]), 0, CAPS[g])
        ev_c[:, g] = np.clip(iso.transform(ev_p[:, g]), 0, CAPS[g])
    shifts = np.zeros(3)
    for g in range(3):
        best = (-1e9, 0.0)
        for sp in SHIFT_GRID:
            t = cal_c.copy(); t[:, g] = np.clip(cal_c[:, g] + sp * CAPS[g], 0, CAPS[g])
            tot, _, _ = score_fn(ycal, t)
            if tot > best[0]: best = (tot, sp)
        shifts[g] = best[1]
    for g in range(3):
        ev_c[:, g] = np.clip(ev_c[:, g] + shifts[g] * CAPS[g], 0, CAPS[g])
    return ev_c, shifts

# ── 2. M1 밴드 보정 ──────────────────────────────────────────────
def fit_band(cal_c, ycal, lo, hi, grid):
    shift = np.zeros(3)
    for g in range(3):
        cap = CAPS[g]
        in_b = (cal_c[:, g] >= lo * cap) & (cal_c[:, g] <= hi * cap)
        best = (-1e9, 0.0)
        for sp in grid:
            t = cal_c.copy(); t[in_b, g] = np.clip(cal_c[in_b, g] + sp * cap, 0, cap)
            tot, _, _ = score_fn(ycal, t)
            if tot > best[0]: best = (tot, sp)
        shift[g] = best[1]
    return shift

def apply_band(ev_c, shift, lo, hi):
    out = ev_c.copy()
    for g in range(3):
        cap = CAPS[g]
        in_b = (out[:, g] >= lo * cap) & (out[:, g] <= hi * cap)
        out[in_b, g] = np.clip(out[in_b, g] + shift[g] * cap, 0, cap)
    return out

# ── 3. main ──────────────────────────────────────────────────────
def main():
    # 3a. v7 GBDT + frozen CNN 블렌드
    base = pd.read_parquet(os.path.join(HERE, "m1_holdout_preds_2024.parquet"))
    base["t"] = pd.to_datetime(base["t"]); base = base.sort_values("t").reset_index(drop=True)
    frozen = pd.read_parquet(os.path.join(HERE, "frozen_cnn_pred_2024_s42.parquet"))
    frozen["t"] = pd.to_datetime(frozen["t"])
    m = base.merge(frozen, on="t", how="left")
    vm = m["cnn_pred_0"].notna().values
    cn = m.loc[vm, ["cnn_pred_0","cnn_pred_1","cnn_pred_2"]].values.astype(np.float64)
    bp = m[["pred_0","pred_1","pred_2"]].values.astype(np.float64)
    blended = blend_with_fallback(bp, cn, vm, W_CNN)
    y_all = m[["y_0","y_1","y_2"]].values.astype(np.float64)
    t_all = m["t"].values

    # 3b. calibrate (cal 09-10 → eval 11-12)
    cm = (t_all >= CAL_S) & (t_all < CAL_E)
    em = (t_all >= EV_S) & (t_all <= EV_E)
    print(f"[cal] 09-10={cm.sum()}행  [eval] 11-12={em.sum()}행")
    ev_c, cal_shifts = calibrate_total(blended[cm], y_all[cm], blended[em])

    # 3c. baseline 체크
    tot_base = score_fn(y_all[em], ev_c)[0]
    print(f"캘리만 FICR: {tot_base:.5f}")

    # 3d. M1 밴드 적합 (self-ref, 팀 관례: cal=11-12 자기자신)
    prod_shift = fit_band(ev_c, y_all[em], BAND_LO, BAND_HI, BAND_GRID)
    print(f"선택 밴드 [{BAND_LO},{BAND_HI}] shift: G1={prod_shift[0]:.3f} G2={prod_shift[1]:.3f} G3={prod_shift[2]:.3f}")

    ev_m1 = apply_band(ev_c, prod_shift, BAND_LO, BAND_HI)
    tot_m1 = score_fn(y_all[em], ev_m1)[0]
    print(f"M1 적용 FICR: {tot_m1:.5f}  (diff +{tot_m1 - tot_base:.5f})")

    # 3e. 제출파일 적용
    sub = pd.read_csv(os.path.join(HERE, "submission_ficr_w1_v7_cnn(0.65183).csv"))
    pred = sub[["kpx_group_1","kpx_group_2","kpx_group_3"]].values.astype(np.float64)
    pred_c = pred.copy()
    for g in range(3):
        cap = CAPS[g]
        in_b = (pred[:, g] >= BAND_LO * cap) & (pred[:, g] <= BAND_HI * cap)
        pred_c[in_b, g] = np.clip(pred[in_b, g] + prod_shift[g] * cap, 0, cap)

    out = sub.copy()
    out["kpx_group_1"], out["kpx_group_2"], out["kpx_group_3"] = pred_c[:, 0], pred_c[:, 1], pred_c[:, 2]
    out_path = os.path.join(HERE, "submission_A.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")

    # 검증
    assert out.shape == (8760, 5)
    assert (out[["kpx_group_1","kpx_group_2","kpx_group_3"]].values >= 0).all()
    assert (out["kpx_group_1"] <= CAPS[0]).all() and (out["kpx_group_2"] <= CAPS[1]).all() and (out["kpx_group_3"] <= CAPS[2]).all()
    assert (out["forecast_id"] == sub["forecast_id"]).all()
    print(f"\n저장: {out_path}  shape={out.shape}  검증 통과 ✓")

if __name__ == "__main__":
    main()
