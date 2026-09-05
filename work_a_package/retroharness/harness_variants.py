#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-RETROHARNESS — harness_variants: 2024 대응 후보 생성 + 5종 검증 방식
==========================================================================
2024 기준 base = 원본 v7 GBDT(2024 홀드아웃) + CNN → 동결 체인(iso/shift/M1/coordBA).
각 재현 가능 후보의 transform을 2024 base에 재적용 → 2024 대응 후보 예측.

재현 가능 후보 (transform 역분석 기반):
  C_g3tail : coordBA + G3 tail(λ1.20, a0.70)          — make_tail_probe 로직
  C_m1v2   : coordBA 제거 (M1 밴드 [0.03,0.15] 유지)   — chain에서 coordBA 단계 생략
  C_band  : M1 밴드 [0.05,0.20] 원본 (coordBA 포함)    — M1 재적합
  C_g3shift: coordBA 제거 + G3 +0.03cap                — const add

하네스 5종: V_A rolling / V_B 연도전이 / V_C 적대가중 / V_D 풍속정합 / V_E 다년풀
각 방식은 2024 평가에서 (후보 Δ vs base)를 산출. 평가식 = 대회 원본(10% 마스크+밴드+발전량가중).
"""
import os, sys, pickle, importlib.util, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
W2C = os.path.join(os.path.dirname(HERE), "w2chain")
AP = os.path.dirname(os.path.dirname(HERE))
PROJ = os.path.dirname(AP)
sys.path.insert(0, W2C)
import chain_frozen as cf

BAND_LO_O, BAND_HI_O = 0.03, 0.15   # 현재 체인 M1 밴드
BAND_LO_ORIG, BAND_HI_ORIG = 0.05, 0.20  # bandcorrected 원본 밴드


def load_const():
    with open(os.path.join(W2C, "frozen_constants.pkl.isos"), "rb") as f:
        isos = pickle.load(f)
    with open(os.path.join(W2C, "frozen_constants.pkl"), "rb") as f:
        c = pickle.load(f)
    return dict(isos=isos, **c)


def build_2024_base():
    """2024 기준 base = v7 GBDT + CNN → iso/shift/M1 + coordBA (submission_A_m1v2_coordBA의 2024 대응)."""
    const = load_const()
    # m1_holdout_preds_2024.parquet = 원본 v7 GBDT 2024 예측 (clean-downweight 0.3)
    df = pd.read_parquet(os.path.join(AP, "m1_holdout_preds_2024.parquet")).sort_values("t").reset_index(drop=True)
    gbdt = df[["pred_0", "pred_1", "pred_2"]].values.astype(np.float64)
    y = df[["y_0", "y_1", "y_2"]].values.astype(np.float64)
    t = df["t"].values
    chained = cf.apply_chain(gbdt, const, t)  # iso→shift→M1→coordBA
    return gbdt, chained, y, t


def g3tail_transform(pred, anchor=0.70, lam=1.20):
    """G3 tail 확장 (make_tail_probe 로직 — 변경 행만 재포맷 불필요, 수치만)."""
    out = pred.copy()
    c3 = out[:, 2] / cf.CAPS[2]
    sel = c3 > anchor
    out[sel, 2] = np.clip(anchor + lam * (c3[sel] - anchor), 0, 1) * cf.CAPS[2]
    return out


def chain_without_coordBA(gbdt, const, t):
    """체인에서 coordBA 단계 제거 (= A_m1v2의 2024 대응)."""
    stages = {}
    out = cf.apply_chain(gbdt, const, t, stages_out=stages)
    return stages["m1"]  # coordBA 직전 = M1까지


def chain_with_orig_band(gbdt, const, t):
    """M1 밴드 [0.05,0.20] 원본으로 재적합 (coordBA 포함)."""
    # iso → shift까지는 동일, M1만 밴드 [0.05,0.20]으로 재적합
    stages = {}
    cf.apply_chain(gbdt, const, t, stages_out=stages)  # 상수 동일
    ev = stages["shift"].copy()
    y = pd.read_parquet(os.path.join(AP, "m1_holdout_preds_2024.parquet")).sort_values("t")[["y_0","y_1","y_2"]].values
    tarr = pd.to_datetime(t)
    em = (tarr >= pd.Timestamp("2024-11-01")) & (tarr <= pd.Timestamp("2024-12-31 23:59:59"))
    # M1 shift 재적합 ([0.05,0.20] 밴드, eval self-ref — run.py fit_band 동일)
    BAND_GRID = np.arange(-0.10, 0.101, 0.005)
    prod = np.zeros(3)
    for g in range(3):
        cap = cf.CAPS[g]
        in_b = (ev[em, g] >= BAND_LO_ORIG * cap) & (ev[em, g] <= BAND_HI_ORIG * cap)
        best = (-1e9, 0.0)
        for sp in BAND_GRID:
            tt = ev.copy(); tt[em, g] = np.clip(ev[em, g] + sp * cap, 0, cap)
            tot, _, _ = cf.score_fn(y[em], tt[em])
            if tot > best[0]: best = (tot, sp)
        prod[g] = best[1]
    out = ev.copy()
    for g in range(3):
        cap = cf.CAPS[g]
        in_b = (ev[:, g] >= BAND_LO_ORIG * cap) & (ev[:, g] <= BAND_HI_ORIG * cap)
        out[in_b, g] = np.clip(ev[in_b, g] + prod[g] * cap, 0, cap)
    # coordBA
    mon = tarr.month
    cf1 = out[:, 0] / cf.CAPS[0]
    sel = (cf1 >= 0.7) & (cf1 <= 1.0) & (~np.isin(mon, [6,7,8]))
    out[sel, 0] = np.clip(out[sel, 0] + 0.010 * cf.CAPS[0], 0, cf.CAPS[0])
    return out


def g3shift_transform(pred):
    out = pred.copy()
    out[:, 2] = np.clip(out[:, 2] + 0.03 * cf.CAPS[2], 0, cf.CAPS[2])
    return out


def candidates_2024():
    gbdt, base_chained, y, t = build_2024_base()
    const = load_const()
    cands = {
        "base": base_chained,
        "C_g3tail": g3tail_transform(base_chained),
        "C_m1v2": chain_without_coordBA(gbdt, const, t),
        "C_band": chain_with_orig_band(gbdt, const, t),
        "C_g3shift": g3shift_transform(chain_without_coordBA(gbdt, const, t)),
    }
    return cands, y, t


# ── 하네스 평가 ─────────────────────────────────────────────────────
def total_metric(y3, p3):
    return cf.total_metric(y3, p3)[0]


def harness_VA_rolling(y, t, base_p, cand_p):
    """rolling 12창 pooled — 월별로 쪼개 각 월 eval로 채점해 평균(1-NMAE·FICR 그룹평균 TOTAL).
    단순화: 2024 12개월 각각 eval → TOTAL 평균 (V_A 정의)."""
    tarr = pd.to_datetime(t)
    scores_b, scores_c = [], []
    for m in range(1, 13):
        sel = tarr.month == m
        if sel.sum() < 10:
            continue
        scores_b.append(total_metric(y[sel], base_p[sel]))
        scores_c.append(total_metric(y[sel], cand_p[sel]))
    return float(np.mean(scores_c) - np.mean(scores_b))


def harness_VB_yeartransfer(y, t, base_p, cand_p):
    """연도전이: 2024 전체 평가 (학습 2022-23 — base/cand 모두 2024 예측)."""
    return total_metric(y, cand_p) - total_metric(y, base_p)


def harness_VC_adversarial(y, t, base_p, cand_p, w):
    """적대적 가중: 각 행에 w_i, Σw=n. 가중 TOTAL = 그룹평균 가중 FICR."""
    def wtotal(y3, p3, wgt):
        t = []
        for g in range(3):
            v = (~np.isnan(y3[:, g])) & (y3[:, g] >= 0.1 * cf.CAPS[g])
            yt, pt, wv = y3[v, g], p3[v, g], wgt[v]
            nmae = float(np.clip(np.average(np.abs(yt - pt), weights=wv) / cf.CAPS[g], 0, 1))
            e = np.abs(yt - pt) / cf.CAPS[g]
            up = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
            fi = float(np.clip((yt * up * wv).sum() / max((yt * 4.0 * wv).sum(), 1e-9), 0, 1))
            t.append(0.5 * (1 - nmae) + 0.5 * fi)
        return float(np.mean(t))
    return wtotal(y, cand_p, w) - wtotal(y, base_p, w)


def harness_VD_windquantile(y, t, base_p, cand_p, w_resample):
    """풍속분위 정합: 2024 행을 2025 분위 구조로 재표집 가중 w_resample (Σw=n)."""
    return harness_VC_adversarial(y, t, base_p, cand_p, w_resample)


def harness_VE_multiyear(y, t, base_p, cand_p, w_pool=None):
    """다년 균등 풀: 2024 균등 (2022/23 예측 부재 — V_E는 2024 균등과 동일)."""
    return harness_VB_yeartransfer(y, t, base_p, cand_p)


if __name__ == "__main__":
    cands, y, t = candidates_2024()
    print("2024 대응 후보 생성 완료:")
    for k, v in cands.items():
        print(f"  {k:10s} shape={v.shape} max={v.max():.1f}")
    np.savez_compressed(os.path.join(HERE, "cands_2024.npz"), **{k: v for k, v in cands.items()},
                        y=y, t=t.astype("datetime64[ns]").astype(np.int64))
    print("저장: cands_2024.npz")
