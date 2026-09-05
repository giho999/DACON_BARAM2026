#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-P2CHAIN — P2(마스크 확률 가중)의 동결 체인 중복 검정
===========================================================
지시서: opencode_TASK_P2CHAIN_20260809.md · 선등록: PREREG_p2chain_20260809.md

chain(X) = coordBA(M1(shift(iso(blend(GBDT=X, CNN=동결))))) 에서 X만 W0↔W2↔P2 스왑.
체인 상수·블렌드 비율·CNN 동결, 재적합 0회. 2024 연도전이 홀드아웃(학습 2022-23),
대회 평가식 원본, 3seed(42/1337/2024).

산출:
  - W0/W2 기준선 재현 (0.63402 / 0.62524) — 실패 시 즉시 중단
  - Δ_체인후(P2) 3seed 평균·seed별 · 중복률 r (Δ_체인전 3seed 평균 명기)
  - 단계별 trace (blend→iso→shift→M1→coordBA, P2 vs W2)
  - 그룹별 분해 (G1/G2/G3, 1/6 희석)
  - 캘리 창(09-10)·M1(11-12) × 2024 겹침 고지
"""
import os, sys, json, pickle, importlib.util, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
W2C = os.path.join(os.path.dirname(HERE), "w2chain")
AP = os.path.dirname(os.path.dirname(HERE))
P2DIR = os.path.join(os.path.dirname(HERE), "metricw_p2")
sys.path.insert(0, W2C)
import chain_frozen as cf

SEEDS = [42, 1337, 2024]


def load_chain():
    with open(os.path.join(W2C, "frozen_constants.pkl.isos"), "rb") as f:
        isos = pickle.load(f)
    with open(os.path.join(W2C, "frozen_constants.pkl"), "rb") as f:
        c = pickle.load(f)
    return dict(isos=isos, **c)


def chain_predict(gbdt_pred, const, t):
    stages = {}
    out = cf.apply_chain(gbdt_pred, const, t, stages_out=stages)
    return out, stages


def run_variant(variant, seed, const):
    if variant in ("W0", "W1", "W2"):
        p, y, t = cf.load_metricw_pred(variant, seed)
    elif variant == "P2":
        df = pd.read_parquet(os.path.join(P2DIR, f"pred_P2_2024_seed{seed}.parquet"))
        df = df.sort_values("t").reset_index(drop=True)
        p = df[["pred_0", "pred_1", "pred_2"]].values.astype(np.float64)
        y = df[["y_0", "y_1", "y_2"]].values.astype(np.float64)
        t = df["t"].values
    else:
        raise ValueError(variant)
    chained, stages = chain_predict(p, const, t)
    tot, nm, fi = cf.total_metric(y, chained)
    stage_totals = {k: cf.total_metric(y, v)[0] for k, v in stages.items()}
    return dict(TOTAL=tot, NMAE1=nm, FICR=fi, stages=stage_totals,
                group=[cf.competition_metric(y[:, g], chained[:, g], cf.CAPS[g]) for g in range(3)])


def main():
    const = load_chain()
    results = {}
    # 사전검증: W0/W2 기준선 재현
    for V in ["W0", "W2"]:
        for s in SEEDS:
            r = run_variant(V, s, const)
            results[f"{V}_{s}"] = r
    w0_mean = float(np.mean([results[f"W0_{s}"]["TOTAL"] for s in SEEDS]))
    w2_mean = float(np.mean([results[f"W2_{s}"]["TOTAL"] for s in SEEDS]))
    print(f"[사전검증] chain(W0) 3seed 평균 = {w0_mean:.5f} (기대 0.63402)")
    print(f"[사전검증] chain(W2) 3seed 평균 = {w2_mean:.5f} (기대 0.62524)")
    ok_w0 = abs(w0_mean - 0.63402) < 5e-4
    ok_w2 = abs(w2_mean - 0.62524) < 5e-4
    if not (ok_w0 and ok_w2):
        print(f"[사전검증] FAIL — 기준선 재현 실패 (W0={ok_w0} W2={ok_w2}). 즉시 중단.")
        return 1
    print("[사전검증] PASS")

    # P2 체인
    for s in SEEDS:
        r = run_variant("P2", s, const)
        results[f"P2_{s}"] = r
        print(f"[P2] seed{s}: TOTAL={r['TOTAL']:.5f} 1-NMAE={r['NMAE1']:.5f} FICR={r['FICR']:.5f}")

    # Δ_체인후
    d_p2 = [results[f"P2_{s}"]["TOTAL"] - results[f"W0_{s}"]["TOTAL"] for s in SEEDS]
    d_w2 = [results[f"W2_{s}"]["TOTAL"] - results[f"W0_{s}"]["TOTAL"] for s in SEEDS]
    m_p2, m_w2 = float(np.mean(d_p2)), float(np.mean(d_w2))
    npos = sum(1 for x in d_p2 if x > 0)
    print(f"\nΔ_체인후(P2) = {m_p2:+.5f} (seed별 {[f'{x:+.5f}' for x in d_p2]}, 양수 {npos}/3)")
    print(f"Δ_체인후(W2) = {m_w2:+.5f} (참조)")

    # Δ_체인전(P2) — P2 체인 전 (blend 직전 GBDT 단독? 아니면 체인 전 raw) → chain 전 = blend 직전 raw GBDT
    # 지시서: Δ_체인전은 TASK-P2 리포트의 체인 전 3-seed 평균. P2의 체인 전 = raw GBDT 2024 홀드아웃.
    d_pre = []
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(P2DIR, f"pred_P2_2024_seed{s}.parquet")).sort_values("t").reset_index(drop=True)
        p_raw = df[["pred_0", "pred_1", "pred_2"]].values.astype(np.float64)
        y = df[["y_0", "y_1", "y_2"]].values.astype(np.float64)
        w0_raw = None
        d_pre.append(cf.total_metric(y, p_raw)[0] - cf.total_metric(y, cf.load_metricw_pred("W0", s)[0])[0])
    m_pre = float(np.mean(d_pre))
    print(f"Δ_체인전(P2) = {m_pre:+.5f} (raw GBDT vs W0 raw)")
    r_dup = 1 - m_p2 / m_pre if abs(m_pre) > 1e-9 else float("nan")
    print(f"중복률 r(P2) = {r_dup*100:.1f}%  (W2 참조: 149.7%)")

    # 단계별 trace (seed42)
    st = {k: results[f"P2_42"]["stages"][k] - results[f"W0_42"]["stages"][k] for k in
          ["blend", "iso", "shift", "m1", "coordBA"]}
    st_w2 = {k: results[f"W2_42"]["stages"][k] - results[f"W0_42"]["stages"][k] for k in
             ["blend", "iso", "shift", "m1", "coordBA"]}
    print(f"\n단계별 trace (chain(P2)−chain(W0), seed42):")
    for k in ["blend", "iso", "shift", "m1", "coordBA"]:
        print(f"  {k:8s} P2={st[k]:+.5f}  W2={st_w2[k]:+.5f}")

    # 그룹별 분해 (seed42)
    g_p2 = results["P2_42"]["group"]
    g_w0 = results["W0_42"]["group"]
    print(f"\n그룹별 분해 (P2 vs W0, seed42):")
    gsum = 0.0
    for g in range(3):
        dt = g_p2[g][0] - g_w0[g][0]
        dfi = g_p2[g][2] - g_w0[g][2]
        dnm = g_p2[g][1] - g_w0[g][1]
        contrib = (dfi - dnm) / 6
        gsum += contrib
        print(f"  G{g+1}: ΔTOTAL_g={dt:+.5f} ΔFICR={dfi:+.5f} ΔNMAE={dnm:+.5f} TOTAL기여={contrib:+.5f}")
    print(f"  Σ(1/6 희석 기여) = {gsum:+.5f}")

    out = dict(precheck=dict(w0_mean=w0_mean, w2_mean=w2_mean, pass_=bool(ok_w0 and ok_w2)),
               results=results,
               delta_chain_after=dict(P2=dict(mean=m_p2, per_seed=d_p2, npos=npos),
                                      W2=dict(mean=m_w2, per_seed=d_w2)),
               delta_chain_before=dict(P2_mean=m_pre, note="raw GBDT vs W0 raw, 3seed"),
               dup_rate=dict(P2_pct=float(r_dup*100), W2_pct=149.7),
               stage_trace_seed42=dict(P2=st, W2=st_w2),
               group_decomp_seed42=dict(P2_sum_contrib=gsum))
    with open(os.path.join(HERE, "p2chain_results.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n저장: p2chain_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
