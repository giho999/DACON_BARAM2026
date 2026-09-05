#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-W2NOSHIFT — W2 × shift 캘리 계수 스케일링 결정실험
=========================================================
지시서: opencode_TASK_W2NOSHIFT_20260809.md · 선등록: PREREG_w2noshift_20260809.md

chain_frozen.py 재사용 (동결 상수·CNN·blend). 재학습 0회. iso/M1/coordBA/블렌드/CNN 불변.
변형 (shift 계수 승수만 변경, 그 외 탐색 금지):
  V1: [0.0, 0.0, 0.0]  (완전 제거)
  V2: [0.5, 0.5, 0.5]  (절반)
  V3: [1.0, 0.0, 1.0]  (G2만 제거, G1/G3 원래 유지)
측정: 2024 연도전이 홀드아웃(2022-23 학습) · 대회 평가식 원본 · 3seed.
  전구간(01-12) + 비겹침 부분구간(01-08, cal 09-10 적합 창과 안 겹침) 별도 산출.
대조: chain(W0) 원본, chain(W2) 원본 (W2CHAIN 결과 0.63402 / 0.62524).
단계별 trace: blend→iso→shift→M1→coordBA (seed42, 전 변형).
"""
import os, sys, json, pickle, importlib.util, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import chain_frozen as cf

SEEDS = [42, 1337, 2024]
VARIANTS = {"V1": [0.0, 0.0, 0.0], "V2": [0.5, 0.5, 0.5], "V3": [1.0, 0.0, 1.0]}
NON_OVERLAP = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-09-01"))  # cal 09-10 미포함

def load_const():
    with open(os.path.join(HERE, "frozen_constants.pkl.isos"), "rb") as f:
        isos = pickle.load(f)
    with open(os.path.join(HERE, "frozen_constants.pkl"), "rb") as f:
        c = pickle.load(f)
    return dict(isos=isos, **c)


def eval_segment(y3, p3, t3, seg):
    m = (t3 >= seg[0]) & (t3 < seg[1])
    return cf.total_metric(y3[m], p3[m])


def main():
    const = load_const()
    results = {}
    traces = {}
    for vid, scale in VARIANTS.items():
        results[vid] = {"scale": scale, "seeds": {}}
        for seed in SEEDS:
            p, y, t = cf.load_metricw_pred("W2", seed)
            stages = {}
            chained = cf.apply_chain(p, const, t, shift_scale=scale, stages_out=stages)
            full = cf.total_metric(y, chained)
            seg = eval_segment(y, chained, pd.to_datetime(t), NON_OVERLAP)
            results[vid]["seeds"][str(seed)] = dict(TOTAL=full[0], NMAE1=full[1],
                                                    FICR=full[2], seg_TOTAL=seg[0])
            if seed == 42:
                traces[vid] = {k: cf.total_metric(y, v)[0] for k, v in stages.items()}
            print(f"[{vid}] seed{seed}: TOTAL={full[0]:.5f} (seg01-08={seg[0]:.5f})")
        fs = [results[vid]["seeds"][str(s)]["TOTAL"] for s in SEEDS]
        ss = [results[vid]["seeds"][str(s)]["seg_TOTAL"] for s in SEEDS]
        results[vid]["mean"] = dict(TOTAL=float(np.mean(fs)), seg_TOTAL=float(np.mean(ss)))
        results[vid]["per_seed_TOTAL"] = fs
        results[vid]["per_seed_seg"] = ss
    results["traces_seed42"] = traces
    with open(os.path.join(HERE, "w2noshift_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n=== 대조 (W2CHAIN) ===")
    print("chain(W0) = 0.63402 · chain(W2) = 0.62524")
    print("\n=== 요약 ===")
    for vid in VARIANTS:
        r = results[vid]["mean"]
        d_full = r["TOTAL"] - 0.63402
        d_seg = r["seg_TOTAL"] - 0.62524
        fs = results[vid]["per_seed_TOTAL"]
        npos = sum(1 for x in fs if x - 0.63402 > 0)
        print(f"{vid}: TOTAL={r['TOTAL']:.5f} (ΔvsW0 {d_full:+.5f}, 양수 {npos}/3) "
              f"seg01-08={r['seg_TOTAL']:.5f} (ΔvsW2원본 {d_seg:+.5f})")
    print("\n=== 단계별 trace (seed42) ===")
    print(f"{'':6s} {'blend':>8s} {'iso':>8s} {'shift':>8s} {'m1':>8s} {'coordBA':>8s}")
    for vid, tr in traces.items():
        print(f"{vid:6s} " + " ".join(f"{tr[k]:8.5f}" for k in ["blend", "iso", "shift", "m1", "coordBA"]))


if __name__ == "__main__":
    main()
