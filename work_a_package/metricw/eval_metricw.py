#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-METRICW — 평가 (대회 평가식 원본 + 연도전이 홀드아웃 + CF 질량비)
======================================================================
지시서: opencode_TASK_METRICW_20260809.md §1-3/1-4

- 평가 지표: 대회 평가식 원본. pipeline_v2/metrics.py의 competition_metric은
  EXACT DACON 구현(10% 마스크 + 6%/8% 밴드 + 발전량가중)으로,
  m1_holdout_gate.py::score_fn과 수식 동일 — 여기서는 그 수식을 직접 인라인하여
  외부 모듈 의존 없이 재현 (TOTAL = 3그룹 평균, 1-NMAE·FICR도 그룹평균).
- 홀드아웃: 2022-23 학습 → 2024 전체 (연도전이). 캘리브레이션 없음 (가중 효과의
  순수 측정 — 캘리는 2024 내부 적합 상수로 게이트 오염 방지).
- 3-seed 평균 ΔTOTAL (vs W0) + seed 양수 개수.
- W2-shuffle(널)과 비교.
- CF 질량비: pred/train-actual 분포 점유율 대조 (G3TAIL 리포트 §3과 동일 정의:
  예측·실측 각각 자체 CF≥0.10cap 행, 빈 = 자체 cf).
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
CAPS = [21600.0, 21600.0, 21000.0]
TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
SEEDS = [42, 1337, 2024]
BINS = [(0.10, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 0.95), (0.95, 1.10)]


def group_metric(y, p, cap):
    """대회 평가식 (단일 그룹): 10% 마스크 + 6%/8% 밴드 + 발전량가중.
    returns (TOTAL_g, 1-NMAE_g, FICR_g)"""
    y = np.asarray(y, float); p = np.asarray(p, float)
    v = (~np.isnan(y)) & (~np.isnan(p)) & (y >= 0.1 * cap)
    yt, pt = y[v], p[v]
    if len(yt) == 0:
        return 0.0, 0.0, 0.0
    nmae = float(np.clip(np.mean(np.abs(yt - pt)) / cap, 0, 1))
    e = np.abs(yt - pt) / cap
    up = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
    fi = float(np.clip((yt * up).sum() / max((yt * 4.0).sum(), 1e-9), 0, 1))
    return 0.5 * (1 - nmae) + 0.5 * fi, 1.0 - nmae, fi


def total_metric(y3, p3):
    """3그룹 평균 (TOTAL, 1-NMAE, FICR)"""
    t = [group_metric(y3[:, g], p3[:, g], CAPS[g]) for g in range(3)]
    return tuple(float(np.mean([x[g] for x in t])) for g in range(3))


def load_pred(variant, seed):
    df = pd.read_parquet(os.path.join(HERE, f"pred_{variant}_seed{seed}.parquet"))
    df = df.sort_values("t").reset_index(drop=True)
    return df[["pred_0", "pred_1", "pred_2"]].values, df[["y_0", "y_1", "y_2"]].values


def mass_ratio_table(preds, y_all, cap):
    """CF 구간별 질량비 (예측/실측 분포 점유율, masked 자체 CF≥0.10).
    preds/y_all는 2024 홀드아웃 (y는 실측). 빈 = 실제(y) cf 기준으로 균일 대조."""
    pc = preds / cap
    yc = y_all / cap
    pm = preds[pc >= 0.10]
    ym = y_all[yc >= 0.10]
    pcm, ycm = pc[pc >= 0.10], yc[yc >= 0.10]
    out = []
    for a, b in BINS:
        sp = pm[(pcm >= a) & (pcm < b)].sum()
        sa = ym[(ycm >= a) & (ycm < b)].sum()
        out.append(sp / sa if sa > 0 else float("nan"))
    return out


def main():
    res = {}
    for variant in ["W0", "W1", "W2", "W2S"]:
        tots, nms, fis = [], [], []
        seeds_use = [42] if variant == "W2S" else SEEDS  # W2S(널)는 seed42 1개 (선등록)
        for seed in seeds_use:
            p, y = load_pred(variant, seed)
            t, nm, fi = total_metric(y, p)
            tots.append(t); nms.append(nm); fis.append(fi)
        res[variant] = dict(
            seeds={str(s): dict(TOTAL=t, NMAE1=n, FICR=f)
                   for s, (t, n, f) in zip(seeds_use, zip(tots, nms, fis))},
            mean=dict(TOTAL=float(np.mean(tots)), NMAE1=float(np.mean(nms)),
                      FICR=float(np.mean(fis))),
            per_seed_TOTAL=[float(x) for x in tots])

    # Δ vs W0 (3-seed)
    w0 = res["W0"]["per_seed_TOTAL"]
    for variant in ["W1", "W2"]:
        d = [res[variant]["per_seed_TOTAL"][i] - w0[i] for i in range(3)]
        res[variant]["delta_vs_W0"] = dict(mean=float(np.mean(d)), per_seed=d,
                                           n_positive=int(sum(1 for x in d if x > 0)))
    # W2 vs W2S(널)
    dnull = [res["W2"]["per_seed_TOTAL"][0] - res["W2S"]["per_seed_TOTAL"][0]]
    res["W2_vs_W2S"] = dict(mean=float(np.mean(dnull)), per_seed=dnull,
                            n_positive=int(sum(1 for x in dnull if x > 0)))

    # CF 질량비 (W0/W1/W2 — seed42 대표, 2024 홀드아웃 예측 vs 실측)
    tl = pd.read_csv(os.path.join(PROJ, "data/train_labels.csv"))
    tl.columns = [c.strip() for c in tl.columns]
    mass = {}
    for variant in ["W0", "W1", "W2", "W2S"]:
        p, y = load_pred(variant, 42)
        mass[variant] = {TARGETS[g]: mass_ratio_table(p[:, g], y[:, g], CAPS[g])
                         for g in range(3)}
    res["mass_ratio_seed42"] = mass

    # 가중 통계 (wstats json 병합)
    ws = {}
    for variant in ["W0", "W1", "W2", "W2S"]:
        for seed in SEEDS:
            fp = os.path.join(HERE, f"wstats_{variant}_seed{seed}.json")
            if os.path.exists(fp):
                ws[f"{variant}_seed{seed}"] = json.load(open(fp))["groups"]
    res["wstats"] = ws

    with open(os.path.join(HERE, "metricw_results.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # 출력 요약
    print("=== 연도전이 홀드아웃 (2022-23→2024), 3-seed 평균 ===")
    for variant in ["W0", "W1", "W2", "W2S"]:
        m = res[variant]["mean"]
        print(f"  {variant:4s}: TOTAL={m['TOTAL']:.5f}  1-NMAE={m['NMAE1']:.5f}  FICR={m['FICR']:.5f}")
    for variant in ["W1", "W2"]:
        d = res[variant]["delta_vs_W0"]
        print(f"  Δ{variant} vs W0: mean={d['mean']:+.5f}  per_seed={[f'{x:+.5f}' for x in d['per_seed']]}  양수 {d['n_positive']}/3")
    d = res["W2_vs_W2S"]
    print(f"  ΔW2 vs W2S(널): mean={d['mean']:+.5f}  per_seed={[f'{x:+.5f}' for x in d['per_seed']]}  양수 {d['n_positive']}/3")
    print("\n=== CF 질량비 (seed42, 2024 홀드아웃) — [0.10,0.30) [0.30,0.50) [0.50,0.70) [0.70,0.85) [0.85,0.95) [0.95,1.10)")
    for variant in ["W0", "W1", "W2"]:
        for g in range(3):
            print(f"  {variant} {TARGETS[g]}: " + " ".join(f"{v:.2f}" for v in mass[variant][TARGETS[g]]))
    print("\nJSON: metricw_results.json")


if __name__ == "__main__":
    main()
