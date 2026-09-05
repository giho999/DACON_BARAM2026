#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-RETROHARNESS — retro_eval: 하네스 5종 역검정 + 널대조
============================================================
정답지(truthset.json)의 재현 가능 후보 4종에 대해 2024 대응 예측(cands_2024.npz)을
사용해 V_A~V_E의 Δ추정을 산출하고, LB 라벨과 부호 정확도·Spearman·오탐률을 비교.
널대조: ① 상수 예측(−0.005) ② 셔플 Spearman 1000회.
"""
import os, sys, json, importlib.util, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness_variants as hv
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEEDS = [42, 1337, 2024]


def load_all():
    d = np.load(os.path.join(HERE, "cands_2024.npz"), allow_pickle=True)
    y = d["y"]
    t = pd.to_datetime(d["t"].astype("datetime64[ns]"))
    cands = {k: d[k] for k in d.files if k not in ("y", "t")}
    return cands, y, t


def adv_weights(t_hold, seed=42):
    """V_C 적대적 가중: 2024/2025 LGB 분류기 p → w=p/(1-p), clip[0.05,20], Σw=n.
    t_hold: 2024 홀드아웃 타임스탬프 (8778) — cache_train의 2024 행과 정렬."""
    tr = pd.read_parquet(os.path.join(PROJ, "cache_train.parquet"))
    tr["t"] = pd.to_datetime(tr["t"])
    te = pd.read_parquet(os.path.join(PROJ, "cache_test.parquet"))
    te["t"] = pd.to_datetime(te["t"])
    tr24 = tr[(tr["t"] >= "2024-01-01") & (tr["t"] < "2025-01-01")].reset_index(drop=True)
    nwp = [c for c in tr24.columns if (c.startswith("ldaps_") or c.startswith("gfs_"))
           and tr24[c].dtype != "O" and c in te.columns]
    X = pd.concat([tr24[nwp], te[nwp]], axis=0)
    yy = np.concatenate([np.zeros(len(tr24)), np.ones(len(te))])
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                           n_jobs=16, verbose=-1, random_state=seed)
    m.fit(X, yy)
    p24 = m.predict_proba(X)[:, 1][: len(tr24)]
    # tr24 → 홀드아웃 t 정렬 (둘 다 2024 전체, t 매칭)
    w_full = np.clip(p24 / (1 - p24), 0.05, 20.0)
    w_ser = pd.Series(w_full, index=tr24["t"]).reindex(pd.to_datetime(t_hold)).fillna(1.0).values
    w_ser = w_ser / w_ser.sum() * len(w_ser)
    return w_ser, m


def wind_quantile_weights(t_hold):
    """V_D: 2024 행을 2025 풍속 분위에 정합 — g5 ws 분위로 재표집 가중."""
    tr = pd.read_parquet(os.path.join(PROJ, "cache_train.parquet"))
    tr["t"] = pd.to_datetime(tr["t"])
    te = pd.read_parquet(os.path.join(PROJ, "cache_test.parquet"))
    te["t"] = pd.to_datetime(te["t"])
    tr24 = tr[(tr["t"] >= "2024-01-01") & (tr["t"] < "2025-01-01")].reset_index(drop=True)
    col = "ldaps_ws_mean_g5"
    v24 = tr24[col].values
    v25 = te[col].dropna().values
    qs = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    q24 = np.quantile(v24, qs)
    q25 = np.quantile(v25, qs)
    # 각 2024 행 → 분위 구간, 2025 목표 밀도/2024 밀도 비율로 가중
    edges = np.unique(q24)
    w = np.ones(len(v24))
    for i in range(len(edges)):
        lo = edges[i]
        hi = edges[i + 1] if i + 1 < len(edges) else np.inf
        sel = (v24 >= lo) & (v24 < hi)
        if sel.sum() == 0:
            continue
        dens24 = sel.sum() / len(v24)
        dens25 = ((v25 >= lo) & (v25 < hi)).sum() / len(v25)
        w[sel] = max(dens25 / max(dens24, 1e-9), 1e-3)
    w_ser = pd.Series(w, index=tr24["t"]).reindex(pd.to_datetime(t_hold)).fillna(1.0).values
    w_ser = w_ser / w_ser.sum() * len(w_ser)
    return w_ser


def main():
    cands, y, t = load_all()
    truth = json.load(open(os.path.join(HERE, "truthset.json")))["candidates"]
    # 재현 가능 후보만 (2024 대응본 존재)
    names = {"C_g3tail": "work_a_package/submission_coordBA_g3tail_l120.csv",
             "C_m1v2": "submission_A_m1v2.csv",
             "C_band": "submission_m1_bandcorrected.csv",
             "C_g3shift": "submission_A_g3shift03.csv"}
    base = cands["base"]

    # V_C 가중 (3-seed 평균), V_D 가중
    ws = []
    for s in SEEDS:
        w, m = adv_weights(t, seed=s)
        ws.append(w)
    w_adv = np.mean(ws, axis=0)
    w_wind = wind_quantile_weights(t)
    aucs = []
    for s in SEEDS:
        _, m = adv_weights(t, seed=s)
        tr = pd.read_parquet(os.path.join(PROJ, "cache_train.parquet"))
        tr["t"] = pd.to_datetime(tr["t"])
        te = pd.read_parquet(os.path.join(PROJ, "cache_test.parquet"))
        te["t"] = pd.to_datetime(te["t"])
        tr24 = tr[(tr["t"] >= "2024-01-01") & (tr["t"] < "2025-01-01")].reset_index(drop=True)
        nwp = [c for c in tr24.columns if (c.startswith("ldaps_") or c.startswith("gfs_"))
               and tr24[c].dtype != "O" and c in te.columns]
        X = pd.concat([tr24[nwp], te[nwp]], axis=0)
        yy = np.concatenate([np.zeros(len(tr24)), np.ones(len(te))])
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(yy, m.predict_proba(X)[:, 1]))
    auc_mean = float(np.mean(aucs))

    # 하네스 Δ 추정 행렬
    results = {}
    for cname, truth_file in names.items():
        cand = cands[cname]
        label = next(c["label"] for c in truth if c["file"] == truth_file)
        d = dict(label=label)
        d["V_A_rolling"] = hv.harness_VA_rolling(y, t, base, cand)
        d["V_B_yeartransfer"] = hv.harness_VB_yeartransfer(y, t, base, cand)
        d["V_C_adversarial"] = hv.harness_VC_adversarial(y, t, base, cand, w_adv)
        d["V_D_windquantile"] = hv.harness_VD_windquantile(y, t, base, cand, w_wind)
        d["V_E_multiyear"] = hv.harness_VE_multiyear(y, t, base, cand)
        results[cname] = d

    # 방식별 지표
    methods = ["V_A_rolling", "V_B_yeartransfer", "V_C_adversarial", "V_D_windquantile", "V_E_multiyear"]
    labels = np.array([results[c]["label"] for c in names])
    print("=" * 80)
    print(f"정답지 재현가능 n={len(names)} · V_C AUC(3seed 평균)={auc_mean:.4f}")
    print(f"{'방식':20s} {'부호정확':>8s} {'Spearman':>9s} {'p값':>7s} {'치명오탐':>8s}")
    summary = {}
    for m in methods:
        est = np.array([results[c][m] for c in names])
        sign_acc = float(np.mean(np.sign(est) == np.sign(labels)))
        rho, pv = spearmanr(est, labels)
        rho = float(rho) if np.isfinite(rho) else 0.0
        fp = float(np.mean((est >= 0.002) & (labels <= -0.002)))
        summary[m] = dict(sign_acc=sign_acc, spearman=rho, p=float(pv), fp_rate=fp)
        print(f"{m:20s} {sign_acc:8.3f} {rho:+9.3f} {pv:7.4f} {fp:8.3f}")
    # 널대조
    null_sacc = float(np.mean(np.full(len(labels), -0.005) < 0 if (labels < 0).all() else np.sign(np.full(len(labels), -0.005)) == np.sign(labels)))
    shuff_rhos = []
    rng = np.random.RandomState(0)
    for _ in range(1000):
        shuf = rng.permutation(labels)
        rr, _ = spearmanr(shuf, labels)
        shuff_rhos.append(float(rr))
    shuff_rhos = np.array(shuff_rhos)
    print(f"\n널(상수 −0.005): 부호정확도 {null_sacc:.3f}")
    print(f"널(셔플) Spearman: 2.5~97.5% = [{np.percentile(shuff_rhos,2.5):+.3f}, {np.percentile(shuff_rhos,97.5):+.3f}]")

    # 판정
    verdict = {}
    for m in methods:
        s = summary[m]
        ok = (s["sign_acc"] >= 0.80 and s["spearman"] >= 0.60 and s["p"] < 0.05 and s["fp_rate"] == 0.0)
        # 널 초과: 97.5% 분위 상한보다 위여야 (양수 방향)
        above_null = s["spearman"] > np.percentile(shuff_rhos, 97.5)
        verdict[m] = dict(pass_thresholds=bool(ok), above_null=bool(above_null),
                          pass_all=bool(ok and above_null))
    print("\n판정:")
    for m in methods:
        v = verdict[m]
        print(f"  {m}: 문턱PASS={v['pass_thresholds']} 널초과={v['above_null']} → {'합격' if v['pass_all'] else '불합격'}")

    out = dict(auc_mean=auc_mean, results=results, summary=summary,
               null=dict(const_sacc=null_sacc, shuffle_ci=[float(np.percentile(shuff_rhos,2.5)),
                                                           float(np.percentile(shuff_rhos,97.5))]),
               verdict=verdict, wind_quantiles=dict())
    with open(os.path.join(HERE, "retro_results.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n저장: retro_results.json")


if __name__ == "__main__":
    main()
