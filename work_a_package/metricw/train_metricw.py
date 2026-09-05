#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-METRICW — 평가지표 정합 표본가중 재학습 (W0/W1/W2)
========================================================
지시서: opencode_TASK_METRICW_20260809.md

- base GBDT 파이프라인: v7 계보 (m1_gen_holdout_preds.py 구조 — cache_train.parquet,
  4-GBDT 앙상블 LGB+XGB+Cat+LGB-L1). pipeline_v2 계보 사용 금지.
- 변경: 학습 표본가중만 교체. 피처·하이퍼파라미터·시드 일절 변경 금지.
- 결정성: XGB/CatBoost를 CPU로 고정 (m1_gen_holdout_preds의 GPU 설정 제외 — device는
  하이퍼파라미터가 아니며 §3-6 결정성 요구가 우선). 시드 고정 → 2회 실행 md5 일치.

가중 (그룹 g, cap C_g, 학습 세트 내에서 계산):
  mask_i = y_i >= 0.1*C_g
  W0: w_i = 1.0                       (전 행 균등, 컨트롤)
  W1: w_i = 1(mask)                   (마스크만)
  W2: w_i = mask * (0.5/n + 0.5*y_i/Σy)  (§0-2 정식, n=마스크 내 행수)
  W2S: W2 가중을 고정 시드로 행 셔플 (널대조)

파이프라인: 2022-23 학습 → 2024 전체 예측 (연도전이 홀드아웃). G3 2022 NaN은 dropna로 제외.

CLI: python3 train_metricw.py --variant {W0,W1,W2,W2S} --seed {42,1337,2024}
출력: work_a_package/metricw/pred_{variant}_seed{seed}.parquet (t, pred_0/1/2, y_0/1/2)
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # /home/gpu_04/DACON_baram2026

TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPS = [21600.0, 21600.0, 21000.0]
CLEAN_COLS = ["clean_1", "clean_2", "clean_3"]
TRAIN_END = "2024-01-01"

LGB_M = dict(n_estimators=317, max_depth=6, learning_rate=0.03, num_leaves=64,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
             reg_alpha=0.1, objective="regression", n_jobs=16, verbose=-1,
             deterministic=True)
XGB_M = dict(n_estimators=317, max_depth=6, learning_rate=0.03, subsample=0.8,
             colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=0.1,
             objective="reg:squarederror", tree_method="hist", n_jobs=16)  # CPU 고정 (결정성)
CB_M = dict(iterations=317, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
            loss_function="RMSE", thread_count=16, verbose=0)  # CPU 고정 (결정성)


def make_models(seed):
    ms = [("lgb", lgb.LGBMRegressor(**dict(LGB_M, random_state=seed)))]
    try:
        from xgboost import XGBRegressor
        ms.append(("xgb", XGBRegressor(**dict(XGB_M, random_state=seed))))
    except Exception as e:
        print(f"  [warn] xgboost unavailable: {e}")
    try:
        from catboost import CatBoostRegressor
        ms.append(("cat", CatBoostRegressor(**dict(CB_M, random_seed=seed))))
    except Exception as e:
        print(f"  [warn] catboost unavailable: {e}")
    ms.append(("lgb_l1", lgb.LGBMRegressor(**dict(LGB_M, random_state=seed, objective="regression_l1"))))
    return ms


def build_weights(y, cap, variant, rng):
    """학습 세트 내 가중 벡터. variant ∈ {W0, W1, W2, W2S}"""
    mask = (y >= 0.10 * cap).astype(float)
    if variant == "W0":
        w = np.ones_like(y, dtype=float)
    elif variant == "W1":
        w = mask
    elif variant == "W2":
        n = mask.sum()
        if n == 0:
            return np.zeros_like(y, dtype=float)
        w = mask * (0.5 / n + 0.5 * y / max(y[mask > 0].sum(), 1e-9))
        w = w / w.sum() * len(y)  # 합=n 정규화 — 상대 비율만 의미(2.6:1), 1e-4 절대 스케일은 lightgbm에서 손상
    elif variant == "W2S":  # W2 가중을 행 셔플 (널대조) — 가중 분포는 유지, y-연관만 파괴
        n = mask.sum()
        if n == 0:
            return np.zeros_like(y, dtype=float)
        w2 = mask * (0.5 / n + 0.5 * y / max(y[mask > 0].sum(), 1e-9))
        w = w2[rng.permutation(len(w2))]
        w = w / w.sum() * len(y)  # 합=n 정규화 (위와 동일)
    else:
        raise ValueError(f"unknown variant {variant}")
    return w


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["W0", "W1", "W2", "W2S"])
    ap.add_argument("--seed", type=int, required=True, choices=[42, 1337, 2024])
    ap.add_argument("--test", action="store_true",
                    help="2025 test(cache_test.parquet) 예측도 생성 — 동일 학습 모델, 예측만 test로")
    args = ap.parse_args()
    variant, SEED = args.variant, args.seed
    rng = np.random.RandomState(SEED)
    t0 = time.time()

    tr = pd.read_parquet(os.path.join(PROJ, "cache_train.parquet"))
    tr["t"] = pd.to_datetime(tr["t"])
    tr = tr.sort_values("t").dropna(subset=TARGETS).reset_index(drop=True)
    tr["forecast_horizon_h"] = ((tr["t"].dt.hour - 13) % 24 + 12).astype(float)
    tr = tr.drop(columns=[c for c in CLEAN_COLS if c in tr.columns])
    cl = pd.read_csv(os.path.join(PROJ, "cleaned_train_v3.csv"),
                     usecols=["forecast_kst_dtm"] + CLEAN_COLS)
    cl["t"] = pd.to_datetime(cl["forecast_kst_dtm"])
    tr = tr.merge(cl[["t"] + CLEAN_COLS], on="t", how="left")
    tr[CLEAN_COLS] = tr[CLEAN_COLS].fillna(1)

    excl = {"t"} | set(TARGETS) | set(CLEAN_COLS)
    FEAT = sorted([c for c in tr.columns if c not in excl and tr[c].dtype != "O"])
    print(f"[{variant} seed{SEED}] 피처 {len(FEAT)}개 · 행 {len(tr)} (캐시 로드 {time.time()-t0:.0f}s)")

    tm = tr["t"] < pd.Timestamp(TRAIN_END)
    em_all = tr["t"] >= pd.Timestamp(TRAIN_END)
    med = tr.loc[tm, FEAT].median()
    Xt = tr.loc[tm, FEAT].fillna(med)
    Xe_all = tr.loc[em_all, FEAT].fillna(med)

    if args.test:
        te = pd.read_parquet(os.path.join(PROJ, "cache_test.parquet"))
        te["t"] = pd.to_datetime(te["t"])
        te = te.sort_values("t").reset_index(drop=True)
        te["forecast_horizon_h"] = ((te["t"].dt.hour - 13) % 24 + 12).astype(float)
        Xte = te[FEAT].fillna(med)
        pred_test = np.zeros((len(te), 3))

    wstats = {}
    pred_all = np.zeros((em_all.sum(), 3))
    for g in range(3):
        y = tr.loc[tm, TARGETS[g]].values
        sw = build_weights(y, CAPS[g], variant, rng)
        ms = make_models(SEED)
        for name, m in ms:
            m.fit(Xt, y, sample_weight=sw)
        pred_all[:, g] = np.clip(np.mean([m.predict(Xe_all) for name, m in ms], axis=0), 0, CAPS[g])
        if args.test:
            pred_test[:, g] = np.clip(np.mean([m.predict(Xte) for name, m in ms], axis=0), 0, CAPS[g])
        msk = sw > 0
        wstats[TARGETS[g]] = dict(n_tr=int(len(y)), n_wpos=int(msk.sum()),
                                  frac_kept=float(msk.mean()),
                                  w_mean=float(sw.mean()), w_max=float(sw.max()),
                                  y_mean=float(y.mean()), y_mask_mean=float(y[msk].mean()) if msk.any() else 0.0)
        print(f"  G{g+1} 완료 ({time.time()-t0:.0f}s) kept={wstats[TARGETS[g]]['frac_kept']:.3f}")

    out = pd.DataFrame({"t": tr.loc[em_all, "t"].values,
                        "pred_0": pred_all[:, 0], "pred_1": pred_all[:, 1], "pred_2": pred_all[:, 2],
                        "y_0": tr.loc[em_all, TARGETS[0]].values,
                        "y_1": tr.loc[em_all, TARGETS[1]].values,
                        "y_2": tr.loc[em_all, TARGETS[2]].values})
    os.makedirs(HERE, exist_ok=True)
    op = os.path.join(HERE, f"pred_{variant}_seed{SEED}.parquet")
    out.to_parquet(op, index=False)
    if args.test:
        opt = os.path.join(HERE, f"pred_test_{variant}_seed{SEED}.parquet")
        pd.DataFrame({"t": te["t"].values,
                      "pred_0": pred_test[:, 0], "pred_1": pred_test[:, 1],
                      "pred_2": pred_test[:, 2]}).to_parquet(opt, index=False)
        print(f"[{variant} seed{SEED}] test 예측 저장 {opt}")
    with open(os.path.join(HERE, f"wstats_{variant}_seed{SEED}.json"), "w") as f:
        json.dump({"variant": variant, "seed": SEED, "groups": wstats}, f, ensure_ascii=False, indent=2)
    print(f"[{variant} seed{SEED}] 저장 {op}  (총 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
