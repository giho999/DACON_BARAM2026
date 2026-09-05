#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2 — 마스크 확률 가중 재학습 (TASK-RETROHARNESS 후속, coordBA 스케일 4-프로브 중 P2)
==============================================================================
가설: METRICW W2가 "2024 실측 마스크(0/1)"에 가중을 맞춰 전이 실패한 원인 = 마스크
      구조가 2025에서 다름. NWP로 추정한 P(y≥0.1C) **확률** 가중은 2025 마스크 구조에
      강건해야 한다 (확률 = 하드 경계 없음, 저마스크 행을 0이 아니라 감소 가중).

학습 가중 (그룹 g, cap C_g):
  w_i = P(y_i ≥ 0.1C_g | NWP_i)   — 연도전이 AUC 0.94~0.96 실증 (2022-23→2024)
  대조 W0(균등)와 동일 학습 파이프라인, 가중만 교체. v7 GBDT 4-앙상블, seed 42.

파이프라인: 2022-23 학습(가중=마스크확률) → 2024 홀드아웃 예측 + 2025 test 예측
          → 동결 체인(iso/shift/M1/coordBA, w2chain/frozen_constants) 적용 → 제출 후보.
출력: work_a_package/metricw_p2/pred_P2_2024.parquet · pred_test_P2_2025.parquet
      submission_P2_maskprob.csv (2025 test, 동결 체인)
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)  # work_a_package의 상위 = 프로젝트 루트
OUT = os.path.join(HERE, "metricw_p2")
os.makedirs(OUT, exist_ok=True)

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
             objective="reg:squarederror", tree_method="hist", n_jobs=16)
CB_M = dict(iterations=317, depth=6, learning_rate=0.03, l2_leaf_reg=5.0,
            loss_function="RMSE", thread_count=16, verbose=0)
SEED = 42  # CLI --seed로 override (TASK-P2CHAIN 3-seed용)


def make_models():
    ms = [("lgb", lgb.LGBMRegressor(**dict(LGB_M, random_state=SEED)))]
    try:
        from xgboost import XGBRegressor
        ms.append(("xgb", XGBRegressor(**dict(XGB_M, random_state=SEED))))
    except Exception as e:
        print(f"  [warn] xgboost: {e}")
    try:
        from catboost import CatBoostRegressor
        ms.append(("cat", CatBoostRegressor(**dict(CB_M, random_seed=SEED))))
    except Exception as e:
        print(f"  [warn] catboost: {e}")
    ms.append(("lgb_l1", lgb.LGBMRegressor(**dict(LGB_M, random_state=SEED, objective="regression_l1"))))
    return ms


def mask_prob_features():
    tr = pd.read_parquet(os.path.join(PROJ, "cache_train.parquet"))
    tr["t"] = pd.to_datetime(tr["t"])
    te = pd.read_parquet(os.path.join(PROJ, "cache_test.parquet"))
    te["t"] = pd.to_datetime(te["t"])
    nwp = [c for c in tr.columns if (c.startswith("ldaps_") or c.startswith("gfs_"))
           and tr[c].dtype != "O" and c in te.columns]
    timef = [c for c in tr.columns if c in ("hour", "month", "day_of_year", "hour_sin", "hour_cos")
             and c in te.columns]
    feats = nwp + timef
    med = tr[feats].median()
    return tr, te, feats, med


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, choices=[42, 1337, 2024])
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    t0 = time.time()
    tr, te, feats, med = mask_prob_features()
    Xtr_all = tr[feats].fillna(med)
    Xte = te[feats].fillna(med)
    p_2025 = np.zeros((len(te), 3))

    # 1) 마스크 확률 (전체 학습 2022-24 → 학습 행 + 2025 test) — t 기준으로 정렬 저장
    tr_all = pd.read_parquet(os.path.join(PROJ, "cache_train.parquet"))
    tr_all["t"] = pd.to_datetime(tr_all["t"])
    tr_all = tr_all.sort_values("t").reset_index(drop=True)
    p_map = {}
    for g in range(3):
        y = tr_all[f"kpx_group_{g+1}"].values
        mask = (y >= 0.1 * CAPS[g]).astype(int)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 n_jobs=16, verbose=-1, random_state=SEED)
        clf.fit(Xtr_all, mask)
        p_map[g] = pd.Series(clf.predict_proba(Xtr_all)[:, 1], index=tr_all["t"])
        p_2025[:, g] = clf.predict_proba(Xte)[:, 1]
    print(f"[P2] 마스크 확률 완료 ({time.time()-t0:.0f}s)")

    # 2) 2022-23 학습 (가중=마스크확률) → 2024 홀드아웃 + 2025 test 예측
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
    tm = tr["t"] < pd.Timestamp(TRAIN_END)
    em = tr["t"] >= pd.Timestamp(TRAIN_END)
    med_f = tr.loc[tm, FEAT].median()
    Xt = tr.loc[tm, FEAT].fillna(med_f)
    Xe = tr.loc[em, FEAT].fillna(med_f)

    te["t"] = pd.to_datetime(te["t"])
    te = te.sort_values("t").reset_index(drop=True)
    te["forecast_horizon_h"] = ((te["t"].dt.hour - 13) % 24 + 12).astype(float)
    Xte_f = te[FEAT].fillna(med_f)

    pred_2024 = np.zeros((em.sum(), 3))
    pred_2025 = np.zeros((len(te), 3))
    for g in range(3):
        y = tr.loc[tm, TARGETS[g]].values
        w = p_map[g].reindex(tr.loc[tm, "t"]).values  # t 정렬 마스크확률
        ms = make_models()
        for name, m in ms:
            m.fit(Xt, y, sample_weight=w)
        pred_2024[:, g] = np.clip(np.mean([m.predict(Xe) for name, m in ms], axis=0), 0, CAPS[g])
        pred_2025[:, g] = np.clip(np.mean([m.predict(Xte_f) for name, m in ms], axis=0), 0, CAPS[g])
        print(f"  G{g+1} 완료 ({time.time()-t0:.0f}s)")
        print(f"    가중: mean={w.mean():.3f} min={w.min():.3f} max={w.max():.3f}")

    out24 = pd.DataFrame({"t": tr.loc[em, "t"].values,
                          "pred_0": pred_2024[:, 0], "pred_1": pred_2024[:, 1], "pred_2": pred_2024[:, 2],
                          "y_0": tr.loc[em, TARGETS[0]].values,
                          "y_1": tr.loc[em, TARGETS[1]].values,
                          "y_2": tr.loc[em, TARGETS[2]].values})
    out24.to_parquet(os.path.join(OUT, f"pred_P2_2024_seed{SEED}.parquet"), index=False)
    out25 = pd.DataFrame({"t": te["t"].values,
                          "pred_0": pred_2025[:, 0], "pred_1": pred_2025[:, 1], "pred_2": pred_2025[:, 2]})
    out25.to_parquet(os.path.join(OUT, f"pred_test_P2_2025_seed{SEED}.parquet"), index=False)
    print(f"[P2 seed{SEED}] 2024 홀드아웃 + 2025 test 예측 저장 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
